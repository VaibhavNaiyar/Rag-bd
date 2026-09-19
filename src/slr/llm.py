"""One LLM provider behind a small protocol.

A turn makes at most two calls: decompose (or refine-plan) and synthesise.
There is no agent framework and no tool loop.

When no API key is configured the engine runs its deterministic offline
strategies instead (heuristic decomposition, extractive synthesis), so the
full pipeline — and ``make eval`` — works on a clean machine with no secrets.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import AsyncIterator
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from slr.config import Settings
from slr.telemetry.cost import UsageLedger, usage_of_stream

log = logging.getLogger(__name__)


class ChatModel(Protocol):
    name: str

    async def complete(
        self, system: str, user: str, *, ledger: UsageLedger, step: str, json_mode: bool = False, max_tokens: int = 600
    ) -> str: ...

    def stream(
        self, system: str, user: str, *, ledger: UsageLedger, step: str, max_tokens: int = 900
    ) -> AsyncIterator[str]: ...


class ModelUnavailable(RuntimeError):
    """The circuit is open: the model is not called, and the caller takes its offline path."""


def trips_breaker(exc: BaseException) -> bool:
    """Does this failure say something about the provider's health?

    A timeout, a dropped connection, a rate limit or a 5xx does: the next call
    is likely to fail the same way. A bad request or a rejected key does not:
    retrying it fails forever, but it is not an outage, and letting one
    malformed call open the circuit would take the model away from every turn.
    """
    import asyncio

    try:
        import openai
    except ImportError:  # pragma: no cover
        return isinstance(exc, (asyncio.TimeoutError, ConnectionError))
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500
    return isinstance(exc, (asyncio.TimeoutError, ConnectionError))


class CircuitBreaker:
    """Closed → open after ``threshold`` consecutive health failures → half-open
    after ``cooldown_s``: one trial call, which closes it on success and reopens
    it on failure.

    While it is open a turn does not wait out a timeout per model call; it goes
    straight to the offline strategy, which keeps time-to-first-token bounded
    when the provider is slow or rate-limiting. Single event loop, so no lock.
    """

    def __init__(self, threshold: int = 3, cooldown_s: float = 30.0, clock=time.monotonic) -> None:
        self.threshold = max(1, threshold)
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._trial = False

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        return "half_open" if self._clock() - self._opened_at >= self.cooldown_s else "open"

    def allow(self) -> bool:
        state = self.state
        if state == "closed":
            return True
        if state == "half_open" and not self._trial:
            self._trial = True  # exactly one call probes the provider
            return True
        return False

    def success(self) -> None:
        self._failures, self._opened_at, self._trial = 0, None, False

    def failure(self, exc: BaseException) -> None:
        if not trips_breaker(exc):
            self._trial = False
            return
        self._failures += 1
        if self._trial or self._failures >= self.threshold:
            if self._opened_at is None or self._trial:
                log.warning("LLM circuit opened after %s (%d consecutive)", type(exc).__name__, self._failures)
            self._opened_at = self._clock()
        self._trial = False


async def with_fallback(
    primary: AsyncIterator[str],
    fallback,
    on_failure,
) -> AsyncIterator[str]:
    """Stream ``primary``; if it fails, finish the turn another way.

    Nothing written yet: the whole answer comes from ``fallback()`` (the offline
    strategy for the same step). Something written already: keep it and stop,
    because a second answer glued onto half of the first would read as one.
    ``on_failure(exc, wrote_something)`` records which happened.
    """
    wrote = False
    try:
        async for delta in primary:
            wrote = True
            yield delta
        return
    except Exception as exc:  # noqa: BLE001 - any model failure degrades, never kills the turn
        log.warning("model stream failed (%s); %s", type(exc).__name__, "stopping" if wrote else "using the offline path")
        on_failure(exc, wrote)
        if wrote:
            return
    async for delta in fallback():
        yield delta


#: Reasoning models (o-series, gpt-5 family) spend completion tokens on hidden
#: reasoning before the answer, take no temperature, and cap output with
#: max_completion_tokens. Their "-chat" variants are ordinary chat models.
_REASONING = re.compile(r"^(o\d|gpt-5)(?!.*chat)")
REASONING_HEADROOM = 2000


class OpenAIChatModel:
    def __init__(
        self,
        model: str,
        base_url: str = "",
        timeout: float = 30.0,
        reasoning_effort: str = "low",
        breaker: CircuitBreaker | None = None,
    ):
        self._client_kwargs = {"timeout": timeout, "max_retries": 1}
        if base_url:
            self._client_kwargs["base_url"] = base_url
        #: one HTTP client per event loop: a client's connection pool belongs to the
        #: loop that opened it, and the eval harness runs one loop per corpus and arm
        self._clients: dict[int, Any] = {}
        self.name = model
        self.reasoning = bool(_REASONING.match(model))
        self.reasoning_effort = reasoning_effort
        self.breaker = breaker or CircuitBreaker()

    @property
    def _client(self):
        import asyncio

        from openai import AsyncOpenAI

        loop = id(asyncio.get_running_loop())
        if loop not in self._clients:
            self._clients[loop] = AsyncOpenAI(**self._client_kwargs)
        return self._clients[loop]

    async def aclose(self) -> None:
        """Close this loop's client while the loop can still run the close."""
        import asyncio

        client = self._clients.pop(id(asyncio.get_running_loop()), None)
        if client is not None:
            await client.close()

    def _admit(self, step: str) -> None:
        if not self.breaker.allow():
            raise ModelUnavailable(f"{step}: circuit open after repeated provider failures")

    def _limits(self, max_tokens: int) -> dict:
        if self.reasoning:
            return {"max_completion_tokens": max_tokens + REASONING_HEADROOM, "reasoning_effort": self.reasoning_effort}
        return {"temperature": 0, "max_tokens": max_tokens}

    async def complete(self, system, user, *, ledger, step, json_mode=False, max_tokens=600) -> str:
        self._admit(step)
        started = time.perf_counter()
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        try:
            resp = await self._client.chat.completions.create(
                model=self.name,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                **self._limits(max_tokens),
                **extra,
            )
        except Exception as exc:
            self.breaker.failure(exc)
            raise
        self.breaker.success()
        usage = resp.usage
        ledger.record_llm(
            step,
            self.name,
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
            (time.perf_counter() - started) * 1000,
        )
        return resp.choices[0].message.content or ""

    async def stream(self, system, user, *, ledger, step, max_tokens=900) -> AsyncIterator[str]:
        self._admit(step)
        started = time.perf_counter()
        readings = []
        completed = False
        written = 0
        try:
            stream = await self._client.chat.completions.create(
                model=self.name,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                **self._limits(max_tokens),
                stream=True,
                stream_options={"include_usage": True},
            )
            async for chunk in stream:
                if chunk.usage:
                    readings.append(chunk.usage)
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    written += len(chunk.choices[0].delta.content)
                    yield chunk.choices[0].delta.content
            completed = True
        except Exception as exc:
            self.breaker.failure(exc)
            raise
        finally:
            if completed:
                self.breaker.success()
            u = usage_of_stream(readings)
            if not readings:
                # Stopped before the provider's usage chunk (the engine dropped a
                # speculative answer): bill an estimate rather than nothing.
                u = {"prompt_tokens": (len(system) + len(user)) // 4, "completion_tokens": written // 4}
            ledger.record_llm(
                step, self.name, u["prompt_tokens"], u["completion_tokens"], (time.perf_counter() - started) * 1000
            )


def build_chat_model(
    settings: Settings, model: str | None = None, breaker: CircuitBreaker | None = None
) -> ChatModel | None:
    choice = settings.llm.lower()
    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    if choice == "offline" or (choice == "auto" and not has_key):
        return None
    if not has_key and not settings.llm_base_url:
        raise RuntimeError("SLR_LLM=openai but OPENAI_API_KEY is not set")
    return OpenAIChatModel(
        model or settings.llm_model,
        settings.llm_base_url,
        settings.llm_timeout_s,
        settings.llm_reasoning_effort,
        breaker or CircuitBreaker(settings.llm_breaker_failures, settings.llm_breaker_cooldown_s),
    )


# --------------------------------------------------------------------------
# prompts — loaded from prompts/*.md at runtime, never inlined
# --------------------------------------------------------------------------


@lru_cache(maxsize=16)
def _read_prompt(prompts_dir: str, name: str) -> tuple[str, str]:
    text = Path(prompts_dir, f"{name}.md").read_text(encoding="utf-8")
    system, _, user = text.partition("\n## USER\n")
    system = system.replace("## SYSTEM\n", "", 1).strip()
    return system, user.strip()


def render_prompt(prompts_dir: str, name: str, **values: str) -> tuple[str, str]:
    system, user = _read_prompt(prompts_dir, name)

    def fill(template: str) -> str:
        return re.sub(r"\{\{(\w+)\}\}", lambda m: str(values.get(m.group(1), "")), template)

    return fill(system), fill(user)


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?|\n?```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        value = json.loads(m.group(0))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value
