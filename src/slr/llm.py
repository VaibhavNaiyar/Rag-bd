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
from typing import Protocol

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


#: Reasoning models (o-series, gpt-5 family) spend completion tokens on hidden
#: reasoning before the answer, take no temperature, and cap output with
#: max_completion_tokens. Their "-chat" variants are ordinary chat models.
_REASONING = re.compile(r"^(o\d|gpt-5)(?!.*chat)")
REASONING_HEADROOM = 2000


class OpenAIChatModel:
    def __init__(self, model: str, base_url: str = "", timeout: float = 30.0, reasoning_effort: str = "low"):
        from openai import AsyncOpenAI

        kwargs = {"timeout": timeout, "max_retries": 1}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)
        self.name = model
        self.reasoning = bool(_REASONING.match(model))
        self.reasoning_effort = reasoning_effort

    def _limits(self, max_tokens: int) -> dict:
        if self.reasoning:
            return {"max_completion_tokens": max_tokens + REASONING_HEADROOM, "reasoning_effort": self.reasoning_effort}
        return {"temperature": 0, "max_tokens": max_tokens}

    async def complete(self, system, user, *, ledger, step, json_mode=False, max_tokens=600) -> str:
        started = time.perf_counter()
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = await self._client.chat.completions.create(
            model=self.name,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **self._limits(max_tokens),
            **extra,
        )
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
        started = time.perf_counter()
        readings = []
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
                    yield chunk.choices[0].delta.content
        finally:
            u = usage_of_stream(readings)
            ledger.record_llm(
                step, self.name, u["prompt_tokens"], u["completion_tokens"], (time.perf_counter() - started) * 1000
            )


def build_chat_model(settings: Settings) -> ChatModel | None:
    choice = settings.llm.lower()
    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    if choice == "offline" or (choice == "auto" and not has_key):
        return None
    if not has_key and not settings.llm_base_url:
        raise RuntimeError("SLR_LLM=openai but OPENAI_API_KEY is not set")
    return OpenAIChatModel(
        settings.llm_model, settings.llm_base_url, settings.llm_timeout_s, settings.llm_reasoning_effort
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
