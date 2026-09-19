"""Per-turn usage ledger.

Pattern from Lattice's ``runtime/usage.py``: every component records into one
ledger, which the engine drains once per turn. Streamed calls take the *last*
usage reading, never the sum — providers report cumulative totals on the final
chunk, and summing double-counts.

Local inference (embedding, rerank, verification) is priced by CPU time so the
reported cost per turn is not flattered by pretending local compute is free.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


def usage_of_stream(readings: list[Any]) -> dict[str, int]:
    """Last non-empty reading wins."""
    for reading in reversed(readings):
        if reading:
            prompt = int(getattr(reading, "prompt_tokens", 0) or 0)
            completion = int(getattr(reading, "completion_tokens", 0) or 0)
            return {"prompt_tokens": prompt, "completion_tokens": completion}
    return {"prompt_tokens": 0, "completion_tokens": 0}


#: Published list prices, USD per million tokens (input, output). Indicative:
#: they go stale, and a deployment on negotiated rates overrides one with
#: ``SLR_PRICE_<MODEL>=<in>,<out>`` (model id upper-cased, non-alphanumerics as
#: underscores: ``SLR_PRICE_GPT_4_1_MINI=0.4,1.6``). A model not listed here is
#: priced at ``SLR_PRICE_IN_PER_M`` / ``SLR_PRICE_OUT_PER_M``.
MODEL_RATES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
}


def rate_for(model: str, default: tuple[float, float]) -> tuple[float, float]:
    """(input, output) USD per million tokens for ``model``.

    A dated snapshot is priced as its family (``gpt-4o-mini-2024-07-18`` as
    ``gpt-4o-mini``), longest name first, and only at a ``-`` boundary, so
    ``gpt-5.4-mini`` is never mistaken for ``gpt-5``.
    """
    override = os.environ.get("SLR_PRICE_" + re.sub(r"[^A-Z0-9]", "_", model.upper()))
    if override:
        try:
            given, taken = (float(part) for part in override.split(","))
            return given, taken
        except ValueError:
            log.warning("ignoring malformed price override for %s: %r", model, override)
    for name in sorted(MODEL_RATES, key=len, reverse=True):
        if model == name or model.startswith(name + "-"):
            return MODEL_RATES[name]
    return default


@dataclass
class UsageLedger:
    price_in_per_m: float
    price_out_per_m: float
    cpu_usd_per_hour: float
    entries: list[dict[str, Any]] = field(default_factory=list)

    def record_llm(self, step: str, model: str, prompt_tokens: int, completion_tokens: int, ms: float) -> None:
        rate_in, rate_out = rate_for(model, (self.price_in_per_m, self.price_out_per_m))
        usd = prompt_tokens / 1e6 * rate_in + completion_tokens / 1e6 * rate_out
        self.entries.append(
            {
                "step": step,
                "kind": "llm",
                "model": model,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "ms": round(ms, 1),
                "usd": usd,
            }
        )

    def record_compute(self, step: str, ms: float, detail: str = "") -> None:
        usd = ms / 3.6e6 * self.cpu_usd_per_hour
        self.entries.append({"step": step, "kind": "compute", "model": detail, "ms": round(ms, 1), "usd": usd})

    def drain(self) -> list[dict[str, Any]]:
        out, self.entries = self.entries, []
        return out

    @staticmethod
    def summarise(entries: list[dict[str, Any]]) -> dict[str, Any]:
        by_step: dict[str, float] = {}
        by_model: dict[str, dict[str, int]] = {}
        tokens = 0
        for e in entries:
            by_step[e["step"]] = by_step.get(e["step"], 0.0) + e["usd"]
            tokens += e.get("prompt_tokens", 0) + e.get("completion_tokens", 0)
            if e["kind"] == "llm":
                counts = by_model.setdefault(e["model"], {"inputTokens": 0, "outputTokens": 0})
                counts["inputTokens"] += e["prompt_tokens"]
                counts["outputTokens"] += e["completion_tokens"]
        return {
            "turnUsd": round(sum(by_step.values()), 6),
            "turnTokens": tokens,
            "steps": [{"step": k, "usd": round(v, 6)} for k, v in by_step.items()],
            "models": [{"model": k, **v} for k, v in by_model.items()],
        }
