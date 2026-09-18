"""Per-turn usage ledger.

Pattern from Lattice's ``runtime/usage.py``: every component records into one
ledger, which the engine drains once per turn. Streamed calls take the *last*
usage reading, never the sum — providers report cumulative totals on the final
chunk, and summing double-counts.

Local inference (embedding, rerank, verification) is priced by CPU time so the
reported cost per turn is not flattered by pretending local compute is free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def usage_of_stream(readings: list[Any]) -> dict[str, int]:
    """Last non-empty reading wins."""
    for reading in reversed(readings):
        if reading:
            prompt = int(getattr(reading, "prompt_tokens", 0) or 0)
            completion = int(getattr(reading, "completion_tokens", 0) or 0)
            return {"prompt_tokens": prompt, "completion_tokens": completion}
    return {"prompt_tokens": 0, "completion_tokens": 0}


@dataclass
class UsageLedger:
    price_in_per_m: float
    price_out_per_m: float
    cpu_usd_per_hour: float
    entries: list[dict[str, Any]] = field(default_factory=list)

    def record_llm(self, step: str, model: str, prompt_tokens: int, completion_tokens: int, ms: float) -> None:
        usd = prompt_tokens / 1e6 * self.price_in_per_m + completion_tokens / 1e6 * self.price_out_per_m
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
        tokens = 0
        for e in entries:
            by_step[e["step"]] = by_step.get(e["step"], 0.0) + e["usd"]
            tokens += e.get("prompt_tokens", 0) + e.get("completion_tokens", 0)
        return {
            "turnUsd": round(sum(by_step.values()), 6),
            "turnTokens": tokens,
            "steps": [{"step": k, "usd": round(v, 6)} for k, v in by_step.items()],
        }
