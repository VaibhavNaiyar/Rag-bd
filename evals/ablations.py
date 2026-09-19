"""Architectural ablations and the baseline comparison, driven by env-level knobs.

1. hybrid (BM25 + dense) vs dense-only retrieval
2. rule-based vs model-based retrieval controller
3. per-intent quota on vs off

And the baseline: the same fixtures through a conventional batch RAG turn
(``SLR_CONTROLLER=batch``: nothing until the speaker stops, then one search for
the whole utterance with ``SLR_DECOMPOSE=off``, no suppression, no
refinement), with the same retriever, reranker, synthesiser and verifier, so
the difference is exactly what streaming, decomposition and session
refinement add.

Arm 3 exists because aggregate recall cannot see the failure it prevents: one
verbose sub-intent taking every slot in the context window while another
question the user asked silently goes unanswered.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from evals.gates import gate_g2, gate_g3, gate_g4, gate_g5
from evals.harness import RunResult, build_engine, run
from slr.config import Settings
from slr.retrieval.store import Index


@dataclass
class Arm:
    name: str
    settings: Settings
    result: RunResult | None = None
    skipped: str = ""


def intent_coverage(result: RunResult) -> dict[str, Any]:
    """Share of decomposed sub-intents that got at least one chunk into the answer context."""
    covered = starved = 0
    starved_examples = []
    for turn in result.turns:
        fusion = turn.trace.get("fusion") or {}
        per_sq = fusion.get("per_sub_query") or {}
        decomposed = [s["id"] for s in turn.trace["sub_queries"] if s["source"] == "decomposed"]
        if len(decomposed) < 2:
            continue
        for sq_id in decomposed:
            if per_sq.get(sq_id, 0) > 0:
                covered += 1
            else:
                starved += 1
                text = next((s["text"] for s in turn.trace["sub_queries"] if s["id"] == sq_id), sq_id)
                starved_examples.append({"fixture": turn.fixture, "sub_query": text})
    total = covered + starved
    return {
        "multi_intent_sub_queries": total,
        "with_evidence_pct": None if total == 0 else round(100 * covered / total, 1),
        "starved": starved,
        "starved_examples": starved_examples[:5],
    }


def summarise(result: RunResult, index: Index) -> dict[str, Any]:
    g2, g4, g5 = gate_g2(result), gate_g4(result, index), gate_g5(result)
    g3 = gate_g3(result, index.embedder)
    ttft = [t.trace["latency_ms"]["first_token_after_end"] for t in result.turns if t.trace.get("latency_ms")]
    done = [t.trace["latency_ms"]["complete_after_end"] for t in result.turns if t.trace.get("latency_ms")]
    costs = [t.trace["cost"]["turnUsd"] for t in result.turns if t.trace.get("cost")]
    tokens = [t.trace["cost"]["turnTokens"] for t in result.turns if t.trace.get("cost")]
    return {
        "turns": len(result.turns),
        "early_retrieval_pct": g2.value,
        "false_trigger_pct": g2.detail["false_trigger_rate_pct"],
        "multi_intent_pct": g3.value,
        "citation_support_pct": g4.value,
        "recall_at_k_pct": g4.detail["recall_at_k_pct"],
        "fabricated_citations": g4.detail["fabricated_citations"],
        "refined_not_restarted_pct": g5.value,
        "intent_coverage": intent_coverage(result),
        "median_ttft_ms": round(statistics.median(ttft)) if ttft else None,
        "median_complete_ms": round(statistics.median(done)) if done else None,
        "mean_cost_usd": round(statistics.fmean(costs), 6) if costs else None,
        "mean_tokens": round(statistics.fmean([float(t) for t in tokens])) if tokens else None,
        "wall_clock_s": round(result.seconds, 1),
        "knobs": result.settings,
    }


def run_arm(arm: Arm, fixtures: list[dict], index: Index, speed: float) -> dict[str, Any]:
    if arm.skipped:
        print(f"[eval]   {arm.name}: skipped — {arm.skipped}", flush=True)
        return {"arm": arm.name, "skipped": arm.skipped}
    print(f"[eval]   {arm.name}: {len(fixtures)} fixtures", flush=True)
    engine = build_engine(arm.settings)
    arm.result = run(engine, fixtures, speed)
    print(f"[eval]   {arm.name}: {len(arm.result.turns)} turns in {arm.result.seconds:.0f}s", flush=True)
    return {"arm": arm.name, **summarise(arm.result, index)}


def compare_baseline(
    ours: RunResult, settings: Settings, fixtures: list[dict], index: Index, speed: float
) -> dict[str, Any]:
    """Our run, and the same fixtures through the batch baseline, summarised side by side."""
    baseline = Arm("baseline (batch RAG)", settings.with_overrides(controller="batch", decompose=False))
    row = run_arm(baseline, fixtures, index, speed)
    return {"ours": {"arm": "streaming live RAG", **summarise(ours, index)}, "baseline": row}


def run_ablations(
    base: Settings, fixtures: list[dict], index: Index, speed: float = 8.0, has_llm: bool = False
) -> list[dict[str, Any]]:
    """Each experiment is a pair of arms that differ in exactly one knob."""
    experiments = [
        (
            "retrieval branches",
            "Does the lexical branch earn its place, or would dense-only do?",
            [
                Arm("hybrid (bm25+dense)", base.with_overrides(branches=("bm25", "dense"))),
                Arm("dense only", base.with_overrides(branches=("dense",))),
            ],
        ),
        (
            "per-intent quota",
            "What happens to the quiet sub-intent when the context window is filled by score alone?",
            [
                Arm("quota = 2 per intent", base.with_overrides(quota_per_intent=2)),
                Arm("quota = 0 (global top-k)", base.with_overrides(quota_per_intent=0)),
            ],
        ),
        (
            "retrieval controller",
            "Is a model call per transcript chunk worth what it costs?",
            [
                Arm("rule controller", base.with_overrides(controller="rule")),
                Arm(
                    "model controller",
                    base.with_overrides(controller="model"),
                    skipped="" if has_llm else "needs an LLM (set OPENAI_API_KEY) — not run",
                ),
            ],
        ),
    ]
    out = []
    for name, question, arms in experiments:
        print(f"[eval] ablation: {name}", flush=True)
        rows = [run_arm(arm, fixtures, index, speed) for arm in arms]
        out.append({"experiment": name, "question": question, "arms": rows})
    return out
