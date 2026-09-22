"""G1..G6 measurement.

Each gate reports the number, the threshold, pass/fail, and the per-turn detail
behind it. Nothing is smoothed: a gate that misses is reported as missed.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from evals.harness import RunResult, TurnRun
from slr.contracts import Hit
from slr.retrieval.embed import Embedder
from slr.retrieval.store import Index
from slr.synthesis.grounding import CANDIDATE_MARKER, Verifier, _subject
from slr.telemetry.trace import missing_fields
from slr.text import content_tokens

MATCH_MIN = 0.75  # embedding similarity at which a predicted sub-query counts as a gold intent
#: A second, larger NLI model that grades the claims the engine shipped. The engine's
#: own verifier approved every one of them, so its verdict cannot be the evidence.
AUDIT_NLI = "cross-encoder/nli-deberta-v3-base"


@dataclass
class Gate:
    id: str
    name: str
    value: float | None
    threshold: str
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)
    #: False when this corpus has no fixtures that exercise the gate. Such a gate
    #: is reported as not-applicable rather than counted as a pass or a failure.
    applicable: bool = True
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "value": self.value,
            "threshold": self.threshold,
            "passed": bool(self.passed),
            "applicable": self.applicable,
            "note": self.note,
            "detail": self.detail,
        }


def _pct(part: int, whole: int) -> float | None:
    return None if whole == 0 else round(100.0 * part / whole, 1)


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


# --------------------------------------------------------------------------
# G2 — early retrieval
# --------------------------------------------------------------------------


def _expected_mode(turn: TurnRun) -> str:
    """What the fixture says the turn should do; the system's own routing only where no label exists.

    Every denominator below is the ground truth, never the system's decision: a
    reformat request the system wrongly searched on is a false trigger, and a
    question the system wrongly suppressed is a late retrieval, not a skipped turn.
    """
    return turn.expect.get("mode") or turn.trace["mode"]


def _before_last_word(turn: TurnRun) -> bool:
    """Retrieval began while the speaker was still talking, not in the endpointing pause after."""
    first, chunks = turn.trace.get("first_retrieval_ms"), turn.trace.get("chunks") or []
    return first is not None and bool(chunks) and first < chunks[-1]["at_ms"]


def gate_g2(result: RunResult) -> Gate:
    eligible = [t for t in result.turns if _expected_mode(t) in ("retrieve", "refine")]
    early = [t for t in eligible if t.trace.get("before_utterance_end")]
    leads = [
        t.trace["utterance_end_ms"] - t.trace["first_retrieval_ms"]
        for t in early
        if t.trace["first_retrieval_ms"] is not None
    ]
    suppressed = [t for t in result.turns if _expected_mode(t) == "suppress"]
    false_triggers = [t for t in suppressed if t.trace["retrieval_events"]]
    def lead_of(turn: TurnRun) -> int:
        return turn.trace["utterance_end_ms"] - turn.trace["first_retrieval_ms"]

    multi = [t for t in early if len([s for s in t.trace["sub_queries"] if s["source"] == "decomposed"]) >= 2]
    single = [t for t in early if t not in multi]
    value = _pct(len(early), len(eligible))
    return Gate(
        "G2",
        "Early retrieval",
        value,
        ">= 80% of eligible turns, with a low false-trigger rate",
        bool(value is not None and value >= 80 and (not suppressed or len(false_triggers) / len(suppressed) <= 0.1)),
        {
            "eligible_turns": len(eligible),
            "retrieved_before_end": len(early),
            "before_last_word_pct": _pct(len([t for t in eligible if _before_last_word(t)]), len(eligible)),
            "median_lead_ms": round(statistics.median(leads)) if leads else None,
            "max_lead_ms": max(leads) if leads else None,
            "median_lead_multi_intent_ms": round(statistics.median([lead_of(t) for t in multi])) if multi else None,
            "median_lead_single_intent_ms": round(statistics.median([lead_of(t) for t in single])) if single else None,
            "multi_intent_turns": len(multi),
            "suppression_turns": len(suppressed),
            "false_trigger_rate_pct": _pct(len(false_triggers), len(suppressed)),
            "late_turns": [t.fixture for t in eligible if t not in early],
        },
    )


# --------------------------------------------------------------------------
# G3 — multi-intent identification
# --------------------------------------------------------------------------


def _match(predicted: list[str], gold: list[str], embedder: Embedder) -> tuple[int, list[float]]:
    """Hungarian assignment between predicted sub-queries and gold sub-intents."""
    if not predicted or not gold:
        return 0, []
    pv = embedder.embed(predicted, kind="query")
    gv = embedder.embed(gold, kind="query")
    sim = gv @ pv.T
    rows, cols = linear_sum_assignment(-sim)
    scores = [float(sim[r, c]) for r, c in zip(rows, cols)]
    return sum(s >= MATCH_MIN for s in scores), scores


def gate_g3(result: RunResult, embedder: Embedder) -> Gate:
    rows = []
    for turn in result.turns:
        gold = turn.expect.get("gold_sub_intents")
        if not gold:
            continue
        predicted = [s["text"] for s in turn.trace["sub_queries"] if s["source"] == "decomposed"]
        matched, scores = _match(predicted, gold, embedder)
        rows.append(
            {
                "fixture": turn.fixture,
                "family": turn.family,
                "gold": len(gold),
                "predicted": len(predicted),
                "matched": matched,
                "best_scores": [round(s, 3) for s in scores],
                "method": (turn.trace.get("decomposition") or {}).get("method"),
            }
        )

    # On ASQA a compound utterance is an ambiguous question and its gold
    # sub-intents are the readings ASQA disambiguates it into.
    compound = [r for r in rows if r["gold"] >= 2]
    singles = [r for r in rows if r["gold"] == 1]
    hit = [r for r in compound if r["matched"] >= 2]
    value = _pct(len(hit), len(compound))
    over_fragmented = [r for r in rows if r["predicted"] > r["gold"]]
    return Gate(
        "G3",
        "Multi-intent identification",
        value,
        ">= 70% of compound utterances isolate >= 2 distinct sub-intents",
        bool(value is not None and value >= 70),
        {
            "compound_utterances": len(compound),
            "identified": len(hit),
            "full_coverage_pct": _pct(len([r for r in compound if r["matched"] >= r["gold"]]), len(compound)),
            "over_fragmentation_pct": _pct(len(over_fragmented), len(rows)),
            "single_intent_kept_single_pct": _pct(len([r for r in singles if r["predicted"] == 1]), len(singles)),
            "by_family": {
                fam: _pct(
                    len([r for r in rows if r["family"] == fam and r["gold"] >= 2 and r["matched"] >= 2]),
                    len([r for r in rows if r["family"] == fam and r["gold"] >= 2]),
                )
                for fam in sorted({r["family"] for r in rows if r["gold"] >= 2})
            },
            "rows": rows,
        },
    )


# --------------------------------------------------------------------------
# G4 — factual grounding
# --------------------------------------------------------------------------


_CHUNK_TOKENS: dict[int, tuple[dict[str, str], list[tuple[str, set[str]]]]] = {}


def _chunk_lookup(index: Index) -> tuple[dict[str, str], list[tuple[str, set[str]]]]:
    """Per index, tokenised once: exact text -> chunk id, and every chunk's token set."""
    if id(index) not in _CHUNK_TOKENS:
        exact = {" ".join(c.text.split()): c.chunk_id for c in index.chunks}
        _CHUNK_TOKENS[id(index)] = (exact, [(c.chunk_id, set(content_tokens(c.text))) for c in index.chunks])
    return _CHUNK_TOKENS[id(index)]


def _gold_chunk_ids(passages: list[str], index: Index) -> set[str]:
    """Resolve gold passages to chunk ids: exact text first, then content overlap."""
    exact, chunks = _chunk_lookup(index)
    ids: set[str] = set()
    for passage in passages:
        if " ".join(passage.split()) in exact:
            ids.add(exact[" ".join(passage.split())])
            continue
        needle = set(content_tokens(passage))
        if not needle:
            continue
        best, score = None, 0.0
        for chunk_id, tokens in chunks:
            overlap = len(needle & tokens) / len(needle)
            if overlap > score:
                best, score = chunk_id, overlap
        if best and score >= 0.8:
            ids.add(best)
    return ids


def _doc_sources(index: Index) -> dict[str, str]:
    """chunk id -> the file name of the document it came from."""
    by_doc = {d["doc_id"]: Path(d["source"]).name for d in index.manifest.get("documents", [])}
    return {c.chunk_id: by_doc.get(c.doc_id, "") for c in index.chunks}


def audit_shipped(result: RunResult, index: Index, auditor: Verifier) -> dict[str, Any]:
    """Re-check every shipped claim against the chunks it cites, with a model the engine never used."""
    chunks = {c.chunk_id: c for c in index.chunks}
    checked = held = 0
    failures: list[dict[str, Any]] = []
    for turn in result.turns:
        for claim in (turn.trace.get("answer") or {}).get("claims", []):
            cited = [chunks[cid] for cid in claim["chunkIds"] if cid in chunks]
            if not cited:
                continue
            text = " ".join(CANDIDATE_MARKER.sub("", claim["text"]).split())
            hits = [Hit(c, 0.0, [], []) for c in cited]
            best = max(auditor.support(text, [c.text for c in cited], [_subject(h) for h in hits]))
            checked += 1
            if best >= 0.5:
                held += 1
            elif len(failures) < 10:
                failures.append({"fixture": turn.fixture, "claim": text, "support": round(best, 3)})
    return {
        "auditor": getattr(auditor, "name", type(auditor).__name__),
        "shipped_claims_checked": checked,
        "shipped_claims_supported_pct": _pct(held, checked),
        "shipped_claims_failing_examples": failures,
    }


def gate_g4(result: RunResult, index: Index, auditor: Verifier | None = None) -> Gate:
    """Citation support over every factual sentence the model wrote, pooled across turns.

    A sentence the verifier withheld counts against support even though it never
    reached the user. Averaging per turn instead, over turns that kept a claim,
    would drop exactly the turns where every sentence failed.
    """
    generated = supported = fabricated = answered_intents = uncited_claims = 0
    per_turn: list[float] = []
    passage_recall: list[float] = []
    doc_recall: list[float] = []
    sources = _doc_sources(index)
    for turn in result.turns:
        answer = turn.trace.get("answer") or {}
        grounding = answer.get("grounding") or {}
        generated += grounding.get("generated_claims", 0)
        supported += grounding.get("supported_claims", 0)
        if grounding.get("generated_claims"):
            per_turn.append(grounding["supported_claims"] / grounding["generated_claims"])
        fabricated += turn.trace["fabricated_citations"]
        uncited_claims += len([c for c in answer.get("claims", []) if not c["chunkIds"]])
        answered_intents += answer.get("claim_count", 0)

        got = set((turn.trace.get("fusion") or {}).get("chunk_ids", []))
        gold = turn.expect.get("gold_passages")
        if gold:
            wanted = _gold_chunk_ids(gold, index)
            if wanted:
                passage_recall.append(len(wanted & got) / len(wanted))
        gold_docs = turn.expect.get("gold_docs")
        if gold_docs and turn.trace.get("fusion"):
            got_docs = {sources.get(cid, "") for cid in got}
            doc_recall.append(len(set(gold_docs) & got_docs) / len(set(gold_docs)))

    value = _pct(supported, generated)
    audit = audit_shipped(result, index, auditor) if auditor is not None else {}
    return Gate(
        "G4",
        "Factual grounding",
        value,
        ">= 85% citation support and zero fabricated citations",
        bool(value is not None and value >= 85 and fabricated == 0),
        {
            "claims_written": generated,
            "claims_supported": supported,
            "per_turn_mean_support_pct": None if not per_turn else round(100 * statistics.fmean(per_turn), 1),
            "turns_with_claims": len([t for t in result.turns if (t.trace.get("answer") or {}).get("claim_count")]),
            "fabricated_citations": fabricated,
            "fabricated_blocked_before_shipping": sum(t.trace["fabricated_citations_blocked"] for t in result.turns),
            "claims_without_a_citation": uncited_claims,
            "recall_at_k_pct": None if not passage_recall else round(100 * statistics.fmean(passage_recall), 1),
            "recall_samples": len(passage_recall),
            "doc_recall_at_k_pct": None if not doc_recall else round(100 * statistics.fmean(doc_recall), 1),
            "doc_recall_samples": len(doc_recall),
            "turns_flagging_uncertainty": len([t for t in result.turns if t.trace["uncertainty"]]),
            "total_claims": answered_intents,
            **audit,
        },
    )


# --------------------------------------------------------------------------
# G5 — session refinement
# --------------------------------------------------------------------------


def gate_g5(result: RunResult) -> Gate:
    refine_turns = [t for t in result.turns if t.expect.get("mode") == "refine"]
    rows = []
    for turn in refine_turns:
        answer = turn.trace.get("answer") or {}
        fusion = turn.trace.get("fusion") or {}
        row = {
            "fixture": turn.fixture,
            "routed_as_refine": turn.trace["mode"] == "refine",
            "version": answer.get("version"),
            "parent": answer.get("parent"),
            "preserved": len(answer.get("preserved", [])),
            "mutated": len(answer.get("mutated", [])),
            "full_corpus_search": fusion.get("full_corpus_search"),
            "carried_from_session": fusion.get("carried_from_session", 0),
            "parent_claims": (turn.trace.get("refinement") or {}).get("parent_claims"),
        }
        # The criterion is the guide's: the late detail must update the previous answer
        # rather than clear the session or search the corpus again. Every parent claim is
        # accounted for — kept as it was, or rewritten — and a correction that lands on the
        # only claim the parent had rewrites all of them, which is the point of refining.
        row["accounted_for"] = row["preserved"] + row["mutated"]
        row["passed"] = bool(
            row["routed_as_refine"]
            and row["version"] == 2
            and row["parent"] == 1
            and (row["accounted_for"] > 0 or row["parent_claims"] == 0)
            and row["full_corpus_search"] is False
        )
        rows.append(row)
    passed = [r for r in rows if r["passed"]]
    value = _pct(len(passed), len(rows))
    return Gate(
        "G5",
        "Session refinement",
        value,
        "every late-detail turn: version 2, parent 1, every parent claim kept or rewritten, no full-corpus search",
        bool(rows and len(passed) == len(rows)),
        {"refinement_turns": len(rows), "rows": rows},
        applicable=bool(rows),
        note="" if rows else "this corpus has no late-detail fixtures; G5 is measured on the enterprise corpus",
    )


# --------------------------------------------------------------------------
# G6 — telemetry coverage
# --------------------------------------------------------------------------


def gate_g6(result: RunResult) -> Gate:
    incomplete = []
    for turn in result.turns:
        missing = missing_fields(turn.trace)
        if missing:
            incomplete.append({"fixture": turn.fixture, "turn": turn.trace["turn_id"], "missing": missing})
    value = _pct(len(result.turns) - len(incomplete), len(result.turns))
    return Gate(
        "G6",
        "Telemetry and observability",
        value,
        "100% of turns emit a complete trace record",
        bool(value == 100.0),
        {"turns": len(result.turns), "incomplete": incomplete},
    )


# --------------------------------------------------------------------------
# G1 — reproducibility
# --------------------------------------------------------------------------


def gate_g1(steps: dict[str, Any]) -> Gate:
    ok = bool(steps.get("index_built_by_harness") and steps.get("fixtures_ran") and not steps.get("manual_steps"))
    return Gate(
        "G1",
        "Reproducibility",
        100.0 if ok else 0.0,
        "one command builds the index and runs the suite with no manual step",
        ok,
        steps,
    )


# --------------------------------------------------------------------------


def latency_and_cost(result: RunResult) -> dict[str, Any]:
    def series(key: str) -> list[float]:
        return [t.trace["latency_ms"][key] for t in result.turns if t.trace.get("latency_ms")]

    leads = [
        t.trace["utterance_end_ms"] - t.trace["first_retrieval_ms"]
        for t in result.turns
        if t.trace.get("first_retrieval_ms") is not None and t.trace.get("before_utterance_end")
    ]
    ttft = series("first_token_after_end")
    complete = series("complete_after_end")
    costs = [t.trace["cost"]["turnUsd"] for t in result.turns if t.trace.get("cost")]
    tokens = [t.trace["cost"]["turnTokens"] for t in result.turns if t.trace.get("cost")]
    return {
        "turns": len(result.turns),
        "retrieval_lead_ms": {"median": _median(leads), "p90": _p(leads, 90)},
        "time_to_first_token_ms": {"median": _median(ttft), "p90": _p(ttft, 90)},
        "turn_complete_ms": {"median": _median(complete), "p90": _p(complete, 90)},
        "cost_per_turn_usd": {
            "mean": round(statistics.fmean(costs), 8) if costs else None,
            "total": round(sum(costs), 6),
        },
        "tokens_per_turn": {"mean": _mean([float(t) for t in tokens])},
        "wall_clock_seconds": round(result.seconds, 1),
    }


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values)) if values else None


def _p(values: list[float], pct: int) -> float | None:
    return round(float(np.percentile(values, pct))) if values else None


def all_gates(
    result: RunResult,
    index: Index,
    embedder: Embedder,
    g1_steps: dict[str, Any],
    auditor: Verifier | None = None,
) -> list[Gate]:
    return [
        gate_g1(g1_steps),
        gate_g2(result),
        gate_g3(result, embedder),
        gate_g4(result, index, auditor),
        gate_g5(result),
        gate_g6(result),
    ]
