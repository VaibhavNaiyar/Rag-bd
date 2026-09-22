"""The gate arithmetic itself, on hand-built traces with known answers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from evals.gates import gate_g2, gate_g4, gate_g5
from evals.harness import RunResult, TurnRun


def _trace(**over: Any) -> dict[str, Any]:
    base = {
        "mode": "retrieve",
        "before_utterance_end": False,
        "first_retrieval_ms": None,
        "utterance_end_ms": 1000,
        "chunks": [{"text": "a", "at_ms": 0}, {"text": "b", "at_ms": 800}],
        "retrieval_events": [],
        "sub_queries": [],
        "fabricated_citations": 0,
        "fabricated_citations_blocked": 0,
        "uncertainty": [],
        "answer": None,
        "fusion": None,
    }
    base.update(over)
    return base


def _turn(expect: dict[str, Any], **trace: Any) -> TurnRun:
    return TurnRun("fx", "enterprise", "single", 0, expect, _trace(**trace), [])


def _early(at: int) -> dict[str, Any]:
    return {"before_utterance_end": True, "first_retrieval_ms": at, "retrieval_events": [{"at_ms": at}]}


def test_g2_denominators_follow_the_fixture_not_the_system() -> None:
    result = RunResult(
        [
            _turn({"mode": "retrieve"}, **_early(500)),
            # The system suppressed a real question: it is a late retrieval, not skipped.
            _turn({"mode": "retrieve"}, mode="suppress"),
            # The system searched on a reformat request: a false trigger.
            _turn({"mode": "suppress"}, mode="retrieve", retrieval_events=[{"at_ms": 900}]),
            _turn({"mode": "suppress"}, mode="suppress"),
        ]
    )
    g2 = gate_g2(result)
    assert g2.value == 50.0
    assert g2.detail["eligible_turns"] == 2
    assert g2.detail["suppression_turns"] == 2
    assert g2.detail["false_trigger_rate_pct"] == 50.0
    assert not g2.passed


def test_g2_before_last_word_is_stricter_than_before_end_of_speech() -> None:
    # Both searches start before the end-of-speech signal; only one starts before the last word.
    result = RunResult([_turn({"mode": "retrieve"}, **_early(500)), _turn({"mode": "retrieve"}, **_early(900))])
    g2 = gate_g2(result)
    assert g2.value == 100.0
    assert g2.detail["before_last_word_pct"] == 50.0


def _index() -> SimpleNamespace:
    chunks = [
        SimpleNamespace(chunk_id="c1", doc_id="d1", text="Leave policy allows twenty days"),
        SimpleNamespace(chunk_id="c2", doc_id="d2", text="Travel is reimbursed within thirty days"),
    ]
    documents = [{"doc_id": "d1", "source": "corpus/leave.md"}, {"doc_id": "d2", "source": "corpus/travel.md"}]
    return SimpleNamespace(chunks=chunks, manifest={"documents": documents})


def _answered(generated: int, supported: int, chunk_ids: list[str], **expect: Any) -> TurnRun:
    answer = {
        "grounding": {"generated_claims": generated, "supported_claims": supported},
        "claims": [{"chunkIds": ["c1"]}] * supported,
        "claim_count": supported,
    }
    return _turn({"mode": "retrieve", **expect}, answer=answer, fusion={"chunk_ids": chunk_ids})


def test_g4_pools_claims_and_counts_turns_where_everything_was_withheld() -> None:
    result = RunResult([_answered(4, 4, ["c1"]), _answered(4, 0, ["c1"])])
    g4 = gate_g4(result, _index())
    # 4 of 8 sentences were supported. Averaging only turns that kept a claim would say 100%.
    assert g4.value == 50.0
    assert g4.detail["claims_written"] == 8
    assert g4.detail["claims_supported"] == 4
    assert g4.detail["per_turn_mean_support_pct"] == 50.0
    assert not g4.passed


def test_g4_pooled_support_weights_by_claim_not_by_turn() -> None:
    result = RunResult([_answered(1, 1, ["c1"]), _answered(9, 0, ["c1"])])
    g4 = gate_g4(result, _index())
    assert g4.value == 10.0
    assert g4.detail["per_turn_mean_support_pct"] == 50.0


def test_g4_document_recall_uses_the_source_file_names() -> None:
    result = RunResult(
        [
            _answered(1, 1, ["c1"], gold_docs=["leave.md", "travel.md"]),
            _answered(1, 1, ["c1", "c2"], gold_docs=["travel.md"]),
        ]
    )
    g4 = gate_g4(result, _index())
    assert g4.detail["doc_recall_at_k_pct"] == 75.0
    assert g4.detail["doc_recall_samples"] == 2
    assert g4.detail["recall_at_k_pct"] is None


def test_g4_passage_recall_resolves_gold_text_to_chunks() -> None:
    result = RunResult([_answered(1, 1, ["c2"], gold_passages=["Leave policy allows twenty days"])])
    assert gate_g4(result, _index()).detail["recall_at_k_pct"] == 0.0


def test_g5_passes_a_refinement_whose_parent_verified_nothing() -> None:
    answer = {"version": 2, "parent": 1, "preserved": [], "mutated": []}
    ok = _turn(
        {"mode": "refine"},
        mode="refine",
        answer=answer,
        refinement={"parent_claims": 0},
        fusion={"full_corpus_search": False},
    )
    # The same turn, but the parent had claims and none survived: that is a restart.
    dropped = _turn(
        {"mode": "refine"},
        mode="refine",
        answer=answer,
        refinement={"parent_claims": 3},
        fusion={"full_corpus_search": False},
    )
    g5 = gate_g5(RunResult([ok, dropped]))
    assert g5.value == 50.0
    assert not g5.passed


def test_g5_counts_a_rewritten_claim_as_the_refinement_it_is():
    """A late detail that lands on the parent's only claim rewrites it; nothing is left to preserve."""
    answer = {"version": 2, "parent": 1, "preserved": [], "mutated": ["c1"]}
    turn = _turn(
        {"mode": "refine"},
        mode="refine",
        answer=answer,
        refinement={"parent_claims": 1},
        fusion={"full_corpus_search": False},
    )
    assert gate_g5(RunResult([turn])).value == 100.0
