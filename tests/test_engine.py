"""End-to-end turns through the real engine (offline arms)."""

from __future__ import annotations

import pytest

from slr.contracts import WIRE_SCHEMA
from slr.telemetry.trace import REQUIRED_FIELDS, missing_fields


async def replay(runner, fixture: str, speed: float = 40.0):
    await runner.start()
    await runner.replay(fixture, speed)
    return runner.completed


async def test_compound_turn_retrieves_before_the_utterance_ends(runner, recorder):
    traces = await replay(runner, "compound_01")
    trace = traces[-1]
    assert trace["before_utterance_end"] is True
    assert trace["first_retrieval_ms"] < trace["utterance_end_ms"]

    started = recorder.of("retrieval.started")
    end = recorder.one("utterance.end")["atMs"]
    assert any(e["atMs"] < end for e in started), "no retrieval began before the utterance ended"
    assert recorder.one("turn.complete")["latencyMs"]["firstRetrieval"] < 0


async def test_compound_turn_answers_every_sub_intent_with_citations(runner, recorder):
    await replay(runner, "compound_01")
    subqueries = recorder.one("subqueries")["items"]
    decomposed = [i for i in subqueries if i["source"] == "decomposed"]
    assert len(decomposed) == 3

    fusion = recorder.one("fusion.final")
    assert fusion["quotaApplied"] is True and fusion["fullCorpusSearch"] is True
    # the coverage guarantee: every intent has evidence in the final context
    for item in decomposed:
        assert any(item["id"] in h["subQueryIds"] for h in fusion["hits"]), f"{item['text']!r} got no evidence"

    version = recorder.one("answer.version")
    assert version["fabricatedCitations"] == 0
    assert version["citationSupportRate"] >= 0.85
    cited = {c for claim in version["claims"] for c in claim["chunkIds"]}
    assert len({h["subQueryIds"][0] for h in fusion["hits"] if h["chunkId"] in cited}) >= 2


async def test_provisional_results_are_reused_not_discarded(runner, recorder):
    await replay(runner, "compound_01")
    results = recorder.of("retrieval.result")
    assert any(r.get("reused") for r in results), "the provisional search's work was thrown away"


async def test_late_detail_refines_without_a_full_corpus_search(runner, recorder):
    traces = await replay(runner, "late_detail_01")
    first, second = traces[-2], traces[-1]
    assert first["mode"] == "retrieve" and second["mode"] == "refine"

    versions = recorder.of("answer.version")
    v1, v2 = versions[0], versions[1]
    assert (v1["version"], v1["parent"]) == (1, None)
    assert (v2["version"], v2["parent"]) == (2, 1)
    assert v2["preserved"], "the refinement preserved nothing"
    assert set(v2["preserved"]) <= {c["id"] for c in v2["claims"]}

    fusion = recorder.of("fusion.final")[-1]
    assert fusion["fullCorpusSearch"] is False
    assert second["fusion"]["carried_from_session"] > 0
    # the delta answer is a superset in meaning: prior claims keep their citations
    kept = {c["id"]: c for c in v2["claims"]}
    for claim_id in v2["preserved"]:
        assert kept[claim_id]["chunkIds"], "a preserved claim lost its citations"


async def test_presentation_turn_runs_zero_vector_queries(runner, recorder):
    traces = await replay(runner, "presentation_01")
    suppressed = traces[-1]
    assert suppressed["mode"] == "suppress"
    assert suppressed["retrieval"] == [] and suppressed["first_retrieval_ms"] is None
    assert suppressed["decisions"][-1]["reason"] == "presentation_restructure"

    turn_id = suppressed["turn_id"]
    assert not [e for e in recorder.of("retrieval.started") if e["turnId"] == turn_id]
    version = [v for v in recorder.of("answer.version") if v["turnId"] == turn_id][-1]
    assert version["preserved"], "prior citations were not retained"
    assert version["fabricatedCitations"] == 0
    body = recorder.answer(turn_id)
    assert "[Doc_" in body, "the reformatted answer dropped its citations"


async def test_chitchat_turn_neither_retrieves_nor_answers(runner, recorder):
    traces = await replay(runner, "chitchat_01")
    assert traces[-1]["mode"] == "suppress"
    assert traces[-1]["decisions"][-1]["reason"] == "no_information_need"
    assert recorder.of("retrieval.started") == []
    assert recorder.one("answer.version")["uncertainty"]


async def test_unanswerable_parts_are_flagged_not_invented(runner, recorder):
    await replay(runner, "unanswerable_01")
    version = recorder.one("answer.version")
    assert version["uncertainty"], "nothing was flagged as unverifiable"
    assert version["fabricatedCitations"] == 0


async def test_single_intent_fixture_stays_single(runner, recorder):
    await replay(runner, "single_01")
    decomposed = [i for i in recorder.one("subqueries")["items"] if i["source"] == "decomposed"]
    assert len(decomposed) == 1


async def test_every_turn_emits_a_complete_trace(runner):
    await runner.start()
    for fixture in ["compound_01", "late_detail_01", "presentation_01", "chitchat_01"]:
        await runner.replay(fixture, 40.0)
    assert len(runner.completed) == 6
    for trace in runner.completed:
        assert missing_fields(trace) == [], f"{trace['turn_id']}: {missing_fields(trace)}"
        assert set(REQUIRED_FIELDS) <= set(trace)
        assert trace["cost"]["turnUsd"] >= 0 and trace["latency_ms"]["complete_after_end"] >= 0
        assert trace["models"]["embedder"] and trace["controller"] == "rule"


async def test_every_emitted_event_matches_the_wire_contract(runner, recorder):
    await replay(runner, "compound_01")
    kinds = {e["type"] for e in recorder.events}
    assert kinds <= set(WIRE_SCHEMA)
    assert {"session.ready", "turn.start", "transcript.chunk", "controller.decision", "utterance.end",
            "subqueries", "retrieval.result", "fusion.final", "answer.token", "answer.version",
            "turn.complete"} <= kinds
    for hit in recorder.one("fusion.final")["hits"]:
        assert set(hit) == {"chunkId", "docId", "section", "text", "score", "branches", "subQueryIds", "citation", "heading"}
        assert hit["citation"].startswith("[") and "§" in hit["citation"]


async def test_session_state_is_scoped_and_clears(runner, recorder):
    await replay(runner, "presentation_01")
    assert runner.store.has_answer
    old_id = runner.store.id
    await runner.new_session()
    assert runner.store.id != old_id
    assert runner.store.topic is None

    # With the session cleared there is nothing to restructure, and the engine
    # says so rather than reaching back into the previous session's answer.
    before = len(runner.completed)
    await runner.play_turns([{"utterance": "Could you rephrase that in simpler words please?"}], 40.0)
    fresh = runner.completed[before]
    assert fresh["mode"] == "suppress"
    assert fresh["answer"]["claims"] == []
    assert fresh["uncertainty"], "a suppressed turn with no prior answer said nothing at all"
    assert "Riverside" not in str(fresh["answer"]["body"])


async def test_no_method_accepts_a_caller_supplied_session_id():
    import inspect

    from slr.session.store import SessionStore

    for name, member in inspect.getmembers(SessionStore, inspect.isfunction):
        params = set(inspect.signature(member).parameters)
        assert not params & {"session_id", "sessionId", "owner", "tenant"}, name


@pytest.mark.parametrize("fixture", ["compound_02", "compound_03", "single_02", "late_detail_02"])
async def test_fixtures_run_clean(runner, recorder, fixture):
    await replay(runner, fixture)
    assert not recorder.of("error")
    assert recorder.one("answer.version")["fabricatedCitations"] == 0
