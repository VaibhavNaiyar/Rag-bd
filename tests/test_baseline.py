"""The baseline pipeline the benchmark compares against: a conventional batch RAG turn."""

from __future__ import annotations

from slr.stream.engine import SessionRunner


async def _play(engine, recorder, fixture: str) -> list[dict]:
    runner = SessionRunner(engine, recorder)
    await runner.start()
    await runner.replay(fixture, 40.0)
    return runner.completed


async def test_the_baseline_waits_splits_nothing_refines_nothing_and_suppresses_nothing(engine, settings, recorder):
    baseline = engine.with_settings(settings.with_overrides(controller="batch", decompose=False))

    compound = (await _play(baseline, recorder, "compound_01"))[-1]
    assert compound["first_retrieval_ms"] >= compound["utterance_end_ms"], "a batch system waits for the end"
    assert compound["before_utterance_end"] is False
    assert [s["source"] for s in compound["sub_queries"]] == ["decomposed"], "one search for the whole utterance"
    assert compound["decomposition"]["method"] == "off"

    late = (await _play(baseline, recorder, "late_detail_01"))[-1]
    assert late["mode"] == "retrieve" and late["fusion"]["full_corpus_search"] is True, "a late detail restarts"

    reformat = (await _play(baseline, recorder, "presentation_01"))[-1]
    assert reformat["mode"] == "retrieve" and reformat["retrieval_events"], "a reformat request searches again"


async def test_our_pipeline_on_the_same_fixtures_does_all_three(engine, recorder):
    compound = (await _play(engine, recorder, "compound_01"))[-1]
    assert compound["before_utterance_end"] is True
    assert len([s for s in compound["sub_queries"] if s["source"] == "decomposed"]) >= 2
    assert (await _play(engine, recorder, "late_detail_01"))[-1]["mode"] == "refine"
    assert (await _play(engine, recorder, "presentation_01"))[-1]["mode"] == "suppress"
