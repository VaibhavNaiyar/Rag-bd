"""A turn's trace record, exported as OpenTelemetry spans for the G6 dashboard."""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from slr.stream.engine import SessionRunner
from slr.telemetry.otel import OtelExporter
from slr.telemetry.trace import TraceSink

UTTERANCE = ["What is the", "venue cancellation", "policy for workshops?"]


@pytest.fixture
def spans(engine, tmp_path):
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    original = engine.sink
    engine.sink = TraceSink(str(tmp_path / "trace.jsonl"), exporter=OtelExporter(provider))
    yield memory
    engine.sink = original


async def _one_turn(engine, recorder) -> dict:
    runner = SessionRunner(engine, recorder)
    await runner.start()
    await runner.utterance_start()
    for piece in UTTERANCE:
        await runner.utterance_chunk(piece)
    await runner.utterance_end()
    return runner.completed[-1]


async def test_a_turn_becomes_one_trace_with_a_span_per_stage(engine, recorder, spans):
    record = await _one_turn(engine, recorder)
    by_name = {s.name: s for s in spans.get_finished_spans()}
    assert {"turn", "listen", "plan", "retrieve", "synthesise"} <= set(by_name)

    root = by_name["turn"]
    for name in ("listen", "plan", "retrieve", "synthesise"):
        span = by_name[name]
        assert span.parent is not None and span.parent.span_id == root.context.span_id
        assert span.context.trace_id == root.context.trace_id
        assert root.start_time <= span.start_time <= span.end_time <= root.end_time

    # the stage boundaries are the record's own timings
    t0 = record["started_at"] * 1_000_000
    assert by_name["listen"].end_time == t0 + record["utterance_end_ms"] * 1_000_000
    assert by_name["retrieve"].start_time == t0 + record["first_retrieval_ms"] * 1_000_000

    assert root.attributes["slr.turn_id"] == record["turn_id"]
    assert root.attributes["slr.mode"] == "retrieve"
    assert root.attributes["slr.searches"] == len(record["retrieval_events"]) > 0
    assert [e.name for e in by_name["listen"].events].count("decision") == len(record["decisions"])
    assert [e.name for e in by_name["retrieve"].events].count("search") == len(record["retrieval_events"])
    assert [e.name for e in by_name["synthesise"].events] == ["first_token"]


async def test_no_user_or_document_text_leaves_in_a_span(engine, recorder, spans):
    record = await _one_turn(engine, recorder)
    words = {w.lower().strip("?") for piece in UTTERANCE for w in piece.split() if len(w) > 4}
    words |= {q["query"].lower() for q in record["retrieval_events"]}

    def values(span):
        yield from span.attributes.values()
        for event in span.events:
            yield from event.attributes.values()

    for span in spans.get_finished_spans():
        for value in values(span):
            text = str(value).lower()
            assert not any(word in text for word in words), f"{span.name} carries text: {value!r}"


def test_export_is_off_without_an_endpoint():
    assert OtelExporter.from_settings("") is None


# --------------------------------------------------------------------------
# Cost per turn, priced per model
# --------------------------------------------------------------------------


def test_each_model_is_priced_at_its_own_rate(monkeypatch):
    from slr.telemetry.cost import UsageLedger, rate_for

    default = (9.0, 9.0)
    assert rate_for("gpt-4o-mini", default) == (0.15, 0.60)
    assert rate_for("gpt-4o-mini-2024-07-18", default) == (0.15, 0.60), "a dated snapshot is its family"
    assert rate_for("gpt-4.1-mini-2025-04-14", default) == (0.40, 1.60), "longest family name wins"
    assert rate_for("gpt-4.1", default) == (2.00, 8.00)
    assert rate_for("gpt-5.4-mini", default) == default, "not gpt-5: unknown models use the configured price"
    monkeypatch.setenv("SLR_PRICE_GPT_4_1", "1.0,4.0")
    assert rate_for("gpt-4.1", default) == (1.0, 4.0)

    ledger = UsageLedger(0.15, 0.60, 0.0)
    ledger.record_llm("decompose", "gpt-4.1", 1_000_000, 0, 1.0)
    ledger.record_llm("synthesise", "gpt-4o-mini", 0, 1_000_000, 1.0)
    assert [round(e["usd"], 6) for e in ledger.entries] == [1.0, 0.60]
