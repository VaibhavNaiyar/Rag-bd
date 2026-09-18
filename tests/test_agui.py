"""The AG-UI boundary: engine events in, protocol-correct AG-UI events out.

``check_stream`` is what any AG-UI client relies on: runs open and close
once, tool calls start before their results, text messages are balanced, and
every STATE_DELTA applies cleanly to the last STATE_SNAPSHOT. test_api.py uses
it on the live WebSocket and SSE streams too.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from ag_ui.core import EventType

from slr.api.agui import AgUiTranslator, wire

RUN_EVENTS = {EventType.RUN_STARTED.value, EventType.RUN_FINISHED.value, EventType.RUN_ERROR.value}


def apply_patch(state: Any, ops: list[dict[str, Any]]) -> Any:
    """Minimal RFC 6902 add/replace, which is all the translator emits."""
    state = copy.deepcopy(state)
    for op in ops:
        assert op["op"] in ("add", "replace"), op
        keys = [k.replace("~1", "/").replace("~0", "~") for k in op["path"].split("/")[1:]]
        target = state
        for key in keys[:-1]:
            target = target[int(key)] if isinstance(target, list) else target[key]
        last = keys[-1]
        if isinstance(target, list):
            target.append(op["value"]) if last == "-" else target.__setitem__(int(last), op["value"])
        else:
            if op["op"] == "replace":
                assert last in target, f"replace of a missing key: {op['path']}"
            target[last] = op["value"]
    return state


def check_stream(events: list[dict[str, Any]]) -> Any:
    """Assert the protocol invariants; return the client's final shared state."""
    state: Any = None
    run: str | None = None
    calls: dict[str, str] = {}  # id -> "open" | "ended"
    message: str | None = None
    step: str | None = None
    for e in events:
        kind = e["type"]
        if kind != "RUN_ERROR":
            assert run is not None or kind == "RUN_STARTED", f"{kind} outside a run"
        if kind == "RUN_STARTED":
            assert run is None, "a run started inside another"
            run = e["runId"]
        elif kind in ("RUN_FINISHED", "RUN_ERROR"):
            assert message is None, "a run ended with a text message still open"
            assert step is None, "a run ended with a step still open"
            if kind == "RUN_FINISHED":
                assert run == e["runId"], "RUN_FINISHED for a run that is not open"
            run = None
        elif kind == "STEP_STARTED":
            assert step is None, "a step started inside another"
            step = e["stepName"]
        elif kind == "STEP_FINISHED":
            assert step == e["stepName"], "STEP_FINISHED for a step that is not open"
            step = None
        elif kind == "STATE_SNAPSHOT":
            state = e["snapshot"]
        elif kind == "STATE_DELTA":
            assert state is not None, "a delta arrived before any snapshot"
            state = apply_patch(state, e["delta"])
        elif kind == "TEXT_MESSAGE_START":
            assert message is None and run is not None
            message = e["messageId"]
        elif kind == "TEXT_MESSAGE_CONTENT":
            assert e["messageId"] == message and e["delta"]
        elif kind == "TEXT_MESSAGE_END":
            assert e["messageId"] == message
            message = None
        elif kind == "TOOL_CALL_START":
            assert e["toolCallId"] not in calls and run is not None
            calls[e["toolCallId"]] = "open"
        elif kind == "TOOL_CALL_ARGS":
            assert calls[e["toolCallId"]] == "open"
            json.loads(e["delta"])
        elif kind == "TOOL_CALL_END":
            calls[e["toolCallId"]] = "ended"
        elif kind == "TOOL_CALL_RESULT":
            assert calls.get(e["toolCallId"]) == "ended", "a result for a call that never started"
            json.loads(e["content"])
        else:
            raise AssertionError(f"unexpected AG-UI event {kind}")
    assert run is None, "the stream ended with a run still open"
    return state


async def _turn_events(engine, utterances: list[str]) -> tuple[list[dict[str, Any]], AgUiTranslator]:
    from slr.api.app import _speak
    from slr.stream.engine import SessionRunner

    translator = AgUiTranslator()
    out: list[dict[str, Any]] = []

    async def emit(event: dict[str, Any]) -> None:
        out.extend(json.loads(wire(e)) for e in translator.translate(event))

    runner = SessionRunner(engine, emit)
    await runner.start()
    for text in utterances:
        await _speak(runner, text)
    await runner.close()
    return out, translator


async def test_a_turn_is_one_protocol_correct_run(engine):
    events, translator = await _turn_events(engine, ["What is the venue cancellation policy for workshops?"])
    state = check_stream(events)
    assert state == translator.state, "the client's state drifted from the server's"

    kinds = [e["type"] for e in events]
    # the session opens as its own short run, then the turn is one run
    assert kinds[:3] == ["RUN_STARTED", "STATE_SNAPSHOT", "RUN_FINISHED"]
    assert kinds[3] == "RUN_STARTED" and kinds[-1] == "RUN_FINISHED"
    assert "TOOL_CALL_RESULT" in kinds and "TEXT_MESSAGE_CONTENT" in kinds
    steps = [e["stepName"] for e in events if e["type"] == "STEP_STARTED"]
    assert steps == ["listen", "plan", "retrieve", "synthesise"]

    turn = next(iter(state["turns"].values()))
    assert turn["transcript"] and turn["decisions"] and turn["subQueries"]
    assert turn["fusion"]["hits"] and turn["latencyMs"] and turn["cost"]
    version = turn["versions"]["1"]
    assert version["fabricatedCitations"] == 0 and version["claims"]

    answer = "".join(e["delta"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT")
    assert "[Doc_" in answer
    searches = [e for e in events if e["type"] == "TOOL_CALL_START"]
    assert searches and all(e["toolCallName"] == "corpus_search" for e in searches)


async def test_a_late_detail_is_a_second_run_with_a_second_version(engine):
    events, _ = await _turn_events(
        engine,
        [
            "Summarize the travel reimbursement rule for an employee trip.",
            "The trip was international and the booking was made after travel.",
        ],
    )
    state = check_stream(events)
    assert [e["type"] for e in events].count("RUN_STARTED") == 3  # the session opening, then two turns
    second = list(state["turns"].values())[1]
    assert second["versions"]["2"]["parent"] == 1
    assert second["fusion"]["fullCorpusSearch"] is False


def test_a_superseded_run_is_closed_as_cancelled():
    t = AgUiTranslator()
    t.translate({"type": "session.ready", "sessionId": "s", "corpus": {"docs": 1, "chunks": 1}})
    t.translate({"type": "turn.start", "turnId": "t1"})
    out = [json.loads(wire(e)) for e in t.translate({"type": "turn.start", "turnId": "t2"})]
    assert out[0] == {"type": "STEP_FINISHED", "stepName": "listen"}
    assert out[1] == {"type": "RUN_FINISHED", "threadId": "s", "runId": "t1", "outcome": {"type": "cancelled"}}
    assert out[2]["type"] == "RUN_STARTED" and out[2]["runId"] == "t2"


def test_usage_rides_on_run_finished_per_model():
    t = AgUiTranslator()
    t.translate({"type": "session.ready", "sessionId": "s", "corpus": {"docs": 1, "chunks": 1}})
    t.translate({"type": "turn.start", "turnId": "t1"})
    cost = {"turnUsd": 0.001, "turnTokens": 150, "steps": [],
            "models": [{"model": "gpt-4o-mini", "inputTokens": 100, "outputTokens": 50}]}
    done = json.loads(wire(t.translate({"type": "turn.complete", "turnId": "t1",
                                        "latencyMs": {"firstRetrieval": None, "firstToken": 0, "complete": 1},
                                        "cost": cost})[-1]))
    assert done["usage"] == [{"model": "gpt-4o-mini", "inputTokens": 100, "outputTokens": 50, "totalTokens": 150}]
    offline = {**cost, "models": []}
    t.translate({"type": "turn.start", "turnId": "t2"})
    done = json.loads(wire(t.translate({"type": "turn.complete", "turnId": "t2",
                                        "latencyMs": {"firstRetrieval": None, "firstToken": 0, "complete": 1},
                                        "cost": offline})[-1]))
    assert "usage" not in done, "no model ran: usage is absent, not zero"


def test_json_pointer_keys_are_escaped():
    t = AgUiTranslator()
    t.translate({"type": "session.ready", "sessionId": "s", "corpus": {"docs": 1, "chunks": 1}})
    delta = t.translate({"type": "turn.start", "turnId": "a/b~c"})[1]
    assert delta.type == "STATE_DELTA"
    assert delta.delta[0].path == "/turns/a~1b~0c"
    assert "a/b~c" in t.state["turns"]
