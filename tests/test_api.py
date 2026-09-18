"""The HTTP and WebSocket surface, through the real app."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from tests.test_agui import check_stream


@pytest.fixture(scope="module")
def client(settings):
    from slr.api.app import app

    with TestClient(app) as c:
        yield c


def test_health_reports_corpus_and_models(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["corpus"]["chunks"] > 0 and body["corpus"]["docs"] > 0
    assert body["models"]["embedder"] and body["models"]["controller"] == "rule"


def test_fixtures_are_listed_for_replay(client):
    names = client.get("/fixtures").json()["fixtures"]
    assert {"compound_01", "late_detail_01", "presentation_01", "unanswerable_01"} <= set(names)


def test_query_endpoint_runs_the_streaming_path(client):
    body = client.post("/query", json={"utterance": "What is the venue cancellation policy?"}).json()
    assert body["answer"].strip()
    trace = body["trace"]
    assert trace["mode"] == "retrieve"
    assert trace["sub_queries"] and trace["citation_support_rate"] is not None
    # even here the controller ran: the text was streamed as chunks, not submitted whole
    assert len(trace["chunks"]) > 1
    assert trace["decisions"], "the controller was bypassed"


def test_trace_endpoints_expose_the_record(client):
    client.post("/query", json={"utterance": "How many days of annual leave do employees get?"})
    traces = client.get("/trace?limit=5").json()["traces"]
    assert traces
    turn_id = traces[-1]["turn_id"]
    one = client.get(f"/trace/{turn_id}").json()
    assert one["turn_id"] == turn_id and one["cost"]["turnUsd"] >= 0
    assert client.get("/trace/nope_not_a_turn").status_code == 404


def _until(ws, kind: str) -> list[dict]:
    """Read AG-UI frames up to and including the first of ``kind``."""
    seen = []
    while True:
        event = ws.receive_json()
        seen.append(event)
        if event["type"] == kind:
            return seen


def _session(ws) -> str:
    """The opening run: RUN_STARTED, the session's STATE_SNAPSHOT, RUN_FINISHED."""
    opening = _until(ws, "RUN_FINISHED")
    assert [e["type"] for e in opening] == ["RUN_STARTED", "STATE_SNAPSHOT", "RUN_FINISHED"]
    return opening[1]["snapshot"]["session"]["id"]


def test_unknown_client_events_are_rejected_not_crashed(client):
    with client.websocket_connect("/stream") as ws:
        _session(ws)
        ws.send_json({"type": "nonsense"})
        assert ws.receive_json() == {"type": "RUN_ERROR", "message": "unknown client event", "code": "bad_request"}
        ws.send_text("{not json")
        assert ws.receive_json()["code"] == "bad_json"


def test_websocket_streams_a_turn_as_ag_ui(client):
    with client.websocket_connect("/stream") as ws:
        opening = _until(ws, "RUN_FINISHED")
        ws.send_json({"type": "utterance.start"})
        for piece in ["What is the", "venue cancellation", "policy for workshops?"]:
            ws.send_json({"type": "utterance.chunk", "text": piece})
        ws.send_json({"type": "utterance.end"})
        events = opening + _until(ws, "RUN_FINISHED")

        state = check_stream(events)
        turn = next(iter(state["turns"].values()))
        assert len(turn["transcript"]) == 3
        assert turn["decisions"] and turn["utteranceEndMs"] is not None
        assert turn["subQueries"] and turn["fusion"]["hits"]
        assert turn["versions"]["1"]["fabricatedCitations"] == 0
        answer = "".join(e["delta"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT")
        assert "[Doc_" in answer, "the streamed answer carried no citation"


def test_new_session_clears_state(client):
    with client.websocket_connect("/stream") as ws:
        first = _session(ws)
        ws.send_json({"type": "session.new"})
        assert _session(ws) != first


def test_replay_drives_the_same_path_as_a_live_utterance(client):
    with client.websocket_connect("/stream") as ws:
        _session(ws)
        ws.send_json({"type": "replay", "fixture": "single_01", "speed": 20})
        kinds = {e["type"] for e in _until(ws, "RUN_FINISHED")}
        assert {"RUN_STARTED", "STEP_STARTED", "STATE_DELTA", "TOOL_CALL_START", "TEXT_MESSAGE_CONTENT"} <= kinds


def test_an_unknown_fixture_is_an_error_not_a_crash(client):
    with client.websocket_connect("/stream") as ws:
        _session(ws)
        ws.send_json({"type": "replay", "fixture": "../../etc/passwd"})
        assert ws.receive_json()["type"] == "RUN_ERROR"
        # the socket still works afterwards
        ws.send_json({"type": "utterance.start"})
        assert ws.receive_json()["type"] == "RUN_STARTED"


def _sse(response) -> list[dict]:
    return [json.loads(line[len("data: "):]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_agui_endpoint_serves_a_standard_client_over_sse(client):
    run = {
        "threadId": "thread-1",
        "runId": "run-7",
        "state": {},
        "messages": [{"id": "m1", "role": "user", "content": "What is the venue cancellation policy?"}],
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }
    response = client.post("/agui", json=run, headers={"accept": "text/event-stream"})
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
    events = _sse(response)
    check_stream(events)
    assert events[0] == {"type": "RUN_STARTED", "threadId": "thread-1", "runId": "run-7"}
    assert events[1]["type"] == "STATE_SNAPSHOT"
    assert events[-1]["type"] == "RUN_FINISHED" and events[-1]["runId"] == "run-7"


def test_agui_endpoint_refines_with_earlier_messages_as_history(client):
    run = {
        "threadId": "thread-2",
        "runId": "run-2",
        "state": {},
        "messages": [
            {"id": "m1", "role": "user", "content": "Summarize the travel reimbursement rule for an employee trip."},
            {"id": "a1", "role": "assistant", "content": "(earlier answer)"},
            {"id": "m2", "role": "user", "content": "The trip was international and the booking was made after travel."},
        ],
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }
    events = _sse(client.post("/agui", json=run))
    state = check_stream(events)
    assert [e["type"] for e in events].count("RUN_STARTED") == 1, "history replays silently"
    latest = list(state["turns"].values())[-1]
    assert latest["versions"]["2"]["parent"] == 1
    assert latest["fusion"]["fullCorpusSearch"] is False


def test_agui_endpoint_rejects_a_run_with_nothing_said(client):
    run = {"threadId": "t", "runId": "r", "state": {}, "messages": [], "tools": [], "context": [], "forwardedProps": {}}
    assert client.post("/agui", json=run).status_code == 422


def test_agui_endpoint_rejects_a_body_that_is_not_run_input(client):
    assert client.post("/agui", json={"hello": "world"}).status_code == 422
