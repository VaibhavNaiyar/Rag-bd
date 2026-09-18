"""The HTTP and WebSocket surface, through the real app."""

from __future__ import annotations


import pytest
from fastapi.testclient import TestClient

from slr.contracts import validate_event


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


def test_unknown_client_events_are_rejected_not_crashed(client):
    with client.websocket_connect("/stream") as ws:
        assert ws.receive_json()["type"] == "session.ready"
        ws.send_json({"type": "nonsense"})
        assert ws.receive_json()["code"] == "bad_request"
        ws.send_text("{not json")
        assert ws.receive_json()["code"] == "bad_json"


def test_websocket_streams_a_turn_end_to_end(client):
    with client.websocket_connect("/stream") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "session.ready" and ready["sessionId"]

        ws.send_json({"type": "utterance.start"})
        for piece in ["What is the", "venue cancellation", "policy for workshops?"]:
            ws.send_json({"type": "utterance.chunk", "text": piece})
        ws.send_json({"type": "utterance.end"})

        seen: list[dict] = []
        while True:
            event = ws.receive_json()
            validate_event(event)
            seen.append(event)
            if event["type"] == "turn.complete":
                break

        kinds = [e["type"] for e in seen]
        assert kinds.count("transcript.chunk") == 3
        assert "controller.decision" in kinds and "utterance.end" in kinds
        assert "subqueries" in kinds and "fusion.final" in kinds
        assert "answer.version" in kinds
        answer = "".join(e["text"] for e in seen if e["type"] == "answer.token")
        assert "[Doc_" in answer, "the streamed answer carried no citation"

        version = [e for e in seen if e["type"] == "answer.version"][-1]
        assert version["fabricatedCitations"] == 0


def test_new_session_clears_state(client):
    with client.websocket_connect("/stream") as ws:
        first = ws.receive_json()["sessionId"]
        ws.send_json({"type": "session.new"})
        while True:
            event = ws.receive_json()
            if event["type"] == "session.ready":
                assert event["sessionId"] != first
                break


def test_replay_drives_the_same_path_as_a_live_utterance(client):
    with client.websocket_connect("/stream") as ws:
        ws.receive_json()
        ws.send_json({"type": "replay", "fixture": "single_01", "speed": 20})
        kinds = set()
        while "turn.complete" not in kinds:
            kinds.add(ws.receive_json()["type"])
        assert {"turn.start", "transcript.chunk", "controller.decision", "retrieval.started"} <= kinds


def test_an_unknown_fixture_is_an_error_not_a_crash(client):
    with client.websocket_connect("/stream") as ws:
        ws.receive_json()
        ws.send_json({"type": "replay", "fixture": "../../etc/passwd"})
        event = ws.receive_json()
        assert event["type"] == "error"
        # the socket still works afterwards
        ws.send_json({"type": "utterance.start"})
        assert ws.receive_json()["type"] == "turn.start"
