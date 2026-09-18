"""Engine events -> AG-UI protocol events (https://docs.ag-ui.com).

The engine speaks its own event vocabulary (``slr.contracts.WIRE_SCHEMA``); the
trace, the eval harness and the gates all read that. This module is the
boundary where it becomes the standard protocol a browser, or any AG-UI
client, reads:

* a turn is one run: ``RUN_STARTED`` ... ``RUN_FINISHED`` (or ``RUN_ERROR``),
  with ``threadId`` the session and ``runId`` the turn;
* the pipeline stages are steps, the four boxes of the theme guide's
  architecture: ``listen`` (the controller, while the speaker talks), ``plan``
  (routing and decomposition), ``retrieve`` (search, fusion, rerank) and
  ``synthesise`` (the grounded answer);
* a retrieval is one ``corpus_search`` tool call: ``TOOL_CALL_START`` /
  ``ARGS`` ``{query, trigger, atMs}`` / ``END``, then ``TOOL_CALL_RESULT``;
* answer tokens are ``TEXT_MESSAGE_START`` / ``CONTENT`` / ``END``, one
  message per answer version (``<turnId>:v<n>``);
* everything else is shared state: ``STATE_SNAPSHOT`` when a session starts,
  ``STATE_DELTA`` (RFC 6902 JSON Patch) after that;
* token usage rides on ``RUN_FINISHED``, per model, in the protocol's own
  ``TokenUsage`` shape.

A new session is announced as a short run of its own (``RUN_STARTED``,
``STATE_SNAPSHOT``, ``RUN_FINISHED``), so every event on the socket sits inside
a run, the shape a strict AG-UI client verifies.

Shared state (the shape the console renders)::

    {"session": {"id": ..., "corpus": {...}},
     "turns": {"<turnId>": {"transcript": [{text, atMs}], "utteranceEndMs": int | None,
                            "decisions": [...], "subQueries": [...], "fusion": {...} | None,
                            "versions": {"<n>": {...}}, "latencyMs": {...} | None,
                            "cost": {...} | None}}}

Only AG-UI leaves the server. The client -> server direction stays three input
messages (``utterance.start`` / ``chunk`` / ``end``), because AG-UI has no
event for input that arrives while a run is already under way, which is
exactly what full-duplex early retrieval needs.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from ag_ui.core import (
    BaseEvent,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateDeltaEvent,
    StateSnapshotEvent,
    StepFinishedEvent,
    StepStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    TokenUsage,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)

SEARCH_TOOL = "corpus_search"
#: engine event -> the step that starts once it has been reported (None: the
#: last one simply ends). Whatever step is open ends first, so a suppressed
#: turn goes listen -> plan -> done without inventing a retrieve step.
STEP_AFTER = {
    "utterance.end": "plan",
    "subqueries": "retrieve",
    "fusion.final": "synthesise",
    "answer.version": None,
}


def _pointer(*parts: Any) -> str:
    """RFC 6901 path from raw keys."""
    return "".join("/" + str(p).replace("~", "~0").replace("/", "~1") for p in parts)


def blank_turn() -> dict[str, Any]:
    return {
        "transcript": [],
        "utteranceEndMs": None,
        "decisions": [],
        "subQueries": [],
        "fusion": None,
        "versions": {},
        "latencyMs": None,
        "cost": None,
    }


def _usage(cost: dict[str, Any]) -> list[TokenUsage] | None:
    """The turn's model spend in AG-UI's shape. None when no model ran: absent, not zero."""
    entries = [
        TokenUsage(
            model=m["model"],
            input_tokens=m["inputTokens"],
            output_tokens=m["outputTokens"],
            total_tokens=m["inputTokens"] + m["outputTokens"],
        )
        for m in cost.get("models", [])
    ]
    return entries or None


class AgUiTranslator:
    """One per session. Feed it engine events in order; it returns AG-UI events.

    It keeps the shared state it has described, so a client that connects
    mid-session, or an HTTP run that replays earlier turns silently, can be
    sent one ``STATE_SNAPSHOT`` instead of the whole history.
    """

    def __init__(self, thread_id: str | None = None) -> None:
        #: when set, overrides the engine's session id as the AG-UI thread id
        self.thread_override = thread_id
        #: engine turn id -> run id the client asked for (HTTP runs)
        self.run_ids: dict[str, str] = {}
        #: the client's run id for the NEXT turn, which the engine has not numbered yet
        self.next_run_id: str | None = None
        self.state: dict[str, Any] = {"session": None, "turns": {}}
        self._session_id = ""
        self._run: str | None = None  # the engine turn whose run is open
        self._step: str | None = None  # the open pipeline step
        self._message: str | None = None  # the open text message id
        self._calls: set[str] = set()  # tool calls already opened
        self._results: dict[str, int] = {}  # tool call id -> results sent

    # ------------------------------------------------------------ helpers

    @property
    def thread_id(self) -> str:
        return self.thread_override or self._session_id

    def run_id(self, turn_id: str) -> str:
        return self.run_ids.get(turn_id, turn_id)

    def snapshot(self) -> StateSnapshotEvent:
        return StateSnapshotEvent(snapshot=copy.deepcopy(self.state))

    def _patch(self, *ops: dict[str, Any]) -> StateDeltaEvent:
        for op in ops:
            self._apply(op)
        return StateDeltaEvent(delta=list(ops))

    def _apply(self, op: dict[str, Any]) -> None:
        """Keep our copy in step with what the client applies. Only add/replace are emitted."""
        keys = [k.replace("~1", "/").replace("~0", "~") for k in op["path"].split("/")[1:]]
        target = self.state
        for key in keys[:-1]:
            target = target[int(key)] if isinstance(target, list) else target[key]
        last = keys[-1]
        if isinstance(target, list):
            if last == "-":
                target.append(op["value"])
            else:
                target[int(last)] = op["value"]
        else:
            target[last] = op["value"]

    def _open_call(self, call_id: str, args: dict[str, Any]) -> list[BaseEvent]:
        self._calls.add(call_id)
        return [
            ToolCallStartEvent(tool_call_id=call_id, tool_call_name=SEARCH_TOOL),
            ToolCallArgsEvent(tool_call_id=call_id, delta=json.dumps(args, ensure_ascii=False)),
            ToolCallEndEvent(tool_call_id=call_id),
        ]

    def _result(self, call_id: str, content: dict[str, Any]) -> ToolCallResultEvent:
        n = self._results.get(call_id, 0) + 1
        self._results[call_id] = n
        return ToolCallResultEvent(
            message_id=f"{call_id}:result" + (f":{n}" if n > 1 else ""),
            tool_call_id=call_id,
            role="tool",
            content=json.dumps(content, ensure_ascii=False),
        )

    def _close_message(self) -> list[BaseEvent]:
        if self._message is None:
            return []
        message, self._message = self._message, None
        return [TextMessageEndEvent(message_id=message)]

    def _close_step(self) -> list[BaseEvent]:
        if self._step is None:
            return []
        step, self._step = self._step, None
        return [StepFinishedEvent(step_name=step)]

    def _open_step(self, name: str) -> list[BaseEvent]:
        self._step = name
        return [StepStartedEvent(step_name=name)]

    def _end_run(self) -> list[BaseEvent]:
        """What must close before a run can: the text message, then the step."""
        out = [*self._close_message(), *self._close_step()]
        self._run = None
        return out

    def _query_of(self, turn_id: str, call_id: str) -> str:
        for item in self.state["turns"].get(turn_id, {}).get("subQueries", []):
            if item["id"] == call_id:
                return item["text"]
        return ""

    # ------------------------------------------------------------ translation

    def translate(self, event: dict[str, Any]) -> list[BaseEvent]:
        out = self._translate(event)
        kind = event["type"]
        if kind in STEP_AFTER and self._run is not None and self._step != STEP_AFTER[kind]:
            out += self._close_step()
            if STEP_AFTER[kind]:
                out += self._open_step(STEP_AFTER[kind])
        return out

    def _translate(self, event: dict[str, Any]) -> list[BaseEvent]:
        kind = event["type"]
        turn_id = event.get("turnId", "")
        turn = _pointer("turns", turn_id)

        if kind == "session.ready":
            out = self.cancel_open_run()
            self._session_id = event["sessionId"]
            self.state = {"session": {"id": event["sessionId"], "corpus": event["corpus"]}, "turns": {}}
            self._calls.clear()
            self._results.clear()
            opening = f"{event['sessionId']}:open"
            return [
                *out,
                RunStartedEvent(thread_id=self.thread_id, run_id=opening),
                self.snapshot(),
                RunFinishedEvent(thread_id=self.thread_id, run_id=opening),
            ]

        if kind == "turn.start":
            out = self.cancel_open_run()
            self._run = turn_id
            if self.next_run_id:
                self.run_ids[turn_id], self.next_run_id = self.next_run_id, None
            out.append(RunStartedEvent(thread_id=self.thread_id, run_id=self.run_id(turn_id)))
            out.append(self._patch({"op": "add", "path": turn, "value": blank_turn()}))
            return [*out, *self._open_step("listen")]

        if kind == "transcript.chunk":
            chunk = {"text": event["text"], "atMs": event["atMs"]}
            return [self._patch({"op": "add", "path": f"{turn}/transcript/-", "value": chunk})]

        if kind == "controller.decision":
            decision = {k: event[k] for k in ("decision", "reason", "atMs", "confidence") if k in event}
            return [self._patch({"op": "add", "path": f"{turn}/decisions/-", "value": decision})]

        if kind == "retrieval.started":
            args = {"query": event.get("query", ""), "trigger": event["trigger"], "atMs": event["atMs"]}
            return self._open_call(event["subQueryId"], args)

        if kind == "retrieval.cancelled":
            call = event["subQueryId"]
            return [self._result(call, {"cancelled": True, "reason": event["reason"]})]

        if kind == "utterance.end":
            return [self._patch({"op": "replace", "path": f"{turn}/utteranceEndMs", "value": event["atMs"]})]

        if kind == "subqueries":
            return [self._patch({"op": "replace", "path": f"{turn}/subQueries", "value": event["items"]})]

        if kind == "retrieval.result":
            call = event["subQueryId"]
            out: list[BaseEvent] = []
            if call not in self._calls:
                # A decomposed sub-query answered by a provisional search that was
                # already running: it never had a start of its own.
                out += self._open_call(call, {"query": self._query_of(turn_id, call), "reused": True})
            content = {"candidates": event["candidates"], "kept": event["kept"], "reused": bool(event.get("reused"))}
            out.append(self._result(call, content))
            return out

        if kind == "fusion.final":
            fusion = {k: event[k] for k in ("hits", "quotaApplied", "fullCorpusSearch")}
            return [self._patch({"op": "replace", "path": f"{turn}/fusion", "value": fusion})]

        if kind == "answer.token":
            if not event["text"]:
                return []
            message = f"{turn_id}:v{event['version']}"
            out = []
            if self._message != message:
                out += self._close_message()
                self._message = message
                out.append(TextMessageStartEvent(message_id=message, role="assistant"))
            out.append(TextMessageContentEvent(message_id=message, delta=event["text"]))
            return out

        if kind == "answer.version":
            version = {k: v for k, v in event.items() if k not in ("type", "turnId", "version")}
            out = self._close_message()
            out.append(self._patch({"op": "add", "path": f"{turn}/versions/{event['version']}", "value": version}))
            return out

        if kind == "turn.complete":
            delta = self._patch(
                {"op": "replace", "path": f"{turn}/latencyMs", "value": event["latencyMs"]},
                {"op": "replace", "path": f"{turn}/cost", "value": event["cost"]},
            )
            run = self.run_id(turn_id)
            return [
                delta,
                *self._end_run(),
                RunFinishedEvent(thread_id=self.thread_id, run_id=run, usage=_usage(event["cost"])),
            ]

        if kind == "error":
            out = self._end_run() if turn_id and turn_id == self._run else []
            return [*out, RunErrorEvent(message=event["message"], code=event["code"])]

        raise ValueError(f"no AG-UI mapping for engine event {kind!r}")

    def cancel_open_run(self) -> list[BaseEvent]:
        """Close a run the engine abandoned (a new session, or a turn superseded mid-flight)."""
        if self._run is None:
            return []
        run = self.run_id(self._run)
        return [
            *self._end_run(),
            RunFinishedEvent(thread_id=self.thread_id, run_id=run, outcome={"type": "cancelled"}),
        ]


def wire(event: BaseEvent) -> str:
    """One AG-UI event as the JSON a client reads: camelCase, unset fields left out."""
    return event.model_dump_json(by_alias=True, exclude_none=True)
