"""FastAPI surface.

* ``WS  /stream``  — the demo and eval path: transcript chunks in, events out
* ``POST /query``  — testing / Swagger only; streams the text as chunks internally
* ``GET  /health`` — readiness, corpus and model identity
* ``GET  /trace``  — recent per-turn trace records (``/trace/{turn_id}`` for one)
* ``GET  /fixtures`` — replayable fixture names

The built console (Next.js static export) is mounted last so it can never
shadow an API route.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from slr import __version__
from slr.config import get_settings
from slr.contracts import CLIENT_EVENTS
from slr.stream.engine import Engine, SessionRunner
from slr.stream.simulator import list_fixtures

log = logging.getLogger("slr.api")
STATE: dict[str, Any] = {}


def _engine() -> Engine:
    engine = STATE.get("engine")
    if engine is None:
        raise HTTPException(503, "engine is still loading")
    return engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    if not (Path(settings.index_dir) / "manifest.json").exists():
        from slr.ingest.build_index import build

        log.info("no index at %s — building from %s", settings.index_dir, settings.corpus_dir)
        await asyncio.to_thread(build, settings.corpus_dir, settings.index_dir)
    STATE["engine"] = await asyncio.to_thread(Engine.from_settings, settings)
    engine: Engine = STATE["engine"]
    # Warm every model once so the first real turn is not a cold start.
    await asyncio.to_thread(engine.retriever.search_many, [_warm_query()])
    await asyncio.to_thread(engine.verifier.support, "warm up", ["warm up text"])
    log.info("engine ready: %s", engine.models_info())
    yield
    STATE.clear()


def _warm_query():
    from slr.contracts import SubQuery

    return SubQuery(id="warm", text="warm up query")


app = FastAPI(title="Streaming Live RAG", version=__version__, lifespan=lifespan)


class QueryIn(BaseModel):
    utterance: str = Field(..., min_length=1, max_length=4000)
    #: replay speed for the internal chunk stream; 0 sends chunks without delay
    speed: float = Field(0, ge=0, le=20)
    #: optional earlier turns in the same (ephemeral) session, e.g. for refinement tests
    history: list[str] = Field(default_factory=list, max_length=5)


@app.get("/health")
async def health() -> dict[str, Any]:
    engine = STATE.get("engine")
    if engine is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {
        "status": "ok",
        "version": __version__,
        "corpus": engine.corpus_info(),
        "models": engine.models_info(),
    }


@app.get("/fixtures")
async def fixtures() -> dict[str, list[str]]:
    return {"fixtures": sorted(list_fixtures(get_settings().fixtures_dir))}


@app.get("/trace")
async def traces(limit: int = 20, session_id: str | None = None) -> dict[str, Any]:
    engine = _engine()
    return {"traces": engine.sink.recent(max(1, min(limit, 200)), session_id)}


@app.get("/trace/{turn_id}")
async def trace(turn_id: str, session_id: str | None = None) -> dict[str, Any]:
    record = _engine().sink.find(turn_id, session_id)
    if record is None:
        raise HTTPException(404, "no such turn in the recent trace ring")
    return record


@app.post("/query")
async def query(body: QueryIn) -> dict[str, Any]:
    """Runs the full streaming path in-process. Not the demo path — the WebSocket is."""
    engine = _engine()
    events: list[dict[str, Any]] = []

    async def collect(event: dict[str, Any]) -> None:
        events.append(event)

    runner = SessionRunner(engine, collect)
    await runner.start()
    turns = [{"utterance": u} for u in [*body.history, body.utterance]]
    if body.speed > 0:
        await runner.play_turns(turns, body.speed)
    else:
        from slr.stream.simulator import chunk_utterance

        for spec in turns:
            await runner.utterance_start()
            for text, _ in chunk_utterance(spec["utterance"]):
                await runner.utterance_chunk(text)
            await runner.utterance_end()
    await runner.close()
    last = runner.completed[-1] if runner.completed else {}
    answer = "".join(e["text"] for e in events if e["type"] == "answer.token" and e["turnId"] == last.get("turn_id"))
    return {
        "answer": answer,
        "trace": last,
        "events": [e for e in events if e["type"] != "answer.token"],
    }


@app.websocket("/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    engine = STATE.get("engine")
    if engine is None:
        await ws.send_json({"type": "error", "code": "loading", "message": "engine is still loading"})
        await ws.close()
        return

    send_lock = asyncio.Lock()

    async def emit(event: dict[str, Any]) -> None:
        async with send_lock:
            await ws.send_text(json.dumps(event, ensure_ascii=False))

    runner = SessionRunner(engine, emit)
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def worker() -> None:
        while True:
            message = await queue.get()
            if message is None:
                return
            await runner.handle(message)

    task = asyncio.create_task(worker())
    try:
        await runner.start()
        while True:
            raw = await ws.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await emit({"type": "error", "code": "bad_json", "message": "frame is not JSON"})
                continue
            if not isinstance(message, dict) or message.get("type") not in CLIENT_EVENTS:
                await emit({"type": "error", "code": "bad_request", "message": "unknown client event"})
                continue
            if message["type"] == "session.new":
                # Out of band: drop queued work for the old session first.
                while not queue.empty():
                    queue.get_nowait()
            await queue.put(message)
    except WebSocketDisconnect:
        pass
    finally:
        await queue.put(None)
        task.cancel()
        await runner.close()


_static = Path(get_settings().static_dir)
if (_static / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(_static), html=True), name="console")
else:

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"service": "streaming-live-rag", "docs": "/docs", "stream": "/stream"}


def main() -> None:
    import uvicorn

    uvicorn.run("slr.api.app:app", host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
