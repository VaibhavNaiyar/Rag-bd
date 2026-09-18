"""One trace record per turn (gate G6).

Every turn — retrieved, refined or suppressed — emits every field in
``REQUIRED_FIELDS``. ``evals/gates.py`` fails G6 on a single missing field.
Records go to a JSONL file and an in-memory ring served by ``GET /trace``.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = (
    "trace_version",
    "session_id",
    "turn_id",
    "started_at",
    "mode",
    "controller",
    "utterance",
    "utterance_end_ms",
    "chunks",
    "decisions",
    "retrieval_events",
    "first_retrieval_ms",
    "before_utterance_end",
    "sub_queries",
    "decomposition",
    "retrieval",
    "fusion",
    "answer",
    "citations",
    "citation_support_rate",
    "fabricated_citations",
    "fabricated_citations_blocked",
    "uncertainty",
    "latency_ms",
    "cost",
    "models",
    "errors",
)


def new_record(session_id: str, turn_id: str, controller: str, models: dict[str, str]) -> dict[str, Any]:
    return {
        "trace_version": 1,
        "session_id": session_id,
        "turn_id": turn_id,
        "started_at": int(time.time() * 1000),
        "mode": None,
        "controller": controller,
        "utterance": "",
        "utterance_end_ms": None,
        "chunks": [],
        "decisions": [],
        "retrieval_events": [],
        "first_retrieval_ms": None,
        "before_utterance_end": None,
        "sub_queries": [],
        "decomposition": None,
        "retrieval": [],
        "fusion": None,
        "answer": None,
        "citations": [],
        "citation_support_rate": None,
        "fabricated_citations": 0,
        "fabricated_citations_blocked": 0,
        "uncertainty": [],
        "latency_ms": None,
        "cost": None,
        "models": models,
        "errors": [],
    }


def missing_fields(record: dict[str, Any]) -> list[str]:
    missing = [f for f in REQUIRED_FIELDS if f not in record]
    # Suppressed turns legitimately have no retrieval; everything else must be filled.
    nullable = {"first_retrieval_ms", "before_utterance_end", "decomposition", "fusion"}
    missing += [f for f in REQUIRED_FIELDS if f in record and record[f] is None and f not in nullable]
    return missing


class TraceSink:
    def __init__(self, path: str, ring: int = 500):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ring: deque[dict[str, Any]] = deque(maxlen=ring)
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=_default)
        with self._lock:
            self.ring.append(json.loads(line))
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def recent(self, limit: int = 50, session_id: str | None = None) -> list[dict[str, Any]]:
        items = [r for r in self.ring if session_id is None or r.get("session_id") == session_id]
        return items[-limit:]

    def find(self, turn_id: str, session_id: str | None = None) -> dict[str, Any] | None:
        for r in reversed(self.ring):
            if r.get("turn_id") == turn_id and (session_id is None or r.get("session_id") == session_id):
                return r
        return None


def _default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except ImportError:  # pragma: no cover
        pass
    return str(value)
