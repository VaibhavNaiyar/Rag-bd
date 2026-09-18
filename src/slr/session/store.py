"""Ephemeral, scope-bound session state.

Pattern from Lattice's ``memory/session.py``: the scope is bound at
construction and no method accepts a caller-supplied session id, so one
session can never read another's state. Nothing is persisted; ``clear`` runs
when the socket closes. There is no cross-session profile to leak because none
exists.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np

from slr.contracts import AnswerVersion, Hit, SubQuery


@dataclass(frozen=True)
class Topic:
    """The answer lineage a refinement or restructure builds on."""

    utterance: str
    utterance_vec: np.ndarray | None
    sub_queries: tuple[SubQuery, ...]
    hits: tuple[Hit, ...]
    answer: AnswerVersion


class SessionStore:
    def __init__(self) -> None:
        self._id = f"s_{uuid.uuid4().hex[:12]}"
        self._topic: Topic | None = None
        self._turns = 0

    @property
    def id(self) -> str:
        return self._id

    @property
    def topic(self) -> Topic | None:
        return self._topic

    @property
    def has_answer(self) -> bool:
        return self._topic is not None and bool(self._topic.answer.claims)

    def next_turn_id(self) -> str:
        self._turns += 1
        return f"t{self._turns}"

    def next_version(self) -> tuple[int, int | None]:
        if self._topic is None:
            return 1, None
        v = self._topic.answer.version
        return v + 1, v

    def commit(self, topic: Topic) -> None:
        self._topic = topic

    def clear(self) -> None:
        self._topic = None
        self._turns = 0
