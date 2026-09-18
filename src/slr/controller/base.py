"""Controller protocol and the per-utterance state it reads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from slr.contracts import Decision, TranscriptChunk


@dataclass
class SessionView:
    """The only session facts a controller may see. Read-only, this session only."""

    has_answer: bool = False
    previous_utterance: str = ""
    previous_vec: np.ndarray | None = None
    previous_answer_text: str = ""


@dataclass
class UtteranceState:
    session: SessionView
    chunks: list[TranscriptChunk] = field(default_factory=list)
    prefix: str = ""
    vecs: list[np.ndarray] = field(default_factory=list)
    stable_run: int = 0
    provisional_count: int = 0
    #: words of the prefix already covered by a launched provisional search
    covered_words: int = 0
    trigger_vec: np.ndarray | None = None
    #: embedding of the segment that triggered the provisional search
    trigger_segment_vec: np.ndarray | None = None
    active_provisional: str | None = None
    mode: Decision = Decision.WAIT
    last_cos: float = 0.0

    @property
    def words(self) -> list[str]:
        return self.prefix.split()

    @property
    def uncovered(self) -> str:
        return " ".join(self.words[self.covered_words :])


@dataclass
class ControllerVerdict:
    decision: Decision
    reason: str
    confidence: float
    #: provisional search to launch now, if any
    query: str | None = None
    #: cancel the active provisional search
    cancel: bool = False
    cancel_reason: str = ""
    signals: dict = field(default_factory=dict)


class Controller(Protocol):
    name: str

    async def observe(self, chunk: TranscriptChunk, state: UtteranceState) -> ControllerVerdict: ...

    async def finalize(self, state: UtteranceState) -> ControllerVerdict: ...
