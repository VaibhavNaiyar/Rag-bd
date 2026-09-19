"""BatchController — the baseline pipeline the benchmark compares against.

A conventional RAG turn: nothing happens while the user speaks, and when they
stop the whole utterance is searched. It never suppresses a presentation-only
turn and never treats a late detail as a refinement, because a batch system
has no notion of either. Paired with ``SLR_DECOMPOSE=off`` (one search for
the whole utterance), it is the system the theme describes as the problem.
"""

from __future__ import annotations

import asyncio

from slr.contracts import Decision, TranscriptChunk
from slr.controller.base import ControllerVerdict, UtteranceState
from slr.retrieval.embed import Embedder


class BatchController:
    name = "batch"

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    async def observe(self, chunk: TranscriptChunk, state: UtteranceState) -> ControllerVerdict:
        state.chunks.append(chunk)
        state.prefix = f"{state.prefix} {chunk.text}".strip()
        return ControllerVerdict(Decision.WAIT, "waiting_for_utterance_end", 1.0)

    async def finalize(self, state: UtteranceState) -> ControllerVerdict:
        if state.prefix:
            vec = await asyncio.to_thread(self.embedder.embed, [state.prefix], "text")
            state.vecs.append(vec[0])
        return ControllerVerdict(Decision.RETRIEVE, "utterance_complete", 1.0)
