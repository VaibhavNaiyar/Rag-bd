"""ModelController — the ablation arm.

One small-model call per transcript chunk returning
``{decision, reason, confidence, query}``. It is slower and costs money on
every chunk; the benchmark report publishes exactly how much.
"""

from __future__ import annotations

import logging

from slr.config import Settings
from slr.contracts import Decision, TranscriptChunk
from slr.controller.base import ControllerVerdict, UtteranceState
from slr.controller.rules import RuleController
from slr.llm import ChatModel, parse_json, render_prompt
from slr.retrieval.embed import Embedder
from slr.retrieval.store import Index
from slr.telemetry.cost import UsageLedger

log = logging.getLogger(__name__)


class ModelController:
    name = "model"

    def __init__(self, model: ChatModel, index: Index, embedder: Embedder, settings: Settings):
        self.model = model
        self.s = settings
        self.embedder = embedder
        # Finalisation and embeddings are shared with the rule arm so the only
        # difference between arms is who makes the per-chunk decision.
        self._rules = RuleController(index, embedder, settings)
        self.ledger: UsageLedger | None = None

    async def observe(self, chunk: TranscriptChunk, state: UtteranceState) -> ControllerVerdict:
        state.chunks.append(chunk)
        state.prefix = f"{state.prefix} {chunk.text}".strip()
        vec = self.embedder.embed([state.prefix], kind="text")[0]
        state.vecs.append(vec)
        system, user = render_prompt(
            self.s.prompts_dir,
            "controller",
            previous_utterance=state.session.previous_utterance or "(none)",
            has_answer="yes" if state.session.has_answer else "no",
            prefix=state.prefix,
        )
        try:
            raw = await self.model.complete(
                system, user, ledger=self.ledger, step="controller", json_mode=True, max_tokens=80
            )
            data = parse_json(raw)
            decision = Decision(str(data.get("decision", "wait")).lower())
            reason = str(data.get("reason", "model"))[:40]
            confidence = float(data.get("confidence", 0.5))
            query = str(data.get("query", "")).strip() or None
        except Exception as exc:
            log.warning("model controller failed: %s", exc)
            return ControllerVerdict(Decision.WAIT, "controller_error", 0.0)

        launch = None
        if decision in (Decision.RETRIEVE, Decision.REFINE) and state.provisional_count < self.s.max_provisional:
            # The model re-announces "retrieve" on every chunk; only a query for
            # uncovered words launches a new search.
            if query and state.covered_words < len(state.words):
                launch = query
                state.covered_words = len(state.words)
                state.provisional_count += 1
                state.trigger_vec = vec
        if decision == Decision.REFINE:
            state.mode = Decision.REFINE
        if decision == Decision.SUPPRESS:
            state.mode = Decision.SUPPRESS
        cancel = decision == Decision.SUPPRESS and state.active_provisional is not None
        return ControllerVerdict(
            decision, reason, confidence, query=launch, cancel=cancel, cancel_reason="presentation_only" if cancel else ""
        )

    async def finalize(self, state: UtteranceState) -> ControllerVerdict:
        return await self._rules.finalize(state)
