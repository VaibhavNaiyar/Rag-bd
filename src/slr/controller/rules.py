"""RuleController — the default. Four signals, no model call.

Per chunk:

* embed the growing prefix (one small-model forward pass, cached)
* SUPPRESS if a prior answer exists and the prefix is presentation-only
* cancel the provisional search if the meaning moved away from where it fired
* RETRIEVE (or REFINE) once the uncovered part of the prefix is sufficient and
  either its meaning has stabilised or a clause just closed
* otherwise WAIT, with the reason that blocked the trigger

A second provisional search may fire for a later clause (``max_provisional``),
which is how a compound request starts retrieving its second intent before
the speaker finishes.
"""

from __future__ import annotations

import asyncio
import time

from slr.config import Settings
from slr.contracts import Decision, TranscriptChunk
from slr.controller import signals as sig
from slr.controller.base import ControllerVerdict, UtteranceState
from slr.retrieval.embed import Embedder
from slr.retrieval.store import Index


class RuleController:
    name = "rule"

    def __init__(self, index: Index, embedder: Embedder, settings: Settings):
        self.index = index
        self.embedder = embedder
        self.s = settings

    async def _embed(self, text: str):
        vecs = await asyncio.to_thread(self.embedder.embed, [text], "text")
        return vecs[0]

    async def observe(self, chunk: TranscriptChunk, state: UtteranceState) -> ControllerVerdict:
        started = time.perf_counter()
        s = self.s
        state.chunks.append(chunk)
        state.prefix = f"{state.prefix} {chunk.text}".strip()
        vec = await self._embed(state.prefix)
        state.vecs.append(vec)
        cos_prev = sig.stability(state.vecs)
        state.last_cos = cos_prev
        state.stable_run = state.stable_run + 1 if cos_prev >= s.tau_stab else 0
        info: dict = {"cos_prev": round(cos_prev, 3), "stable_run": state.stable_run}

        def verdict(decision, reason, confidence, **kw) -> ControllerVerdict:
            info["ms"] = round((time.perf_counter() - started) * 1000, 2)
            return ControllerVerdict(decision, reason, round(confidence, 3), signals=info, **kw)

        session = state.session
        # 3. suppression
        if session.has_answer:
            presentation, detail = sig.is_presentation_only(state.prefix, session.previous_answer_text, self.index)
            info["suppression"] = detail
            if presentation:
                state.mode = Decision.SUPPRESS
                cancel = state.active_provisional is not None
                return verdict(
                    Decision.SUPPRESS,
                    "presentation_restructure",
                    0.9,
                    cancel=cancel,
                    cancel_reason="presentation_only" if cancel else "",
                )
            if state.mode == Decision.SUPPRESS:
                state.mode = Decision.WAIT  # new content arrived after all

        # Thrash guard: has the meaning moved away from where the search fired?
        # Measured on the words spoken SINCE the trigger, not on the whole
        # prefix — a growing prefix dilutes a topic change until it vanishes.
        if state.active_provisional and state.trigger_segment_vec is not None and state.uncovered.strip():
            segment_vec = await self._embed(state.uncovered)
            drift = float(segment_vec @ state.trigger_segment_vec)
            info["cos_trigger"] = round(drift, 3)
            if drift < s.tau_shift:
                state.trigger_vec = None
                state.trigger_segment_vec = None
                state.covered_words = 0  # the whole prefix needs searching again
                return verdict(Decision.WAIT, "topic_shift", 1 - drift, cancel=True, cancel_reason="topic_shift")

        # 4. refinement
        refine = False
        if session.has_answer:
            refine, detail = sig.refinement(state.prefix, vec, session.previous_vec, s.tau_refine)
            info["refinement"] = detail
        if refine:
            state.mode = Decision.REFINE

        # 2. sufficiency on the part no search has covered yet
        segment = state.uncovered
        suff = sig.sufficiency(segment, self.index, s.min_content_tokens if not refine else 2)
        info["sufficiency"] = {"content": suff.content, "entities": suff.entities[:6]}
        boundary = sig.clause_boundary(chunk.text)
        info["boundary"] = boundary
        # 1. stability — the primary trigger; a closed clause is the secondary one
        ready = state.stable_run >= s.stab_run or boundary
        can_launch = state.provisional_count < s.max_provisional

        if suff.ok and ready and can_launch:
            carry = []
            if state.covered_words:
                prior = " ".join(state.words[: state.covered_words])
                carry = sig.proper_entities(prior)[:3]
            query = sig.retrieval_shape(segment, carry)
            info["segment"] = segment
            state.trigger_segment_vec = await self._embed(segment)
            state.covered_words = len(state.words)
            state.trigger_vec = vec
            state.provisional_count += 1
            reason = "intent_stable" if state.stable_run >= s.stab_run else "clause_complete"
            if state.provisional_count > 1:
                reason = "multi_intent_detected"
            confidence = min(0.95, 0.5 + 0.08 * suff.content + 0.1 * len(suff.entities))
            decision = Decision.REFINE if refine else Decision.RETRIEVE
            if refine:
                reason = "late_constraint"
            return verdict(decision, reason, confidence, query=query)

        if refine:
            return verdict(Decision.REFINE, "late_constraint", 0.7)
        if not suff.ok:
            reason = "insufficient_content" if state.provisional_count == 0 else "awaiting_next_intent"
            return verdict(Decision.WAIT, reason, 0.3 + 0.05 * suff.content)
        if not can_launch:
            return verdict(Decision.WAIT, "provisional_budget_spent", 0.5)
        return verdict(Decision.WAIT, "intent_unstable", max(0.0, cos_prev))

    async def finalize(self, state: UtteranceState) -> ControllerVerdict:
        """The decision at utterance end, on the full text."""
        session = state.session
        text = state.prefix
        vec = state.vecs[-1] if state.vecs else (await self._embed(text) if text else None)
        if session.has_answer:
            presentation, detail = sig.is_presentation_only(text, session.previous_answer_text, self.index)
            if presentation:
                return ControllerVerdict(Decision.SUPPRESS, "presentation_restructure", 0.9, signals=detail)
            refine, rdetail = sig.refinement(text, vec, session.previous_vec, self.s.tau_refine)
            if refine:
                return ControllerVerdict(Decision.REFINE, "late_constraint", 0.8, signals=rdetail)
        suff = sig.sufficiency(text, self.index, 1)
        # Nothing the corpus could answer: no content words, or none the corpus
        # recognises ("okay thanks, that is great").
        if suff.content == 0 or not suff.entities:
            return ControllerVerdict(
                Decision.SUPPRESS, "no_information_need", 0.8, signals={"content": suff.content}
            )
        return ControllerVerdict(Decision.RETRIEVE, "utterance_complete", 0.9)
