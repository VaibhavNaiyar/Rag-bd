"""The turn loop: controller -> decomposer -> retrieval & fusion -> grounded synthesis.

``Engine`` holds the process-wide pieces (index, models, trace sink).
``SessionRunner`` is one conversation: it owns a scope-bound ``SessionStore``,
consumes client events in order and emits wire events through ``emit``.

Plain asyncio. CPU-bound work (embedding, rerank, verification) runs in worker
threads so transcript chunks keep flowing while a provisional search runs.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from slr.config import Settings, get_settings
from slr.contracts import (
    AnswerVersion,
    Claim,
    Decision,
    Hit,
    SubQuery,
    SubQueryResult,
    TranscriptChunk,
    validate_event,
)
from slr.controller.base import (
    Controller,
    ControllerVerdict,
    SessionView,
    UtteranceState,
)
from slr.controller.model import ModelController
from slr.controller.rules import RuleController
from slr.decompose.decomposer import decompose, heuristic_split
from slr.llm import ChatModel, build_chat_model
from slr.retrieval.context import EvidencePackage, assemble
from slr.retrieval.embed import Embedder
from slr.retrieval.fusion import FusionOutcome, fuse, union_prior
from slr.retrieval.pipeline import Retriever
from slr.retrieval.rerank import load_reranker
from slr.retrieval.store import Index, load_index
from slr.session.store import SessionStore, Topic
from slr.stream.simulator import load_fixture, timed_chunks
from slr.synthesis import generate as gen
from slr.synthesis import refine as ref
from slr.synthesis.grounding import (
    Grounder,
    LexicalVerifier,
    SentenceStream,
    Verifier,
    load_nli,
)
from slr.telemetry.cost import UsageLedger
from slr.telemetry.trace import TraceSink, new_record
from slr.text import content_tokens

log = logging.getLogger(__name__)

Emit = Callable[[dict[str, Any]], Awaitable[None]]


# ==========================================================================
# process-wide engine
# ==========================================================================


class Engine:
    def __init__(
        self,
        settings: Settings,
        index: Index,
        retriever: Retriever,
        verifier: Verifier,
        model: ChatModel | None,
        sink: TraceSink,
    ):
        self.s = settings
        self.index = index
        self.embedder: Embedder = index.embedder
        self.retriever = retriever
        self.verifier = verifier
        self.model = model
        self.sink = sink

    @classmethod
    def from_settings(cls, settings: Settings | None = None, index: Index | None = None) -> "Engine":
        s = settings or get_settings()
        index = index or load_index(s.index_dir)
        reranker = None
        if s.reranker != "none":
            try:
                reranker = load_reranker(s.rerank_model, s.rerank_max_length, s.rerank_temperature)
            except Exception as exc:
                if s.reranker == "cross":
                    raise
                log.warning("reranker unavailable (%s); fused order will be used", exc)
        verifier: Verifier = LexicalVerifier()
        if s.verifier != "lexical":
            try:
                verifier = load_nli(s.nli_model)
            except Exception as exc:
                if s.verifier == "nli":
                    raise
                log.warning("NLI verifier unavailable (%s); lexical support will be used", exc)
        model = build_chat_model(s)
        sink = TraceSink(s.trace_path)
        return cls(s, index, Retriever(index, reranker, s), verifier, model, sink)

    def with_settings(self, settings: Settings) -> "Engine":
        """Same loaded models, different knobs — used by the ablation harness."""
        clone = copy.copy(self)
        clone.s = settings
        clone.retriever = Retriever(self.index, self.retriever.reranker if settings.reranker != "none" else None, settings)
        clone.model = self.model if settings.llm != "offline" else None
        return clone

    def controller(self, ledger: UsageLedger) -> Controller:
        if self.s.controller == "model":
            if self.model is None:
                raise RuntimeError("SLR_CONTROLLER=model needs an LLM (set OPENAI_API_KEY)")
            ctl = ModelController(self.model, self.index, self.embedder, self.s)
            ctl.ledger = ledger
            return ctl
        return RuleController(self.index, self.embedder, self.s)

    def corpus_info(self) -> dict[str, Any]:
        return {
            "docs": self.index.doc_count,
            "chunks": len(self.index.chunks),
            "indexedAt": self.index.manifest.get("built_at"),
        }

    def models_info(self) -> dict[str, str]:
        return {
            "embedder": self.index.manifest.get("embed_model", "?"),
            "reranker": getattr(self.retriever.reranker, "name", "none"),
            "verifier": self.verifier.name,
            "llm": getattr(self.model, "name", "offline"),
            "controller": self.s.controller,
            "branches": ",".join(self.s.branches),
            "quota_per_intent": str(self.s.quota_per_intent),
        }

    def ledger(self) -> UsageLedger:
        return UsageLedger(self.s.price_in_per_m, self.s.price_out_per_m, self.s.cpu_usd_per_hour)


# ==========================================================================
# one turn
# ==========================================================================


@dataclass
class Provisional:
    sub_query: SubQuery
    task: asyncio.Task
    started_ms: int
    trigger: str
    cancelled: bool = False


@dataclass
class ActiveTurn:
    id: str
    t0: float
    state: UtteranceState
    record: dict[str, Any]
    ledger: UsageLedger
    controller: Controller
    provisional: list[Provisional] = field(default_factory=list)
    first_retrieval_ms: int | None = None
    end_ms: int | None = None
    first_token_ms: int | None = None
    last_decision: tuple[str, str] | None = None

    def ms(self) -> int:
        return int(round((time.perf_counter() - self.t0) * 1000))

    @property
    def live_provisional(self) -> Provisional | None:
        for p in reversed(self.provisional):
            if not p.cancelled:
                return p
        return None


class SessionRunner:
    def __init__(self, engine: Engine, emit: Emit):
        self.engine = engine
        self._emit = emit
        self.store = SessionStore()
        self.turn: ActiveTurn | None = None
        self.completed: list[dict[str, Any]] = []
        self._turn_done = asyncio.Event()

    # ------------------------------------------------------------------ io

    async def emit(self, event: dict[str, Any]) -> None:
        validate_event(event)
        if self.turn is not None and event["type"] not in ("answer.token", "transcript.chunk"):
            self.turn.record.setdefault("events", []).append(
                {k: v for k, v in event.items() if k not in ("kept", "hits", "claims")}
            )
        await self._emit(event)

    async def start(self) -> None:
        await self.emit(
            {"type": "session.ready", "sessionId": self.store.id, "corpus": self.engine.corpus_info()}
        )

    async def handle(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        try:
            if kind == "utterance.start":
                await self.utterance_start()
            elif kind == "utterance.chunk":
                await self.utterance_chunk(str(message.get("text", "")))
            elif kind == "utterance.end":
                await self.utterance_end()
            elif kind == "replay":
                await self.replay(str(message.get("fixture", "")), float(message.get("speed") or 1.0))
            elif kind == "session.new":
                await self.new_session()
            else:
                await self.emit({"type": "error", "code": "bad_request", "message": f"unknown event {kind!r}"})
        except Exception as exc:  # a failed turn must not kill the socket
            log.exception("turn failed")
            turn_id = self.turn.id if self.turn else None
            if self.turn is not None:
                self.turn.record["errors"].append(f"{type(exc).__name__}: {exc}")
                self._finish_trace(self.turn)
                self.turn = None
            event = {"type": "error", "code": "turn_failed", "message": str(exc)[:300]}
            if turn_id:
                event["turnId"] = turn_id
            await self.emit(event)

    async def close(self) -> None:
        if self.turn is not None:
            for p in self.turn.provisional:
                p.task.cancel()
        self.store.clear()

    async def new_session(self) -> None:
        await self.close()
        self.store = SessionStore()
        self.turn = None
        await self.start()

    # --------------------------------------------------------------- turn

    def _session_view(self) -> SessionView:
        topic = self.store.topic
        if topic is None:
            return SessionView()
        return SessionView(
            has_answer=True,
            previous_utterance=topic.utterance,
            previous_vec=topic.utterance_vec,
            previous_answer_text=topic.answer.body,
        )

    async def utterance_start(self) -> None:
        if self.turn is not None and self.turn.end_ms is None:
            await self.utterance_end()  # an unterminated utterance is closed, not dropped
        e = self.engine
        ledger = e.ledger()
        turn_id = self.store.next_turn_id()
        controller = e.controller(ledger)
        self.turn = ActiveTurn(
            id=turn_id,
            t0=time.perf_counter(),
            state=UtteranceState(session=self._session_view()),
            record=new_record(self.store.id, turn_id, controller.name, e.models_info()),
            ledger=ledger,
            controller=controller,
        )
        self._turn_done.clear()
        await self.emit({"type": "turn.start", "turnId": turn_id})

    async def utterance_chunk(self, text: str) -> None:
        text = " ".join(text.split())
        if not text:
            return
        if self.turn is None or self.turn.end_ms is not None:
            await self.utterance_start()
        turn = self.turn
        at = turn.ms()
        turn.record["chunks"].append({"text": text, "at_ms": at})
        await self.emit({"type": "transcript.chunk", "turnId": turn.id, "text": text, "atMs": at})

        started = time.perf_counter()
        turn.state.active_provisional = turn.live_provisional.sub_query.id if turn.live_provisional else None
        verdict = await turn.controller.observe(TranscriptChunk(text, at), turn.state)
        turn.ledger.record_compute("controller", (time.perf_counter() - started) * 1000, turn.controller.name)
        await self._apply_verdict(turn, verdict)

    async def _decision(self, turn: ActiveTurn, verdict: ControllerVerdict, at: int | None = None) -> None:
        at = turn.ms() if at is None else at
        turn.record["decisions"].append(
            {
                "decision": verdict.decision.value,
                "reason": verdict.reason,
                "confidence": verdict.confidence,
                "at_ms": at,
                "query": verdict.query,
                "signals": verdict.signals,
            }
        )
        await self.emit(
            {
                "type": "controller.decision",
                "turnId": turn.id,
                "decision": verdict.decision.value,
                "reason": verdict.reason,
                "atMs": at,
                "confidence": verdict.confidence,
            }
        )
        turn.last_decision = (verdict.decision.value, verdict.reason)

    async def _apply_verdict(self, turn: ActiveTurn, verdict: ControllerVerdict) -> None:
        await self._decision(turn, verdict)
        if verdict.cancel and turn.live_provisional is not None:
            await self._cancel(turn, turn.live_provisional, verdict.cancel_reason or "cancelled")
        if verdict.query:
            trigger = "refine" if verdict.decision == Decision.REFINE else "provisional"
            # One clause can already carry two needs ("the cancellation policy and
            # the catering options"). Splitting here — no model call — means both
            # start retrieving before the speaker finishes, instead of one blended
            # query that serves neither.
            segment = verdict.signals.get("segment") or verdict.query
            parts = [
                p["text"]
                for p in heuristic_split(segment, self.engine.index, self.engine.embedder, self.engine.s.carry_cos)
            ] or [verdict.query]
            for query in parts[:2]:
                await self._launch_provisional(turn, query, trigger)

    async def _launch_provisional(self, turn: ActiveTurn, query: str, trigger: str) -> None:
        n = len(turn.provisional) + 1
        sq = SubQuery(id=f"{turn.id}_p{n}", text=query, source="provisional", span=turn.state.prefix)
        at = turn.ms()
        task = asyncio.create_task(asyncio.to_thread(self.engine.retriever.search_provisional, sq))
        turn.provisional.append(Provisional(sq, task, at, trigger))
        self._mark_retrieval(turn, sq, trigger, at)
        await self.emit(
            {
                "type": "retrieval.started",
                "turnId": turn.id,
                "subQueryId": sq.id,
                "trigger": trigger,
                "atMs": at,
                "query": sq.text,
            }
        )

    def _mark_retrieval(self, turn: ActiveTurn, sq: SubQuery, trigger: str, at: int) -> None:
        if turn.first_retrieval_ms is None:
            turn.first_retrieval_ms = at
        turn.record["retrieval_events"].append(
            {
                "sub_query_id": sq.id,
                "query": sq.text,
                "trigger": trigger,
                "at_ms": at,
                "before_utterance_end": turn.end_ms is None,
            }
        )

    async def _cancel(self, turn: ActiveTurn, p: Provisional, reason: str) -> None:
        p.cancelled = True
        p.task.cancel()
        turn.record["retrieval_events"].append(
            {"sub_query_id": p.sub_query.id, "event": "retrieval_cancelled", "reason": reason, "at_ms": turn.ms()}
        )
        await self.emit(
            {"type": "retrieval.cancelled", "turnId": turn.id, "subQueryId": p.sub_query.id, "reason": reason}
        )

    async def _provisional_results(self, turn: ActiveTurn) -> list[SubQueryResult]:
        out = []
        for p in turn.provisional:
            if p.cancelled:
                continue
            try:
                out.append(await asyncio.wait_for(asyncio.shield(p.task), timeout=30))
            except (asyncio.CancelledError, asyncio.TimeoutError) as exc:
                turn.record["errors"].append(f"provisional {p.sub_query.id}: {type(exc).__name__}")
        return out

    async def utterance_end(self) -> None:
        turn = self.turn
        if turn is None or turn.end_ms is not None:
            return
        turn.end_ms = turn.ms()
        turn.record["utterance"] = turn.state.prefix
        turn.record["utterance_end_ms"] = turn.end_ms
        await self.emit({"type": "utterance.end", "turnId": turn.id, "atMs": turn.end_ms})

        final = await turn.controller.finalize(turn.state)
        await self._decision(turn, final, turn.end_ms)
        turn.record["mode"] = final.decision.value

        if final.decision == Decision.SUPPRESS:
            for p in turn.provisional:
                if not p.cancelled:
                    await self._cancel(turn, p, "presentation_only")
            await self._suppressed_turn(turn, final.reason)
        elif final.decision == Decision.REFINE and self.store.has_answer:
            await self._refine_turn(turn)
        else:
            turn.record["mode"] = Decision.RETRIEVE.value
            await self._retrieve_turn(turn)

        self._finish_trace(turn)
        self._turn_done.set()

    # ------------------------------------------------------ retrieve path

    async def _retrieve_turn(self, turn: ActiveTurn) -> None:
        e, s = self.engine, self.engine.s
        topic = self.store.topic
        context = topic.utterance if topic else ""
        decomposition = await decompose(
            turn.state.prefix,
            turn_id=turn.id,
            model=e.model,
            embedder=e.embedder,
            index=e.index,
            settings=s,
            ledger=turn.ledger,
            context=context,
        )
        turn.record["decomposition"] = {
            "method": decomposition.method,
            "raw": decomposition.raw,
            "merged": decomposition.merged,
            "capped": decomposition.capped,
            "ms": round(decomposition.ms, 1),
        }
        decomposed = decomposition.items

        # Reuse is decided on the *queries*, which are known now, so the fresh
        # searches can start while the provisional ones are still running. That
        # overlap is what turns early retrieval into a latency win.
        live = [p for p in turn.provisional if not p.cancelled]
        reuse_of: dict[str, int] = {}
        if live and decomposed:
            pv = e.embedder.embed([p.sub_query.text for p in live], kind="query")
            dv = e.embedder.embed([sq.text for sq in decomposed], kind="query")
            sims = dv @ pv.T
            taken: set[int] = set()
            for i, sq in enumerate(decomposed):
                j = int(sims[i].argmax())
                if float(sims[i, j]) >= s.reuse_cos and j not in taken:
                    taken.add(j)
                    reuse_of[sq.id] = j
                    sq.reused_from = live[j].sub_query.id

        fresh = [sq for sq in decomposed if sq.id not in reuse_of]
        for sq in fresh:
            at = turn.ms()
            self._mark_retrieval(turn, sq, "multi_intent", at)
            await self.emit(
                {
                    "type": "retrieval.started",
                    "turnId": turn.id,
                    "subQueryId": sq.id,
                    "trigger": "multi_intent",
                    "atMs": at,
                    "query": sq.text,
                }
            )
        items = [{"id": p.sub_query.id, "text": p.sub_query.text, "source": "provisional"} for p in live]
        items += [{"id": sq.id, "text": sq.text, "source": "decomposed"} for sq in decomposed]
        await self.emit({"type": "subqueries", "turnId": turn.id, "items": items})

        started = time.perf_counter()
        fresh_task = asyncio.create_task(asyncio.to_thread(e.retriever.search_many, fresh)) if fresh else None
        provisional = await self._provisional_results(turn)
        fresh_results = await fresh_task if fresh_task is not None else []
        turn.ledger.record_compute("retrieve_rerank", (time.perf_counter() - started) * 1000, str(len(fresh)))

        by_provisional = {r.sub_query.id: r for r in provisional}
        by_id = {r.sub_query.id: r for r in fresh_results}
        late: list[SubQuery] = []
        for sq in decomposed:
            if sq.id not in reuse_of:
                continue
            src = by_provisional.get(live[reuse_of[sq.id]].sub_query.id)
            if src is None:  # the provisional search failed; search for real
                late.append(sq)
                continue
            by_id[sq.id] = SubQueryResult(
                sub_query=sq,
                candidates=src.candidates,
                kept=[Hit(h.chunk, h.score, list(h.branches), [sq.id], h.rrf) for h in src.kept],
                branch_counts=src.branch_counts,
                reranked=src.reranked,
            )
        if late:
            for r in await asyncio.to_thread(e.retriever.search_many, late):
                by_id[r.sub_query.id] = r
        results = [by_id[sq.id] for sq in decomposed]
        reused = {sq.id for sq in decomposed if sq.id in reuse_of and sq.id not in {x.id for x in late}}
        extras = [r for r in provisional if r.sub_query.id not in {live[j].sub_query.id for j in reuse_of.values()}]

        for r in [*provisional, *results]:
            await self.emit(
                {
                    "type": "retrieval.result",
                    "turnId": turn.id,
                    "subQueryId": r.sub_query.id,
                    "candidates": r.candidates,
                    "kept": [h.to_wire() for h in r.kept],
                    "reused": r.sub_query.id in reused,
                }
            )
        turn.record["sub_queries"] = [
            {
                "id": sq.id,
                "text": sq.text,
                "source": sq.source,
                "span": sq.span,
                "confidence": sq.confidence,
                "reused_from": sq.reused_from,
            }
            for sq in [*(r.sub_query for r in provisional), *decomposed]
        ]
        turn.record["retrieval"] = [_result_record(r, r.sub_query.id in reused) for r in [*provisional, *results]]

        fusion = fuse(results, s.top_k, s.quota_per_intent, extras)
        await self._emit_fusion(turn, fusion.hits, fusion.quota_applied, True, fusion)

        all_sq = [*decomposed, *(r.sub_query for r in extras)]
        evidence = assemble(fusion.hits, all_sq, s.context_char_budget)
        version, parent = 1, None
        grounder = self._grounder(turn, evidence, version, decomposed[0].id if decomposed else "")
        if e.model is not None:
            stream = gen.llm_answer(e.model, turn.state.prefix, decomposed, evidence, turn.ledger, s)
        else:
            stream = gen.extractive_answer(turn.state.prefix, decomposed, evidence, idf=e.index.idf)
        body = await self._stream_grounded(turn, stream, grounder, version)
        answer = self._version(grounder, body, version, parent, (), (), True)
        await self._emit_version(turn, answer, grounder)
        self.store.commit(
            Topic(
                utterance=turn.state.prefix,
                utterance_vec=turn.state.vecs[-1] if turn.state.vecs else None,
                sub_queries=tuple(decomposed),
                hits=tuple(evidence.hits),
                answer=answer,
            )
        )
        await self._complete(turn)

    # --------------------------------------------------------- refine path

    async def _refine_turn(self, turn: ActiveTurn) -> None:
        e, s = self.engine, self.engine.s
        topic = self.store.topic
        assert topic is not None
        previous = topic.answer
        detail = turn.state.prefix
        plan = await ref.plan_refinement(
            detail,
            turn_id=turn.id,
            previous=previous,
            previous_utterance=topic.utterance,
            sub_queries=list(topic.sub_queries),
            model=e.model,
            embedder=e.embedder,
            ledger=turn.ledger,
            s=s,
        )
        turn.record["decomposition"] = {
            "method": f"refine_{plan.method}",
            "affected_claims": plan.affected,
            "affected_sub_queries": plan.affected_sub_queries,
            "claim_similarity": plan.similarities,
        }
        provisional = await self._provisional_results(turn)
        for sq in plan.delta:
            at = turn.ms()
            self._mark_retrieval(turn, sq, "refine", at)
            await self.emit(
                {
                    "type": "retrieval.started",
                    "turnId": turn.id,
                    "subQueryId": sq.id,
                    "trigger": "refine",
                    "atMs": at,
                    "query": sq.text,
                }
            )
        items = [{"id": r.sub_query.id, "text": r.sub_query.text, "source": "provisional"} for r in provisional]
        items += [{"id": sq.id, "text": sq.text, "source": "decomposed"} for sq in plan.delta]
        await self.emit({"type": "subqueries", "turnId": turn.id, "items": items})

        started = time.perf_counter()
        delta_results = await asyncio.to_thread(e.retriever.search_many, plan.delta)
        turn.ledger.record_compute("retrieve_rerank", (time.perf_counter() - started) * 1000, "delta")
        for r in [*provisional, *delta_results]:
            await self.emit(
                {
                    "type": "retrieval.result",
                    "turnId": turn.id,
                    "subQueryId": r.sub_query.id,
                    "candidates": r.candidates,
                    "kept": [h.to_wire() for h in r.kept],
                }
            )
        turn.record["sub_queries"] = [
            {"id": sq.id, "text": sq.text, "source": sq.source, "refines": sq.reused_from, "span": sq.span}
            for sq in [*(r.sub_query for r in provisional), *plan.delta]
        ]
        turn.record["retrieval"] = [_result_record(r, False) for r in [*provisional, *delta_results]]

        # Delta evidence competes among itself (quota per delta query), then is
        # appended to the prior evidence, which stays in place.
        delta_fusion = fuse(delta_results, max(4, s.quota_per_intent * len(plan.delta)), s.quota_per_intent, provisional)
        final_hits = union_prior(list(topic.hits), delta_fusion.hits)
        await self._emit_fusion(turn, final_hits, delta_fusion.quota_applied, False, delta_fusion, len(topic.hits))

        # delta hits answer the sub-query they refine
        refines = {sq.id: sq.reused_from for sq in plan.delta}
        all_sq = [*topic.sub_queries, *plan.delta]
        evidence = assemble(final_hits, all_sq, s.context_char_budget + 3000)
        version, parent = self.store.next_version()
        grounder = self._grounder(turn, evidence, version, plan.affected_sub_queries[0] if plan.affected_sub_queries else "")

        if e.model is not None:
            stream = ref.llm_refine(e.model, detail, topic.utterance, previous, evidence, turn.ledger, s)
        else:
            stream = ref.extractive_refine(detail, previous, plan, evidence)
        body, preserved, mutated, dropped = await self._apply_refine_ops(turn, stream, grounder, previous, version, refines)
        answer = self._version(grounder, body, version, parent, tuple(preserved), tuple(mutated), False)
        turn.record["refinement"] = {"preserved": preserved, "mutated": mutated, "dropped": dropped}
        await self._emit_version(turn, answer, grounder)
        self.store.commit(
            Topic(
                utterance=f"{topic.utterance} {detail}",
                utterance_vec=turn.state.vecs[-1] if turn.state.vecs else topic.utterance_vec,
                sub_queries=tuple(all_sq),
                hits=tuple(evidence.hits),
                answer=answer,
            )
        )
        await self._complete(turn)

    async def _apply_refine_ops(
        self,
        turn: ActiveTurn,
        stream: AsyncIterator[str],
        grounder: Grounder,
        previous: AnswerVersion,
        version: int,
        refines: dict[str, str | None],
    ) -> tuple[str, list[str], list[str], list[str]]:
        lookup = ref.claim_lookup(previous)
        seen: set[int] = set()
        preserved: list[str] = []
        mutated: list[str] = []
        dropped: list[str] = []
        parts: list[str] = []
        last_sq: str | None = None

        async def ship(claim: Claim) -> None:
            nonlocal last_sq
            sq = refines.get(claim.sub_query_id) or claim.sub_query_id
            sep = "" if not parts else ("\n\n" if sq != last_sq else " ")
            last_sq = sq
            parts.append(sep + claim.text)
            await self._tokens(turn, sep + claim.text, version)

        async def keep(i: int) -> None:
            claim = lookup[i]
            grounder.claims.append(claim)
            grounder.generated += 1
            grounder.supported += int(claim.support >= grounder.support_min)
            preserved.append(claim.id)
            await ship(claim)

        async def handle(line: str) -> None:
            op = ref.parse_op(line)
            if op is None:
                return
            if op.op in ("KEEP", "EDIT", "DROP") and (op.ref not in lookup or op.ref in seen):
                return
            if op.ref is not None:
                seen.add(op.ref)
            if op.op == "KEEP":
                await keep(op.ref)
            elif op.op == "DROP":
                dropped.append(lookup[op.ref].id)
            elif op.op == "UNCERTAIN":
                note, _ = grounder.resolve(op.text)
                if note:
                    grounder.uncertainty.append(note)
            elif op.op in ("EDIT", "ADD"):
                result = await asyncio.to_thread(grounder.ground, op.text)
                if result is None:
                    if op.op == "EDIT":
                        await keep(op.ref)  # an unverifiable edit never replaces a verified claim
                    return
                text, hits, support = result
                sq = refines.get(hits[0].sub_query_ids[0]) if hits and hits[0].sub_query_ids else None
                claim = Claim(
                    id=f"{turn.id}_v{version}_c{len(grounder.claims) + 1}",
                    text=text,
                    chunk_ids=tuple(h.chunk.chunk_id for h in hits),
                    sub_query_id=(lookup[op.ref].sub_query_id if op.op == "EDIT" else (sq or _first_sq(hits))),
                    support=support,
                )
                grounder.claims.append(claim)
                if op.op == "EDIT":
                    mutated.append(claim.id)
                await ship(claim)

        buf = ""
        async for delta in stream:
            buf += delta
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                await handle(line)
        if buf.strip():
            await handle(buf)
        # Claims the model did not mention are kept: refinement never silently loses a fact.
        for i in lookup:
            if i not in seen:
                await keep(i)
        return "".join(parts), preserved, mutated, dropped

    # ------------------------------------------------------- suppress path

    async def _suppressed_turn(self, turn: ActiveTurn, reason: str) -> None:
        e, s = self.engine, self.engine.s
        topic = self.store.topic
        if topic is None or not topic.answer.claims:
            if reason == "no_information_need":
                note = "No information request was detected, so nothing was retrieved."
            elif topic is None:
                note = "There is no earlier answer in this session to reformat."
            else:
                note = "The earlier answer had no verified facts to reformat, so nothing was retrieved."
            answer = AnswerVersion(1, None, (), (), (), (note,), False, body="")
            turn.record["answer"] = _answer_record(answer)
            await self._emit_version(turn, answer, None)
            await self._complete(turn)
            return

        previous = topic.answer
        await self._emit_fusion(turn, list(topic.hits), False, False, None, len(topic.hits))
        evidence = EvidencePackage(hits=list(topic.hits), text="")
        version, parent = self.store.next_version()
        grounder = self._grounder(turn, evidence, version, previous.claims[0].sub_query_id)
        stream = gen.restructure(e.model, turn.state.prefix, previous, turn.ledger, s)
        body = await self._stream_grounded(turn, stream, grounder, version)
        # A restructured claim citing the same chunks as a prior claim is that claim, carried over.
        prior = {c.chunk_ids: c for c in previous.claims}
        preserved = []
        remapped = []
        for c in grounder.claims:
            match = prior.get(c.chunk_ids)
            if match is not None and match.id not in preserved:
                preserved.append(match.id)
                remapped.append(Claim(match.id, c.text, c.chunk_ids, match.sub_query_id, c.support))
            else:
                remapped.append(c)
        grounder.claims = remapped
        answer = self._version(grounder, body, version, parent, tuple(preserved), (), False)
        await self._emit_version(turn, answer, grounder)
        self.store.commit(
            Topic(
                utterance=topic.utterance,
                utterance_vec=topic.utterance_vec,
                sub_queries=topic.sub_queries,
                hits=topic.hits,
                answer=answer,
            )
        )
        await self._complete(turn)

    # ------------------------------------------------------------ shared

    def _grounder(self, turn: ActiveTurn, evidence: EvidencePackage, version: int, default_sq: str) -> Grounder:
        return Grounder(
            evidence=evidence,
            verifier=self.engine.verifier,
            support_min=self.engine.s.support_min,
            auto_cite_min=self.engine.s.auto_cite_min,
            claim_prefix=f"{turn.id}_v{version}",
            default_sub_query=default_sq,
        )

    async def _emit_fusion(
        self,
        turn: ActiveTurn,
        hits: list[Hit],
        quota_applied: bool,
        full_corpus: bool,
        fusion: FusionOutcome | None,
        carried: int = 0,
    ) -> None:
        turn.record["fusion"] = {
            "final_count": len(hits),
            "quota_applied": quota_applied,
            "quota_promoted": fusion.quota_promoted if fusion else [],
            "per_sub_query": fusion.per_sub_query if fusion else {},
            "full_corpus_search": full_corpus,
            "carried_from_session": carried,
            "chunk_ids": [h.chunk.chunk_id for h in hits],
        }
        await self.emit(
            {
                "type": "fusion.final",
                "turnId": turn.id,
                "hits": [h.to_wire() for h in hits],
                "quotaApplied": quota_applied,
                "fullCorpusSearch": full_corpus,
            }
        )

    async def _tokens(self, turn: ActiveTurn, text: str, version: int) -> None:
        if not text:
            return
        if turn.first_token_ms is None:
            turn.first_token_ms = turn.ms()
        # word-sized tokens; whitespace stays attached so the client can concatenate
        pieces = [p for p in _split_tokens(text) if p]
        for p in pieces:
            await self.emit({"type": "answer.token", "turnId": turn.id, "version": version, "text": p})

    async def _stream_grounded(
        self, turn: ActiveTurn, stream: AsyncIterator[str], grounder: Grounder, version: int
    ) -> str:
        splitter = SentenceStream()
        shipped: list[str] = []
        started = time.perf_counter()
        verify_ms = 0.0

        async def release(segments: list[str]) -> None:
            nonlocal verify_ms
            for seg in segments:
                t = time.perf_counter()
                out = await asyncio.to_thread(grounder.process, seg)
                verify_ms += (time.perf_counter() - t) * 1000
                if out.text:
                    text = out.text
                    if not shipped:
                        text = text.lstrip()
                    shipped.append(text)
                    await self._tokens(turn, text, version)

        async for delta in stream:
            await release(splitter.feed(delta))
        await release(splitter.flush())
        turn.ledger.record_compute("verify", verify_ms, self.engine.verifier.name)
        if self.engine.model is None:
            turn.ledger.record_compute("synthesise", (time.perf_counter() - started) * 1000 - verify_ms, "extractive")
        return "".join(shipped).rstrip()

    def _version(
        self,
        grounder: Grounder,
        body: str,
        version: int,
        parent: int | None,
        preserved: tuple[str, ...],
        mutated: tuple[str, ...],
        full_corpus: bool,
    ) -> AnswerVersion:
        uncertainty = list(dict.fromkeys(grounder.uncertainty))
        if not grounder.claims and not uncertainty:
            uncertainty.append("The retrieved documents do not contain enough evidence to answer this request.")
        return AnswerVersion(
            version=version,
            parent=parent,
            claims=tuple(grounder.claims),
            preserved=preserved,
            mutated=mutated,
            uncertainty=tuple(uncertainty),
            full_corpus_search=full_corpus,
            body=body,
            citation_support_rate=round(grounder.support_rate, 4),
            fabricated_citations=_unresolved_markers(body, grounder),
            fabricated_blocked=grounder.fabricated_blocked,
        )

    async def _emit_version(self, turn: ActiveTurn, answer: AnswerVersion, grounder: Grounder | None) -> None:
        rec = _answer_record(answer)
        if grounder is not None:
            rec["grounding"] = {
                "generated_claims": grounder.generated,
                "supported_claims": grounder.supported,
                "demoted_claims": grounder.demoted,
                "auto_cited": grounder.auto_cited,
                "fabricated_markers": grounder.fabricated_markers,
                "verifier": self.engine.verifier.name,
            }
        turn.record["answer"] = rec
        cmap = {c.chunk_id: c for c in self.engine.index.chunks}
        turn.record["citations"] = sorted(
            {cmap[cid].citation for claim in answer.claims for cid in claim.chunk_ids if cid in cmap}
        )
        turn.record["citation_support_rate"] = answer.citation_support_rate
        turn.record["fabricated_citations"] = answer.fabricated_citations
        turn.record["fabricated_citations_blocked"] = answer.fabricated_blocked
        turn.record["uncertainty"] = list(answer.uncertainty)
        await self.emit(
            {
                "type": "answer.version",
                "turnId": turn.id,
                "version": answer.version,
                "parent": answer.parent,
                "claims": [c.to_wire() for c in answer.claims],
                "preserved": list(answer.preserved),
                "mutated": list(answer.mutated),
                "uncertainty": list(answer.uncertainty),
                "citationSupportRate": answer.citation_support_rate,
                "fabricatedCitations": answer.fabricated_citations,
            }
        )

    async def _complete(self, turn: ActiveTurn) -> None:
        now = turn.ms()
        end = turn.end_ms or now
        first_token = (turn.first_token_ms if turn.first_token_ms is not None else now) - end
        first_retrieval = None if turn.first_retrieval_ms is None else turn.first_retrieval_ms - end
        entries = turn.ledger.drain()
        cost = UsageLedger.summarise(entries)
        latency = {"firstRetrieval": first_retrieval, "firstToken": max(0, first_token), "complete": now - end}
        turn.record["first_retrieval_ms"] = turn.first_retrieval_ms
        turn.record["before_utterance_end"] = (
            None if turn.first_retrieval_ms is None else turn.first_retrieval_ms < end
        )
        turn.record["latency_ms"] = {
            "first_retrieval_rel_end": first_retrieval,
            "first_token_after_end": latency["firstToken"],
            "complete_after_end": latency["complete"],
            "first_token_abs": turn.first_token_ms,
            "complete_abs": now,
        }
        turn.record["cost"] = {**cost, "entries": entries}
        await self.emit({"type": "turn.complete", "turnId": turn.id, "latencyMs": latency, "cost": cost})

    def _finish_trace(self, turn: ActiveTurn) -> None:
        if turn.record.get("_written"):
            return
        turn.record["_written"] = True
        record = {k: v for k, v in turn.record.items() if k != "_written"}
        self.engine.sink.write(record)
        self.completed.append(record)

    # ------------------------------------------------------------ replay

    async def replay(self, fixture: str, speed: float = 1.0) -> None:
        data = load_fixture(self.engine.s.fixtures_dir, fixture)
        await self.play_turns(data["turns"], speed)

    async def play_turns(self, turns: list[dict[str, Any]], speed: float = 1.0) -> None:
        speed = max(0.1, min(speed, 20.0))
        for spec in turns:
            await self.utterance_start()
            for text, delay_ms in timed_chunks(spec):
                if delay_ms > 0:
                    await asyncio.sleep(delay_ms / 1000 / speed)
                await self.utterance_chunk(text)
            tail = float(spec.get("end_pause_ms", 250))
            await asyncio.sleep(tail / 1000 / speed)
            await self.utterance_end()


def _split_tokens(text: str) -> list[str]:
    out, cur = [], ""
    for ch in text:
        cur += ch
        if ch in " \n":
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    # merge pure-whitespace pieces into the previous token
    merged: list[str] = []
    for p in out:
        if merged and not p.strip():
            merged[-1] += p
        else:
            merged.append(p)
    return merged


def _first_sq(hits: list[Hit]) -> str:
    for h in hits:
        if h.sub_query_ids:
            return h.sub_query_ids[0]
    return ""


def _unresolved_markers(body: str, grounder: Grounder) -> int:
    """Post-validation audit of the shipped body: markers that do not resolve."""
    from slr.synthesis.grounding import CANDIDATE_MARKER, normalise_marker

    cmap = grounder.evidence.citation_map
    bad = 0
    for m in CANDIDATE_MARKER.finditer(body):
        if m.group(2) is None and not m.group(1).lower().startswith("doc"):
            continue
        if normalise_marker(m.group(1), m.group(2)) not in cmap:
            bad += 1
    return bad


def _result_record(r: SubQueryResult, reused: bool) -> dict[str, Any]:
    return {
        "sub_query_id": r.sub_query.id,
        "candidates": r.candidates,
        "branch_counts": r.branch_counts,
        "kept": [{"chunk_id": h.chunk.chunk_id, "score": round(h.score, 4), "branches": h.branches} for h in r.kept],
        "reranked": r.reranked,
        "reused": reused,
        "ms": round(r.elapsed_ms, 1),
    }


def _answer_record(answer: AnswerVersion) -> dict[str, Any]:
    return {
        "version": answer.version,
        "parent": answer.parent,
        "body": answer.body,
        "claims": [c.to_wire() for c in answer.claims],
        "preserved": list(answer.preserved),
        "mutated": list(answer.mutated),
        "full_corpus_search": answer.full_corpus_search,
        "claim_count": len(answer.claims),
        "word_count": len(content_tokens(answer.body)),
    }
