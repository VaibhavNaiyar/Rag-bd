"""Claim-level delta refinement (gate G5).

A late detail does not restart the turn:

1. Affected claims are found by similarity between the new detail and each
   claim (plus the sub-query it answers).
2. Targeted delta sub-queries are issued for the affected sub-intents only;
   ``full_corpus_search`` stays False.
3. The model rewrites the answer as KEEP / EDIT / ADD / DROP instructions, so
   unaffected claims survive with their original ids and citations.
4. The result is version n+1 with parent n and explicit preserved / mutated lists.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import numpy as np

from slr.config import Settings
from slr.contracts import AnswerVersion, Claim, SubQuery
from slr.controller.signals import retrieval_shape
from slr.llm import ChatModel, parse_json, render_prompt
from slr.retrieval.context import EvidencePackage
from slr.retrieval.embed import Embedder
from slr.telemetry.cost import UsageLedger
from slr.text import REQUEST_WORDS, content_tokens, sentences

log = logging.getLogger(__name__)

OP_LINE = re.compile(r"^\s*(KEEP|EDIT|ADD|DROP|UNCERTAIN)\b\s*(c\d+)?\s*:?\s*(.*)$", re.I)


@dataclass
class RefinePlan:
    affected: list[str]  # claim ids
    affected_sub_queries: list[str]
    delta: list[SubQuery]
    method: str
    similarities: dict[str, float] = field(default_factory=dict)


def affected_claims(
    detail: str, previous: AnswerVersion, sub_queries: list[SubQuery], embedder: Embedder, margin: float
) -> tuple[list[str], dict[str, float]]:
    claims = list(previous.claims)
    if not claims:
        return [], {}
    sq_text = {sq.id: sq.text for sq in sub_queries}
    texts = [f"{c.text} {sq_text.get(c.sub_query_id, '')}" for c in claims]
    vecs = embedder.embed([detail, *texts], kind="text")
    sims = vecs[1:] @ vecs[0]
    top = float(np.max(sims))
    chosen = [c.id for c, s in zip(claims, sims) if s >= top - margin]
    return chosen, {c.id: round(float(s), 3) for c, s in zip(claims, sims)}


async def plan_refinement(
    detail: str,
    *,
    turn_id: str,
    previous: AnswerVersion,
    previous_utterance: str,
    sub_queries: list[SubQuery],
    model: ChatModel | None,
    embedder: Embedder,
    ledger: UsageLedger,
    s: Settings,
) -> RefinePlan:
    affected, sims = affected_claims(detail, previous, sub_queries, embedder, s.affect_margin)
    by_id = {c.id: c for c in previous.claims}
    affected_sq = []
    for cid in affected:
        sq = by_id[cid].sub_query_id
        if sq not in affected_sq:
            affected_sq.append(sq)
    if not affected_sq and sub_queries:
        affected_sq = [sub_queries[0].id]

    sq_text = {sq.id: sq.text for sq in sub_queries}
    items: list[dict] = []
    method = "heuristic"
    if model is not None:
        try:
            system, user = render_prompt(
                s.prompts_dir,
                "refine_plan",
                previous_utterance=previous_utterance,
                sub_queries="\n".join(f"{i}: {t}" for i, t in sq_text.items()),
                answer=previous.body,
                utterance=detail,
            )
            raw = await model.complete(system, user, ledger=ledger, step="refine_plan", json_mode=True, max_tokens=200)
            for d in parse_json(raw).get("delta_queries", [])[:2]:
                if isinstance(d, dict) and str(d.get("text", "")).strip():
                    target = str(d.get("for", "")) if str(d.get("for", "")) in sq_text else affected_sq[0]
                    items.append({"text": str(d["text"]).strip()[:200], "for": target})
            method = "llm"
        except Exception as exc:
            log.warning("refine planning failed (%s); using heuristic", exc)
            method = "heuristic_fallback"
    if not items:
        detail_terms = retrieval_shape(detail)
        for sq_id in affected_sq[:2]:
            topic = " ".join(t for t in sq_text.get(sq_id, "").split() if t.lower() not in detail_terms.lower())
            items.append({"text": f"{detail_terms} {topic}".strip(), "for": sq_id})

    delta = [
        SubQuery(id=f"{turn_id}_sq{n}", text=i["text"], source="refine", span=detail, confidence=0.8, reused_from=i["for"])
        for n, i in enumerate(items, start=1)
    ]
    return RefinePlan(affected, affected_sq, delta, method, sims)


def format_claims(previous: AnswerVersion) -> str:
    return "\n".join(f"c{i}: {c.text}" for i, c in enumerate(previous.claims, start=1))


async def llm_refine(
    model: ChatModel,
    detail: str,
    previous_utterance: str,
    previous: AnswerVersion,
    evidence: EvidencePackage,
    ledger: UsageLedger,
    s: Settings,
) -> AsyncIterator[str]:
    system, user = render_prompt(
        s.prompts_dir,
        "refine",
        previous_utterance=previous_utterance,
        utterance=detail,
        claims=format_claims(previous),
        evidence=evidence.text,
    )
    async for delta in model.stream(system, user, ledger=ledger, step="refine", max_tokens=700):
        yield delta


async def extractive_refine(
    detail: str, previous: AnswerVersion, plan: RefinePlan, evidence: EvidencePackage
) -> AsyncIterator[str]:
    """Offline: keep every claim; add the delta evidence that speaks to the detail."""
    terms = {t for t in content_tokens(detail) if t not in REQUEST_WORDS}
    cited = {cid for c in previous.claims for cid in c.chunk_ids}
    delta_ids = {sq.id for sq in plan.delta}
    candidates = []
    for hit in evidence.hits:
        if hit.chunk.chunk_id in cited or not (set(hit.sub_query_ids) & delta_ids):
            continue
        for sent in sentences(hit.chunk.text):
            overlap = len(terms & set(content_tokens(sent))) / max(1, len(terms))
            if overlap >= 0.2 and len(sent.split()) >= 5:
                candidates.append((overlap + 0.2 * hit.score, sent, hit))
    candidates.sort(key=lambda x: -x[0])
    adds = candidates[:2]
    last_affected = max(
        (i for i, c in enumerate(previous.claims, start=1) if c.id in plan.affected), default=len(previous.claims)
    )
    if not previous.claims:
        # Nothing was verified last time: the detail's evidence is the whole answer.
        for _, sent, hit in adds:
            yield f"ADD: {sent.rstrip().rstrip('.!?')} {hit.chunk.citation}.\n"
    for i, _claim in enumerate(previous.claims, start=1):
        yield f"KEEP c{i}\n"
        if i == last_affected:
            for _, sent, hit in adds:
                body = sent.rstrip().rstrip(".!?")
                yield f"ADD: {body} {hit.chunk.citation}.\n"
    if not adds:
        yield f"UNCERTAIN: The retrieved documents say nothing specific about: {detail.strip()}\n"


@dataclass
class RefineOp:
    op: str
    ref: int | None
    text: str


def parse_op(line: str) -> RefineOp | None:
    m = OP_LINE.match(line)
    if not m:
        return None
    ref = int(m.group(2)[1:]) if m.group(2) else None
    return RefineOp(m.group(1).upper(), ref, m.group(3).strip())


def claim_lookup(previous: AnswerVersion) -> dict[int, Claim]:
    return {i: c for i, c in enumerate(previous.claims, start=1)}
