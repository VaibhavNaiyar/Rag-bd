"""Multi-intent decomposition: one model call -> N sub-queries, capped and deduped.

Over-fragmentation guard (guide pitfall #5):

* at most ``max_subqueries``
* any pair with cosine >= ``merge_cos`` is merged (the higher-confidence one wins),
  unless the two differ in a number: '2014 winner' and '2018 winner' are two
  readings of one question, not a duplicate
* a single-intent utterance yields exactly one sub-query

Without a model (no API key) or when the model call fails, a deterministic
clause splitter runs instead. The trace records which path ran.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from slr.config import Settings
from slr.contracts import Hit, SubQuery
from slr.controller.signals import proper_entities, retrieval_shape
from slr.llm import ChatModel, parse_json, render_prompt
from slr.retrieval.embed import Embedder
from slr.retrieval.store import Index
from slr.telemetry.cost import UsageLedger
from slr.text import (
    REQUEST_WORDS,
    content_tokens,
    heading_subject,
    is_number,
    raw_words,
)

log = logging.getLogger(__name__)


@dataclass
class Decomposition:
    items: list[SubQuery]
    method: str
    merged: int = 0
    capped: int = 0
    raw: list[dict] = field(default_factory=list)
    ms: float = 0.0


# Split points: sentence ends, semicolons, and a conjunction that opens a new request.
_OPENERS = (
    r"(?:what|which|who|when|where|why|how|whether|is|are|does|do|can|could|should|will|"
    r"tell|give|show|list|explain|describe|find|compare|the|any|also|i|i'd|we|please)"
)
_SPLIT = re.compile(
    rf"(?:[?.!;]\s+|,?\s+(?:and|plus|also|as well as)\s+(?:also\s+)?(?={_OPENERS}\b)|,\s+(?=(?:and\s+)?{_OPENERS}\b))",
    re.I,
)
_INNER_AND = re.compile(r"\s+and\s+(?=the\b|its\b|their\b|any\b)", re.I)
_FILLER = re.compile(
    r"^(?:and|also|plus|so|then|well|okay|ok|please|as well as)?\s*"
    r"(?:(?:i\s+(?:also\s+)?(?:need|want|would like|'d like|wanted)\s+(?:to\s+(?:know|see|understand|check|find out)\s+)?)"
    r"|(?:can|could|would)\s+you\s+(?:please\s+)?(?:tell|show|give)\s+me\s+|tell\s+me\s+|give\s+me\s+|show\s+me\s+)?",
    re.I,
)


def _content_count(text: str) -> int:
    return len([t for t in content_tokens(text) if t not in REQUEST_WORDS])


def heuristic_split(
    utterance: str, index: Index | None, embedder: Embedder | None = None, carry_cos: float = 0.35
) -> list[dict]:
    parts = [p.strip(" ,.;") for p in _SPLIT.split(utterance) if p and p.strip(" ,.;")]
    # "the cancellation policy and the catering options" -> two needs
    expanded: list[str] = []
    for p in parts:
        pieces = [q.strip(" ,") for q in _INNER_AND.split(p)]
        if len(pieces) > 1 and all(_content_count(q) >= 2 for q in pieces):
            expanded.extend(pieces)
        else:
            expanded.append(p)
    # Fragments too thin to search on are folded into their neighbour.
    clauses: list[str] = []
    for p in expanded:
        if clauses and _content_count(p) < 2:
            clauses[-1] = f"{clauses[-1]} {p}"
        else:
            clauses.append(p)
    if clauses and _content_count(clauses[0]) < 2 and len(clauses) > 1:
        clauses[1] = f"{clauses[0]} {clauses[1]}"
        clauses = clauses[1:]
    if not clauses:
        clauses = [utterance]

    context = proper_entities(clauses[0])[:3]
    related = [True] * len(clauses)
    if context and embedder is not None and len(clauses) > 1:
        vecs = embedder.embed(clauses, kind="text")
        related = [True] + [float(vecs[i] @ vecs[0]) >= carry_cos for i in range(1, len(clauses))]
    out = []
    for i, clause in enumerate(clauses):
        stripped = _FILLER.sub("", clause).strip() or clause
        carry = []
        if i > 0 and related[i]:
            carry = [c for c in context if c.lower() not in stripped.lower()]
        out.append({"text": retrieval_shape(stripped, carry), "span": clause, "confidence": 0.6})
    return out


def format_passages(hits: list[Hit], n: int, chars: int = 300) -> str:
    """The top passages an early search found, titled, for the decomposer to read."""
    lines = []
    for h in sorted(hits, key=lambda h: -h.score)[:n]:
        text = " ".join(h.chunk.text.split())
        lines.append(f"- [{heading_subject(h.chunk.heading) or h.chunk.doc_label}] {text[:chars]}")
    return "\n".join(lines)


async def _llm_split(
    utterance: str,
    context: str,
    model: ChatModel,
    ledger: UsageLedger,
    s: Settings,
    examples: str = "",
    passages: str = "",
) -> list[dict]:
    system, user = render_prompt(
        s.prompts_dir,
        "decompose",
        max_subqueries=str(s.max_subqueries),
        utterance=utterance,
        context=context or "(none)",
        examples=examples or "(none)",
        passages=passages or "(none yet)",
    )
    raw = await model.complete(system, user, ledger=ledger, step="decompose", json_mode=True, max_tokens=400)
    items = parse_json(raw).get("sub_queries", [])
    out = []
    for item in items:
        if isinstance(item, dict) and str(item.get("text", "")).strip():
            out.append(
                {
                    "text": str(item["text"]).strip()[:200],
                    "span": str(item.get("span", ""))[:300],
                    "confidence": float(item.get("confidence", 0.8) or 0.8),
                }
            )
    if not out:
        raise ValueError("decomposer returned no sub-queries")
    return out


def _numbers_differ(a: dict, b: dict) -> bool:
    """Both name numbers, and not the same ones: "2014 winner" vs "2018 winner". A number
    only one of them has ("for 30 people") is added context, not a different reading."""
    numbers = lambda item: {t for t in raw_words(item["text"]) if is_number(t)}  # noqa: E731
    na, nb = numbers(a), numbers(b)
    return bool(na) and bool(nb) and na != nb


def _words(item: dict) -> set[str]:
    return set(content_tokens(item["text"]))


def _duplicate(a: dict, b: dict, cos: float, s: Settings) -> bool:
    """One need asked twice: near-identical meaning, or one query that only adds context words
    to the other ("cancellation policy Pune" inside "cancellation policy workshop venue Pune").
    Never when a number differs: "2014 winner" and "2018 winner" are two readings."""
    if _numbers_differ(a, b):
        return False
    if cos >= s.merge_cos:
        return True
    wa, wb = _words(a), _words(b)
    return bool(wa) and bool(wb) and (wa <= wb or wb <= wa)


def guard(items: list[dict], embedder: Embedder, s: Settings) -> tuple[list[dict], int, int]:
    """Merge near-duplicates, then cap."""
    if len(items) <= 1:
        return items, 0, 0
    vecs = embedder.embed([i["text"] for i in items], kind="query")
    keep: list[int] = []
    merged = 0
    order = sorted(range(len(items)), key=lambda i: -items[i]["confidence"])
    for i in order:
        slot = next(
            (n for n, j in enumerate(keep) if _duplicate(items[i], items[j], float(vecs[i] @ vecs[j]), s)), None
        )
        if slot is None:
            keep.append(i)
            continue
        merged += 1
        if _words(items[i]) > _words(items[keep[slot]]):
            keep[slot] = i  # the more specific phrasing carries the shared context

    keep.sort()  # utterance order
    capped = max(0, len(keep) - s.max_subqueries)
    kept = [items[i] for i in keep[: s.max_subqueries]]
    return kept, merged, capped


async def decompose(
    utterance: str,
    *,
    turn_id: str,
    model: ChatModel | None,
    embedder: Embedder,
    index: Index | None,
    settings: Settings,
    ledger: UsageLedger,
    context: str = "",
    examples: str = "",
    passages: str = "",
) -> Decomposition:
    started = time.perf_counter()
    method = "heuristic"
    items: list[dict]
    if not settings.decompose:
        # The baseline: the utterance, as spoken, is the only query.
        items, method = [{"text": utterance, "span": utterance, "confidence": 1.0}], "off"
    elif model is not None:
        try:
            items = await _llm_split(utterance, context, model, ledger, settings, examples, passages)
            method = "llm"
        except Exception as exc:
            log.warning("LLM decomposition failed (%s); using heuristic split", exc)
            items = heuristic_split(utterance, index, embedder, settings.carry_cos)
            method = "heuristic_fallback"
    else:
        items = heuristic_split(utterance, index, embedder, settings.carry_cos)
    raw = list(items)
    items, merged, capped = guard(items, embedder, settings)
    subs = [
        SubQuery(
            id=f"{turn_id}_sq{n}",
            text=i["text"],
            source="decomposed",
            span=i.get("span", ""),
            confidence=i.get("confidence", 0.8),
        )
        for n, i in enumerate(items, start=1)
    ]
    ms = (time.perf_counter() - started) * 1000
    if method != "llm":
        ledger.record_compute("decompose", ms, method)
    return Decomposition(subs, method, merged, capped, raw, ms)
