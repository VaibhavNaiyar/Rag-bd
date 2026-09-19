"""Streamed answer generation from evidence only.

Two producers share one output format (sentences ending in citation markers,
``UNCERTAIN:`` lines for gaps), so the grounder downstream is identical:

* ``llm_answer`` — the synthesise prompt, streamed.
* ``extractive_answer`` — deterministic, offline: picks the evidence sentences
  that best cover each sub-query. Grounded by construction; used when no LLM
  is configured.

``restructure`` handles presentation-only turns from the previous answer alone.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

from slr.config import Settings
from slr.contracts import AnswerVersion, SubQuery
from slr.llm import ChatModel, render_prompt
from slr.retrieval.context import EvidencePackage
from slr.telemetry.cost import UsageLedger
from slr.text import REQUEST_WORDS, content_tokens, sentences

#: a number or year in a sub-question ("2001", "30", "1,500")
_NUMBER = re.compile(r"\b\d(?:[\d,.]*\d)?\b")


def missing_numbers(sub_query: SubQuery, evidence: EvidencePackage) -> list[str]:
    """Numbers the sub-question hinges on that no retrieved block contains.

    "Venue capacity for the 2031 offsite" against evidence that never says 2031
    cannot be answered from it, whatever the model happens to remember.
    """
    return [n for n in dict.fromkeys(_NUMBER.findall(sub_query.text)) if n not in evidence.text]


def format_sub_queries(sub_queries: list[SubQuery], evidence: EvidencePackage | None = None) -> str:
    lines = []
    for i, sq in enumerate(sub_queries, start=1):
        gap = missing_numbers(sq, evidence) if evidence is not None else []
        note = f" (no retrieved block mentions {', '.join(gap)})" if gap else ""
        lines.append(f"{i}. ({sq.id}) {sq.text}{note}")
    return "\n".join(lines)


async def llm_answer(
    model: ChatModel,
    utterance: str,
    sub_queries: list[SubQuery],
    evidence: EvidencePackage,
    ledger: UsageLedger,
    s: Settings,
) -> AsyncIterator[str]:
    system, user = render_prompt(
        s.prompts_dir,
        "synthesise",
        utterance=utterance,
        sub_queries=format_sub_queries(sub_queries, evidence),
        evidence=evidence.text or "(no evidence was retrieved)",
    )
    async for delta in model.stream(system, user, ledger=ledger, step="synthesise", max_tokens=700):
        yield delta


#: Share of a question's content words that may be missing from the corpus
#: vocabulary before the question counts as one the corpus cannot answer.
ABSENT_SHARE_MAX = 0.4


def _weights(query_terms: set[str], idf: dict[str, float] | None) -> dict[str, float]:
    """IDF weight per query term, over the terms the corpus actually uses.

    Coverage is measured against what the corpus *could* match: a question is
    not unanswered because it used a synonym the documents never use.
    """
    if not idf:
        return {t: 1.0 for t in query_terms}
    return {t: idf[t] for t in query_terms if t in idf}


def _absent_share(query_terms: set[str], idf: dict[str, float] | None) -> float:
    """How much of the question is vocabulary the corpus does not contain.

    This is the signal that separates "the corpus does not discuss pets" from
    "the corpus says booked where the speaker said book": a topic word absent
    from the whole index means the question is about something not in scope,
    however well the remaining generic words match.
    """
    if not idf or not query_terms:
        return 0.0
    return sum(t not in idf for t in query_terms) / len(query_terms)


def _sentence_score(weights: dict[str, float], sentence: str) -> float:
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    terms = set(content_tokens(sentence))
    return sum(w for t, w in weights.items() if t in terms) / total


async def extractive_answer(
    utterance: str,
    sub_queries: list[SubQuery],
    evidence: EvidencePackage,
    min_cover: float = 0.33,
    idf: dict[str, float] | None = None,
) -> AsyncIterator[str]:
    used: set[str] = set()
    gaps: list[str] = []
    first = True
    for sq in sub_queries:
        terms = {t for t in content_tokens(sq.text) if t not in REQUEST_WORDS}
        weights = _weights(terms, idf)
        if not weights or _absent_share(terms, idf) > ABSENT_SHARE_MAX:
            gaps.append(sq.text)
            continue
        hits = [h for h in evidence.hits if sq.id in h.sub_query_ids] or evidence.hits
        candidates = []
        for rank, hit in enumerate(hits[:4]):
            for sent in sentences(hit.chunk.text):
                if sent in used or len(sent.split()) < 4:
                    continue
                # strip speaker labels from transcript lines
                clean = re.sub(r"^[A-Z][\w .'-]{0,30}:\s+", "", sent)
                candidates.append((rank, clean, hit))
        scored = [(_sentence_score(weights, c[1]) - 0.03 * c[0], c[1], c[2]) for c in candidates]
        scored.sort(key=lambda x: -x[0])
        picked = [x for x in scored if x[0] >= min_cover][:3]
        if not picked:
            gaps.append(sq.text)
            continue
        if not first:
            yield "\n\n"
        first = False
        for _, sent, hit in picked:
            used.add(sent)
            body = sent.rstrip()
            end = body[-1] if body[-1] in ".!?" else "."
            body = body.rstrip(".!?")
            for word in f"{body} {hit.chunk.citation}{end} ".split(" "):
                if word:
                    yield word + " "
    for gap in gaps:
        yield f"\nUNCERTAIN: The retrieved documents do not answer: {gap}\n"


_COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


async def restructure(
    model: ChatModel | None,
    utterance: str,
    previous: AnswerVersion,
    ledger: UsageLedger,
    s: Settings,
) -> AsyncIterator[str]:
    if model is not None:
        system, user = render_prompt(s.prompts_dir, "restructure", answer=previous.body, utterance=utterance)
        async for delta in model.stream(system, user, ledger=ledger, step="restructure", max_tokens=500):
            yield delta
        return
    # Offline: regroup the existing claims; nothing new is written.
    claims = list(previous.claims)
    if not claims:
        return
    m = re.search(r"\b(\d+|" + "|".join(_COUNT_WORDS) + r")\b", utterance.lower())
    n = len(claims)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else _COUNT_WORDS[m.group(1)]
    elif re.search(r"\b(short|shorter|shorten|brief|briefly|concise|tl;?dr|condense)\b", utterance, re.I):
        n = max(1, (len(claims) + 1) // 2)
    n = max(1, min(n, len(claims)))
    bullets = re.search(r"\bbullet|\blist\b|\bpoints?\b", utterance, re.I) is not None
    if n < len(claims) and not bullets:
        claims = claims[:n]  # "shorter": keep the leading claims
        groups = [[c] for c in claims]
    else:
        size = -(-len(claims) // n)
        groups = [claims[i : i + size] for i in range(0, len(claims), size)]
    for g in groups:
        text = " ".join(c.text for c in g)
        yield ("- " if bullets else "") + text + ("\n" if bullets else " ")
