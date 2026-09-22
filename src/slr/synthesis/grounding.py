"""Grounding: nothing unverified reaches the user.

Generation is streamed, but the stream is released one *validated sentence* at
a time. For each sentence:

1. Every citation marker is matched against the retrieved set. A marker that
   does not resolve is a fabricated citation: it is stripped and counted
   (``fabricated_blocked``). The shipped answer therefore carries zero
   fabricated citations by construction, not by request.
2. A sentence with no resolvable marker is attributed to the best-supporting
   retrieved chunk if one clears the bar; otherwise it is withheld and listed
   under uncertainty.
3. Support is scored between the claim and its cited chunks (NLI entailment,
   max over sentence windows). Below ``support_min`` the claim is demoted into
   ``uncertainty`` rather than shipped as fact.

``citation_support_rate = supported / generated claims``.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

from slr.contracts import Claim, Hit
from slr.retrieval.context import EvidencePackage
from slr.retrieval.rerank import predict_by_length
from slr.text import (
    containment,
    content_tokens,
    ends_with_abbreviation,
    heading_subject,
    sentences,
)

log = logging.getLogger(__name__)

#: anything bracketed that looks like an attempt at a citation
CANDIDATE_MARKER = re.compile(r"\[\s*([A-Za-z][A-Za-z0-9_.:-]*)\s*(?:§\s*([^\[\]\n]{1,40}))?\]")
#: the answer model's own attribution, written before each paragraph and never shown
EVIDENCE_LINE = re.compile(r"^\s*(?:[-*]\s*)?(?:\*\*)?EVIDENCE(?:\*\*)?\s*:\s*(.*)$", re.I | re.S)
UNCERTAIN_PREFIX = re.compile(r"^\s*(?:[-*]\s*)?(?:\*\*)?UNCERTAIN(?:\*\*)?\s*:\s*", re.I)
#: A sentence whose point is that the documents do NOT say something ("the lead
#: actor is not mentioned in the retrieved documents"). The prompt asks for these
#: as UNCERTAIN lines; written as prose they are still uncertainty, not a claim
#: about the world, and verifying them against the evidence would only fail.
ABSENCE = re.compile(
    r"\b(?:not|never|no)\b[^.]{0,50}\b(?:mention(?:ed)?|specif(?:y|ied|ies)|stated?|provided?|includ(?:e|ed)|"
    r"contain(?:s|ed)?|verif(?:y|ied)|found|given|listed|identified|named|detailed)\b[^.]{0,40}"
    r"\b(?:documents?|evidence|sources?|passages?|retrieved|provided (?:text|information))\b"
    r"|\b(?:the )?(?:evidence|documents?|sources?|retrieved (?:documents?|evidence|passages?))\b[^.]{0,30}"
    r"\b(?:do(?:es)? not|don't|doesn't|did not|fails? to)\b"
    # "There is no information available regarding …", "no evidence indicating …"
    r"|\bno (?:specific |further |additional |relevant )?(?:information|evidence|data)\b[^.]{0,20}"
    r"\b(?:available|found|indicat(?:es|ing))\b"
    # "… is unknown from the provided documents"
    r"|\b(?:unknown|unclear) (?:from|in) the (?:provided |retrieved )?(?:documents?|evidence|sources?)\b"
    # first person only: "the venue cannot provide AV equipment" is a claim, "I cannot provide" is not
    r"|\b(?:I|we) (?:cannot|can't|could not|am unable to|are unable to) (?:provide|determine|confirm|identify)\b",
    re.I,
)


#: a discourse connective opening a sentence, which carries no fact of its own
CONNECTIVE = re.compile(
    r"^\s*(?:additionally|also|however|furthermore|moreover|in addition|meanwhile|similarly|likewise|"
    r"notably|overall|specifically|in contrast|by contrast|on the other hand)\s*,\s*",
    re.I,
)


_PERCENT = re.compile(r"(\d)\s*%")


def _units(text: str) -> str:
    """One spelling for the same quantity: "65.46%" and "65.46 percent" say the same thing."""
    return _PERCENT.sub(r"\1 percent", text)


def normalise_marker(doc: str, section: str | None) -> str:
    return f"[{doc} §{(section or '').strip()}]".lower()


# --------------------------------------------------------------------------
# verifiers
# --------------------------------------------------------------------------


class Verifier(Protocol):
    name: str

    def support(self, claim: str, sources: list[str], contexts: list[str] | None = None) -> list[float]: ...


class LexicalVerifier:
    """Offline fallback: share of the claim's content words found in the source."""

    name = "lexical"

    def support(self, claim: str, sources: list[str], contexts: list[str] | None = None) -> list[float]:
        ctx = contexts or [""] * len(sources)
        return [containment(claim, f"{c} {s}".strip()) for c, s in zip(ctx, sources)]


class NliVerifier:
    """Entailment probability, max over overlapping sentence windows of the source."""

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self._model = CrossEncoder(model_name, device="cpu")
        labels = {v.lower(): k for k, v in self._model.model.config.id2label.items()}
        self._entail = labels.get("entailment", 1)
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str], float] = {}

    #: Windows are small on purpose. These models are trained on single-sentence
    #: premises: appending one unrelated sentence to a premise that entails the
    #: claim flips the verdict to neutral (measured: 0.99 -> 0.02). Scoring
    #: 1- and 2-sentence windows keeps the premise in distribution, and the
    #: 2-sentence windows cover claims that join two adjacent facts.
    @staticmethod
    def _windows(source: str, claim: str, limit: int = 5) -> list[str]:
        sents = sentences(source)
        if not sents:
            return [source]
        wins = list(sents) + [" ".join(sents[i : i + 2]) for i in range(len(sents) - 1)]
        wins.sort(key=lambda w: -containment(claim, w))
        return wins[:limit]

    def support(self, claim: str, sources: list[str], contexts: list[str] | None = None) -> list[float]:
        """``contexts[i]`` names what source ``i`` is about (its page title and section), put in
        front of every window so "the game" or "it" in a single sentence resolves to its subject."""
        ctx = contexts or [""] * len(sources)
        # "However, the venue opened in 2004" asserts what the bare
        # clause asserts; the connective only links it to the previous sentence.
        claim = _units(CONNECTIVE.sub("", claim, count=1) or claim)
        sources = [_units(s) for s in sources]
        out: list[float | None] = [self._cache.get((claim, c, s)) for c, s in zip(ctx, sources)]
        todo = [i for i, v in enumerate(out) if v is None]
        if todo:
            pairs, owner = [], []
            for i in todo:
                for w in self._windows(sources[i], claim):
                    pairs.append((f"{ctx[i]}: {w}" if ctx[i] else w, claim))
                    owner.append(i)
            with self._lock:
                probs = predict_by_length(self._model, pairs, batch_size=32, apply_softmax=True)
            best: dict[int, float] = {}
            for i, p in zip(owner, probs):
                best[i] = max(best.get(i, 0.0), float(p[self._entail]))
            for i in todo:
                # A claim whose every content word is in the source is lexically
                # supported even where the entailment model stays unsure.
                lexical = containment(claim, f"{ctx[i]} {sources[i]}".strip())
                out[i] = max(best.get(i, 0.0), lexical if lexical >= 0.9 else 0.0)
                self._cache[(claim, ctx[i], sources[i])] = out[i]
        return [float(v) for v in out]  # type: ignore[arg-type]


@lru_cache(maxsize=2)
def load_nli(model_name: str) -> NliVerifier:
    return NliVerifier(model_name)


class FactChecker:
    """A model trained for exactly this question: is this sentence supported by this document?

    MiniCheck (Tang, Laban and Durrett, EMNLP 2024) reads a whole passage against a
    sentence and was trained on claims that join facts across sentences, so it needs no
    sentence windows. It is larger and slower than the NLI model, so it only sees the
    sentences the NLI model would reject (``CascadeVerifier``).
    """

    def __init__(self, model_name: str, max_length: int = 512):
        from sentence_transformers import CrossEncoder

        self.name = model_name
        try:
            # From the local cache first. Loaded online, a checkpoint without safetensors
            # makes transformers start a conversion process, which on Windows re-imports
            # the caller's main module.
            self._model = CrossEncoder(model_name, max_length=max_length, device="cpu", local_files_only=True)
        except OSError:
            self._model = CrossEncoder(model_name, max_length=max_length, device="cpu")
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str, str], float] = {}

    def support(self, claim: str, sources: list[str], contexts: list[str] | None = None) -> list[float]:
        ctx = contexts or [""] * len(sources)
        claim = _units(CONNECTIVE.sub("", claim, count=1) or claim)
        docs = [_units(f"{c}: {s}" if c else s) for c, s in zip(ctx, sources)]
        out: list[float | None] = [self._cache.get((claim, c, s)) for c, s in zip(ctx, sources)]
        todo = [i for i, v in enumerate(out) if v is None]
        if todo:
            with self._lock:
                probs = predict_by_length(self._model, [(docs[i], claim) for i in todo], batch_size=8, apply_softmax=True)
            for i, p in zip(todo, probs):
                out[i] = float(p[1])  # label 1 = supported
                self._cache[(claim, ctx[i], sources[i])] = out[i]
        return [float(v) for v in out]  # type: ignore[arg-type]


@lru_cache(maxsize=2)
def load_checker(model_name: str) -> FactChecker:
    return FactChecker(model_name)


class CascadeVerifier:
    """The fast verifier decides; a sentence it scores below ``below`` gets a second reading.

    Both readings apply the grounder's usual bars, so the cascade never lowers a threshold:
    it corrects the small model's false rejections, such as "France is the most recent
    winner, having won in 2018" read against "The current champion is France, who won
    the title in 2018".
    """

    def __init__(self, first: Verifier, second: Verifier, below: float):
        self.first, self.second, self.below = first, second, below
        self.name = f"{first.name} + {second.name}"

    def support(self, claim: str, sources: list[str], contexts: list[str] | None = None) -> list[float]:
        scores = self.first.support(claim, sources, contexts)
        todo = [i for i, v in enumerate(scores) if v < self.below]
        if todo:
            ctx = contexts or [""] * len(sources)
            again = self.second.support(claim, [sources[i] for i in todo], [ctx[i] for i in todo])
            for i, v in zip(todo, again):
                scores[i] = max(scores[i], v)
        return scores


# --------------------------------------------------------------------------
# incremental sentence splitting
# --------------------------------------------------------------------------

#: sentence punctuation, then any citation markers that trail it, then whitespace
_END = re.compile(r"[.!?](?:\s*\[[^\[\]\n]*\])*(\s+)")


class SentenceStream:
    """Feed model deltas, get back complete segments (sentence + trailing space)."""

    def __init__(self) -> None:
        self.buf = ""

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        out = []
        while (seg := self._next()) is not None:
            out.append(seg)
        return out

    def flush(self) -> list[str]:
        rest, self.buf = self.buf, ""
        return [rest] if rest else []

    def _next(self) -> str | None:
        buf = self.buf
        if buf.lstrip().lstrip("-* ").upper().startswith("EVIDENCE"):
            # A quote may hold several sentences; the whole line is one segment.
            nl = buf.find("\n")
            if nl == -1:
                return None
            seg, self.buf = buf[: nl + 1], buf[nl + 1 :]
            return seg
        cut = None
        for m in _END.finditer(buf):
            end = m.end()
            if end >= len(buf):
                break  # cannot yet see what follows the whitespace
            if "\n" not in m.group(1) and (buf[end] == "[" or _open_bracket(buf[: m.start()])):
                continue  # a marker still belongs to this sentence
            if "\n" not in m.group(1) and ends_with_abbreviation(buf[: m.start() + 1]):
                continue  # "St. John": the full stop belongs to a word, not the sentence
            cut = end
            break
        nl = buf.find("\n")
        if nl != -1 and (cut is None or nl < cut):
            end = nl
            while end < len(buf) and buf[end] == "\n":
                end += 1
            if end >= len(buf):
                return None  # more newlines may belong to this break
            cut = end
        if cut is None:
            return None
        seg, self.buf = buf[:cut], buf[cut:]
        return seg


def _open_bracket(text: str) -> bool:
    return text.rfind("[") > text.rfind("]")


# --------------------------------------------------------------------------
# the grounder
# --------------------------------------------------------------------------


@dataclass
class GroundedSegment:
    text: str  # what to stream ("" when withheld)
    claim: Claim | None = None
    uncertainty: str | None = None


@dataclass
class Grounder:
    evidence: EvidencePackage
    verifier: Verifier
    support_min: float
    auto_cite_min: float
    #: how many retrieved blocks a failed citation may be re-checked against
    attribution_pool: int
    claim_prefix: str
    default_sub_query: str
    claims: list[Claim] = field(default_factory=list)
    uncertainty: list[str] = field(default_factory=list)
    generated: int = 0
    supported: int = 0
    fabricated_blocked: int = 0
    auto_cited: int = 0
    #: cited claims moved to a different retrieved block that does support them
    recited: int = 0
    #: claims withheld because a figure in them was in no retrieved block
    ungrounded_numbers: int = 0
    demoted: int = 0
    fabricated_markers: list[str] = field(default_factory=list)
    #: what the model quoted before each paragraph ("NONE" when it found nothing)
    attributions: list[str] = field(default_factory=list)
    _cmap: dict[str, Hit] = field(init=False)

    def __post_init__(self) -> None:
        self._cmap = self.evidence.citation_map

    @property
    def support_rate(self) -> float:
        return self.supported / self.generated if self.generated else 1.0

    def _next_id(self) -> str:
        return f"{self.claim_prefix}_c{len(self.claims) + 1}"

    def resolve(self, text: str) -> tuple[str, list[Hit]]:
        """Strip every marker; return (bare text, resolved hits). Counts fabrications."""
        hits: list[Hit] = []

        def sub(m: re.Match) -> str:
            doc, section = m.group(1), m.group(2)
            looks_like_citation = section is not None or doc.lower().startswith("doc")
            if not looks_like_citation:
                return m.group(0)  # ordinary bracketed prose
            hit = self._cmap.get(normalise_marker(doc, section))
            if hit is None:
                self.fabricated_blocked += 1
                self.fabricated_markers.append(m.group(0))
            elif hit not in hits:
                hits.append(hit)
            return ""

        bare = CANDIDATE_MARKER.sub(sub, text)
        bare = re.sub(r"\s+([.,;:!?])", r"\1", bare)
        bare = re.sub(r"[ \t]{2,}", " ", bare).strip()
        return bare, hits

    def _score(self, claim: str, hits: list[Hit]) -> list[float]:
        if not hits:
            return []
        return self.verifier.support(claim, [h.chunk.text for h in hits], [_subject(h) for h in hits])

    def _best_attribution(self, claim: str, exclude: list[Hit] | None = None) -> tuple[Hit | None, float]:
        """The retrieved block that best supports this sentence, best word-overlap first.

        Word overlap is a weak proxy for what a verifier will accept — a block that states
        the fact in other words sits well down the order — so the search goes as deep as
        ``attribution_pool`` and stops at the first block over the bar rather than scoring
        the whole pool: measured on ASQA, the supporting block is outside the top three for
        about half the sentences that reach here.
        """
        others = [h for h in self.evidence.hits if h not in (exclude or [])]
        pool = sorted(others, key=lambda h: -containment(claim, h.chunk.text))[: self.attribution_pool]
        best, best_score = None, 0.0
        for hit in pool:
            score = self._score(claim, [hit])[0]
            if score > best_score:
                best, best_score = hit, score
            if best_score >= self.auto_cite_min:
                break
        return best, best_score

    def ground(self, sentence: str, *, count: bool = True) -> tuple[str, list[Hit], float] | None:
        """Validate one sentence. Returns (shipped text, hits, support) or None if withheld."""
        bare, hits = self.resolve(sentence)
        if not content_tokens(bare):
            return None
        if count:
            self.generated += 1
        if not hits:
            best, score = self._best_attribution(bare)
            if best is None or score < max(self.support_min, self.auto_cite_min):
                self.demoted += 1
                self.uncertainty.append(f"Not verified in the retrieved documents: {_plain(bare)}")
                return None
            hits, support = [best], score
            self.auto_cited += 1
        else:
            scores = self._score(bare, hits)
            support = max(scores) if scores else 0.0
            if support < self.support_min:
                # The model cited the wrong block. If another retrieved block states the
                # fact, cite that one instead, at the same bar as an engine-picked citation.
                best, score = self._best_attribution(bare, exclude=hits)
                if best is None or score < max(self.support_min, self.auto_cite_min):
                    self.demoted += 1
                    self.uncertainty.append(f"Not verified in the cited documents: {_plain(bare)}")
                    return None
                hits, scores, support = [best], [score], score
                self.recited += 1
            # keep only the markers that actually support the claim
            strong = [h for h, s in zip(hits, scores) if s >= self.support_min]
            hits = strong or hits
        missing = _ungrounded_numbers(bare, hits)
        if missing:
            # A number the cited blocks do not contain: entailment models read "1983 and
            # 2011" as close enough to a block that says 1983. Try a block that does carry
            # every number, at the engine's own bar; otherwise the sentence is withheld.
            best, score = self._best_attribution(bare, exclude=hits)
            if best is None or score < self.auto_cite_min or _ungrounded_numbers(bare, [best]):
                self.demoted += 1
                self.ungrounded_numbers += 1
                self.uncertainty.append(
                    f"Not verified in the retrieved documents ({', '.join(missing)} is not in them): {_plain(bare)}"
                )
                return None
            hits, support = [best], score
            self.recited += 1
        if count:
            self.supported += 1
        return _attach(bare, hits), hits, support

    def process(self, segment: str) -> GroundedSegment:
        if not segment.strip():
            return GroundedSegment(text=segment if self.claims else "")
        trailing = segment[len(segment.rstrip()) :]
        body = segment.strip()
        if m := EVIDENCE_LINE.match(body):
            self.attributions.append(m.group(1).strip()[:300])
            return GroundedSegment(text="")
        if UNCERTAIN_PREFIX.match(body) or ABSENCE.search(body):
            note = UNCERTAIN_PREFIX.sub("", body)
            note, _ = self.resolve(note)
            if note:
                self.uncertainty.append(note)
            return GroundedSegment(text="", uncertainty=note)
        lead = re.match(r"^([-*•]\s+|\d+[.)]\s+)", body)
        prefix = lead.group(1) if lead else ""
        result = self.ground(body[len(prefix) :])
        if result is None:
            return GroundedSegment(text="")
        shipped, hits, support = result
        claim = Claim(
            id=self._next_id(),
            text=shipped,
            chunk_ids=tuple(h.chunk.chunk_id for h in hits),
            sub_query_id=_sub_query_of(hits, self.default_sub_query),
            support=support,
        )
        self.claims.append(claim)
        return GroundedSegment(text=prefix + shipped + (trailing or " "), claim=claim)


#: a number as written in prose: 1983, 65.46, 15,921, 2,297
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
#: ordinals and small counts that read as words elsewhere in the same fact
_NUMBER_STOP = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}


def _figures(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUMBER.findall(text)}


def _ungrounded_numbers(claim: str, hits: list[Hit]) -> list[str]:
    """Figures in the claim that none of its blocks contain.

    A number is the part of a sentence an entailment model is least likely to check and a
    reader most likely to act on. Small counts are skipped: "the two houses" against a block
    that writes "two" as a word is a spelling difference, not an invented figure.
    """
    if not hits:
        return []
    source = _figures(" ".join(f"{h.chunk.heading} {h.chunk.text}" for h in hits))
    return sorted(n for n in _figures(claim) - source if n not in _NUMBER_STOP)


def _plain(text: str) -> str:
    return text.strip().rstrip(".")


def _attach(bare: str, hits: list[Hit]) -> str:
    markers = " ".join(h.chunk.citation for h in hits)
    m = re.search(r"([.!?]+)[\"')\]]*$", bare)
    if m:
        return f"{bare[: m.start()].rstrip()} {markers}{bare[m.start():]}"
    return f"{bare} {markers}."


def _sub_query_of(hits: list[Hit], default: str) -> str:
    counts: dict[str, int] = {}
    for h in hits:
        for sq in h.sub_query_ids:
            counts[sq] = counts.get(sq, 0) + 1
    if not counts:
        return default
    return max(counts, key=lambda k: (counts[k], k == default))


def _subject(hit: Hit) -> str:
    """What a chunk is about, from its heading trail: page title and named sections, not numbers."""
    return heading_subject(hit.chunk.heading)
