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
from slr.text import containment, content_tokens, sentences

log = logging.getLogger(__name__)

#: anything bracketed that looks like an attempt at a citation
CANDIDATE_MARKER = re.compile(r"\[\s*([A-Za-z][A-Za-z0-9_.:-]*)\s*(?:§\s*([^\[\]\n]{1,40}))?\]")
UNCERTAIN_PREFIX = re.compile(r"^\s*(?:[-*]\s*)?(?:\*\*)?UNCERTAIN(?:\*\*)?\s*:\s*", re.I)


def normalise_marker(doc: str, section: str | None) -> str:
    return f"[{doc} §{(section or '').strip()}]".lower()


# --------------------------------------------------------------------------
# verifiers
# --------------------------------------------------------------------------


class Verifier(Protocol):
    name: str

    def support(self, claim: str, sources: list[str]) -> list[float]: ...


class LexicalVerifier:
    """Offline fallback: share of the claim's content words found in the source."""

    name = "lexical"

    def support(self, claim: str, sources: list[str]) -> list[float]:
        return [containment(claim, s) for s in sources]


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

    def support(self, claim: str, sources: list[str]) -> list[float]:
        out: list[float | None] = [self._cache.get((claim, s)) for s in sources]
        todo = [i for i, v in enumerate(out) if v is None]
        if todo:
            pairs, owner = [], []
            for i in todo:
                for w in self._windows(sources[i], claim):
                    pairs.append((w, claim))
                    owner.append(i)
            with self._lock:
                probs = self._model.predict(pairs, apply_softmax=True, batch_size=32, show_progress_bar=False)
            best: dict[int, float] = {}
            for i, p in zip(owner, probs):
                best[i] = max(best.get(i, 0.0), float(p[self._entail]))
            for i in todo:
                # A claim whose every content word is in the source is lexically
                # supported even where the entailment model stays unsure.
                lexical = containment(claim, sources[i])
                out[i] = max(best.get(i, 0.0), lexical if lexical >= 0.9 else 0.0)
                self._cache[(claim, sources[i])] = out[i]
        return [float(v) for v in out]  # type: ignore[arg-type]


@lru_cache(maxsize=2)
def load_nli(model_name: str) -> NliVerifier:
    return NliVerifier(model_name)


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
        cut = None
        for m in _END.finditer(buf):
            end = m.end()
            if end >= len(buf):
                break  # cannot yet see what follows the whitespace
            if "\n" not in m.group(1) and (buf[end] == "[" or _open_bracket(buf[: m.start()])):
                continue  # a marker still belongs to this sentence
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
    claim_prefix: str
    default_sub_query: str
    claims: list[Claim] = field(default_factory=list)
    uncertainty: list[str] = field(default_factory=list)
    generated: int = 0
    supported: int = 0
    fabricated_blocked: int = 0
    auto_cited: int = 0
    demoted: int = 0
    fabricated_markers: list[str] = field(default_factory=list)
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
        return self.verifier.support(claim, [h.chunk.text for h in hits])

    def _best_attribution(self, claim: str) -> tuple[Hit | None, float]:
        pool = sorted(self.evidence.hits, key=lambda h: -containment(claim, h.chunk.text))[:3]
        scores = self._score(claim, pool)
        if not scores:
            return None, 0.0
        i = max(range(len(scores)), key=scores.__getitem__)
        return pool[i], scores[i]

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
                self.demoted += 1
                self.uncertainty.append(f"Not verified in the cited documents: {_plain(bare)}")
                return None
            # keep only the markers that actually support the claim
            strong = [h for h, s in zip(hits, scores) if s >= self.support_min]
            hits = strong or hits
        if count:
            self.supported += 1
        return _attach(bare, hits), hits, support

    def process(self, segment: str) -> GroundedSegment:
        if not segment.strip():
            return GroundedSegment(text=segment if self.claims else "")
        trailing = segment[len(segment.rstrip()) :]
        body = segment.strip()
        if UNCERTAIN_PREFIX.match(body):
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
