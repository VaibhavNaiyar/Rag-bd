"""The four controller signals. No model call; each is a pure function.

1. semantic stability   — has the prefix's meaning stopped moving?
2. content sufficiency  — is there anything searchable yet?
3. suppression          — is this a presentation-only turn?
4. refinement           — is this a late constraint on the previous request?

"Entity" here is corpus-grounded rather than NER-based: a number, a
capitalised non-initial word, or a token the index itself marks as salient
(high IDF). That keeps the controller corpus-agnostic with no extra model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from slr.retrieval.store import Index
from slr.text import (
    REQUEST_WORDS,
    STOPWORDS,
    content_tokens,
    is_number,
    raw_words,
    tokens,
)

PRESENTATION_WORDS = frozenset(
    """repeat again shorter shorten short summarise summarize summary rephrase reword simplify simpler
    translate translation format bullet bullets point points list table briefly brief condense
    concise tldr recap restate paragraph paragraphs sentence sentences line lines word words
    answer response reply last previous above earlier said version into in hindi french german
    spanish english language one two three four five six seven eight nine ten plain simple
    words form way style longer expand read slowly back once more put write numbered bulleted simply
    clearly less""".split()
)
PRESENTATION_PATTERNS = re.compile(
    r"\b(repeat|again|shorten|shorter|summari[sz]e (that|it|this|what you)|in (\w+ )?bullet|bullet points?|"
    r"rephrase|reword|simplify|translate|as a (numbered |bulleted )?(table|list)|say that|tl;?dr|condense|"
    r"restate|recap|make (it|that) (shorter|brief|concise|simpler)|(put|write|give) (it|that|this) (in|as|into)|"
    r"in (one|a|two|three) (sentence|line|paragraph)s?)\b",
    re.I,
)
ANAPHORA = re.compile(
    r"\b(that|it|this|those|these|above|your (last|previous|earlier) (answer|response|reply)|"
    r"the (last|previous) (answer|response|one)|what you (just )?said)\b",
    re.I,
)
#: The speaker correcting what they just asked. "I meant X" can only be about
#: the previous request, so it needs no topical similarity to count.
CORRECTION_CUES = re.compile(
    r"\b(i meant|i mean|sorry|to be (more )?specific|specifically|to clarify|i should have said)\b"
    r"|^\s*(no|correction)\b",
    re.I,
)
REFINE_CUES = re.compile(
    r"^\s*(actually|also|but|oh|wait|however|instead|plus|additionally|and (also )?it|what if|"
    r"make (it|that)|turns out|correction|only|just)\b|\b(what if|instead of|turns out|by the way)\b",
    re.I,
)
QUESTION_START = re.compile(
    r"^\s*(what|which|who|whom|whose|when|where|why|how|is|are|does|do|did|can|could|should|would|will|"
    r"tell|explain|describe|summari[sz]e|list|give|show|find|compare|walk|i need|i want|i'd like)\b",
    re.I,
)
#: a clause closes at sentence punctuation or a comma — mid-chunk too, since a
#: transcript chunk rarely ends exactly where the speaker paused
CLAUSE_END = re.compile(r"[?.!;,](?:\s|$)")


@dataclass
class Sufficiency:
    ok: bool
    content: int
    entities: list[str]


def entities(text: str, index: Index | None) -> list[str]:
    words = raw_words(text)
    found = []
    for i, w in enumerate(words):
        low = w.lower()
        if low in STOPWORDS or low in REQUEST_WORDS:
            continue
        if is_number(w):
            found.append(w)
        elif i > 0 and w[:1].isupper() and low != "i":
            found.append(w)
        elif index is not None and index.is_salient(content_tokens(low)[0] if content_tokens(low) else low):
            found.append(w)
    return found


def proper_entities(text: str) -> list[str]:
    """Named things only: the shared context worth carrying into a sibling query.

    A salient common noun ("cancellation") belongs to the intent that said it;
    carrying it into the next intent's query poisons that search.
    """
    words = raw_words(text)
    return [w for i, w in enumerate(words) if i > 0 and w[:1].isupper() and w.lower() not in STOPWORDS]


def sufficiency(text: str, index: Index | None, min_content: int) -> Sufficiency:
    content = [t for t in content_tokens(text) if t not in REQUEST_WORDS]
    ents = entities(text, index)
    return Sufficiency(ok=len(content) >= min_content and len(ents) >= 1, content=len(content), entities=ents)


def stability(vecs: list[np.ndarray]) -> float:
    if len(vecs) < 2:
        return 0.0
    return float(np.dot(vecs[-1], vecs[-2]))


def clause_boundary(chunk_text: str) -> bool:
    return bool(CLAUSE_END.search(chunk_text.strip() + " "))


def new_content(text: str, previous_answer: str, index: Index | None) -> list[str]:
    """Content the prior answer does not already contain and that is not presentation vocabulary."""
    prior = set(content_tokens(previous_answer))
    out = []
    for w in raw_words(text):
        low = w.lower()
        if low in STOPWORDS or low in PRESENTATION_WORDS or low in REQUEST_WORDS or is_number(w):
            continue
        norm = content_tokens(low)
        if not norm or norm[0] in prior or norm[0] in PRESENTATION_WORDS:
            continue
        if index is not None and norm[0] not in index.idf:
            continue  # a word the corpus never uses cannot be a retrievable need
        out.append(w)
    return out


def is_presentation_only(text: str, previous_answer: str, index: Index | None) -> tuple[bool, dict]:
    lexicon = bool(PRESENTATION_PATTERNS.search(text))
    anaphora = bool(ANAPHORA.search(text))
    # Look for new content OUTSIDE the presentation request itself: the verb in
    # "make that shorter" is part of the request, not a topic the user added.
    rest = ANAPHORA.sub(" ", PRESENTATION_PATTERNS.sub(" ", text))
    fresh = new_content(rest, previous_answer, index)
    return lexicon and anaphora and not fresh, {"lexicon": lexicon, "anaphora": anaphora, "new_content": fresh}


def is_request(text: str) -> bool:
    return "?" in text or bool(QUESTION_START.search(text))


def refinement(text: str, vec: np.ndarray | None, previous_vec: np.ndarray | None, tau: float) -> tuple[bool, dict]:
    """Is this a late constraint on the previous request, or a new one?

    Four bars, by how much the sentence form already tells us:

    * a self-correction on a statement ("sorry, I meant the 2019 one") is
      decisive whenever there is a previous request: it can only narrow it;
    * a modifier cue on a statement ("but we also want an outside caterer") is
      the strongest signal — a low similarity bar, because a constraint often
      shares little vocabulary with the request it narrows;
    * a cue on a question still has to be about the same thing;
    * no cue at all: only a statement, and only when clearly on topic.

    A bare new question is never a refinement, however similar it looks.
    """
    correction = bool(CORRECTION_CUES.search(text))
    cue = correction or bool(REFINE_CUES.search(text))
    cos = float(np.dot(vec, previous_vec)) if vec is not None and previous_vec is not None else 0.0
    request = is_request(text)
    if correction and not request and previous_vec is not None:
        bar = -1.0  # below any cosine: decided by the cue alone
    elif cue and not request:
        bar = tau * 0.6
    elif cue:
        bar = tau * 0.9
    elif not request:
        bar = tau
    else:
        bar = 2.0  # unreachable: a fresh question starts a new topic
    decided = cos >= bar
    return decided, {"cue": cue, "correction": correction, "cos_previous": round(cos, 3), "is_request": request, "bar": round(bar, 3)}


def retrieval_shape(text: str, carry: list[str] | None = None, limit: int = 12) -> str:
    """Keyword form of a spoken fragment.

    "I need to know the seating limit for 30 people" -> "seating limit 30 people".
    Stopwords and asking-words go; names and numbers stay exactly as spoken.
    """
    words = []
    for w in raw_words(text):
        low = w.lower()
        if low in STOPWORDS or low in REQUEST_WORDS or low in {"plan", "planning", "question"}:
            continue
        if w not in words:
            words.append(w)
    for c in carry or []:
        if c.lower() not in {w.lower() for w in words}:
            words.append(c)
    return " ".join(words[:limit]) if words else " ".join(tokens(text)[:limit])
