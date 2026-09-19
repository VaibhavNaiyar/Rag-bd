"""Tokenisation shared by BM25, the controller signals and lexical grounding.

Deliberately small and language-generic: no stemmer download, no NLP model.
"""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """
a about above after again against all also am an and any are aren't as at be because been before
being below between both but by can can't cannot could couldn't did didn't do does doesn't doing
don't down during each few for from further had hadn't has hasn't have haven't having he he'd
he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into
is isn't it it's its itself just let's me more most mustn't my myself no nor not of off on once
only or other ought our ours ourselves out over own same shan't she she'd she'll she's should
shouldn't so some such than that that's the their theirs them themselves then there there's these
they they'd they'll they're they've this those through to too under until up very was wasn't we
we'd we'll we're we've were weren't what what's when when's where where's which while who who's
whom why why's will with won't would wouldn't you you'd you'll you're you've your yours yourself
yourselves um uh like okay ok yeah hey hi please need want wanted know tell give show get got
maybe really actually just also well so thing things something anything kind sort bit lot lots
going gonna wanna let us i'd could would might may must shall one ones
""".split()
)

#: Words that ask for something; they carry intent, not retrievable content.
REQUEST_WORDS = frozenset(
    """what which who whom whose when where why how whether is are does do did can could should
    would will tell explain describe summarise summarize list give show find compare walk""".split()
)

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")
_NUMBER = re.compile(r"^\d")


def tokens(text: str) -> list[str]:
    return [t.lower().strip("'-") for t in _TOKEN.findall(text)]


def _norm(token: str) -> str:
    """Light suffix folding so a question's wording matches the corpus's.

    "booked" and "booking" must reach "book": a speaker says "how do I book a
    venue" and the document says "venues must be booked". No stemmer download,
    and conservative on short words where stripping changes the meaning.
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def content_tokens(text: str) -> list[str]:
    """Lower-cased, stopword-free, plural-folded tokens."""
    return [_norm(t) for t in tokens(text) if t and t not in STOPWORDS]


def raw_words(text: str) -> list[str]:
    return _TOKEN.findall(text)


def is_number(token: str) -> bool:
    return bool(_NUMBER.match(token))


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def sentences(text: str) -> list[str]:
    parts = []
    for block in re.split(r"\n\s*\n|\n(?=\s*[-*•]\s)", text):
        block = " ".join(block.split())
        if not block:
            continue
        parts.extend(s.strip() for s in _SENT_SPLIT.split(block) if s.strip())
    return parts


def containment(claim: str, source: str) -> float:
    """Share of the claim's content tokens that appear in the source."""
    c = set(content_tokens(claim))
    if not c:
        return 0.0
    s = set(content_tokens(source))
    return len(c & s) / len(c)


_SECTION_LABEL = re.compile(r"^(?:passage|section|part|page)\s*\d+$", re.I)


def heading_subject(heading: str) -> str:
    """What a chunk is about, from its heading trail: page title and named sections, not numbers."""
    parts = [p.strip() for p in heading.split("›")]
    return ", ".join(p for p in parts if p and not _SECTION_LABEL.match(p))
