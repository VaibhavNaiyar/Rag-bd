"""Transcript-aware and section-aware chunking.

* Documents split on heading boundaries, then pack paragraphs to a word bound.
  The heading trail becomes the chunk's heading, and an outline number becomes
  its citation section: ``[Doc_3 §2.1]``.
* Transcripts split on speaker turns and never mid-utterance. The section is
  the speaker and start time: ``[Doc_7 §Priya @05:02]``.
* PDFs split per page: ``[Doc_2 §page 4]``.

Citation markers are built here, from the stored record, and nowhere else.
"""

from __future__ import annotations

import re

from slr.contracts import Chunk, Document
from slr.text import sentences

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_PAGE = re.compile(r"^\f?\[\[page (\d+)\]\]$")
SECTION_MAX = 24


def _clock(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def _words(text: str) -> int:
    return len(text.split())


def _pack(paragraphs: list[str], target: int, hard_max: int) -> list[str]:
    """Greedy paragraph packing; oversize paragraphs split on sentences."""
    units: list[str] = []
    for para in paragraphs:
        if _words(para) <= hard_max:
            units.append(para)
            continue
        buf: list[str] = []
        for sent in sentences(para):
            if buf and _words(" ".join(buf)) + _words(sent) > target:
                units.append(" ".join(buf))
                buf = []
            buf.append(sent)
        if buf:
            units.append(" ".join(buf))
    out: list[str] = []
    buf = []
    for unit in units:
        if buf and _words("\n\n".join(buf)) + _words(unit) > target:
            out.append("\n\n".join(buf))
            buf = []
        buf.append(unit)
    if buf:
        out.append("\n\n".join(buf))
    return out


def _paragraphs(lines: list[str]) -> list[str]:
    text = "\n".join(lines).strip()
    if not text:
        return []
    paras = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if block:
            paras.append(block)
    return paras


def _sections(doc: Document) -> list[tuple[str, str, list[str]]]:
    """[(section_number, heading_trail, paragraphs)]"""
    lines = doc.text.splitlines()
    headings = [(i, len(m.group(1)), m.group(2).strip()) for i, ln in enumerate(lines) if (m := _HEADING.match(ln))]
    pages = [(i, int(m.group(1))) for i, ln in enumerate(lines) if (m := _PAGE.match(ln.strip()))]

    if pages:
        out = []
        bounds = pages + [(len(lines), -1)]
        for (start, page), (end, _) in zip(bounds, bounds[1:]):
            out.append((f"page {page}", f"{doc.title} › page {page}", _paragraphs(lines[start + 1 : end])))
        return out

    if not headings:
        return [("", doc.title, _paragraphs(lines))]

    # A single leading H1 is the document title, not a numbered section.
    title_idx = None
    h1s = [h for h in headings if h[1] == 1]
    if len(h1s) == 1 and headings[0][1] == 1:
        title_idx = headings[0][0]
    numbered = [h for h in headings if h[0] != title_idx]
    base_level = min((h[1] for h in numbered), default=1)

    out = []
    preamble_end = numbered[0][0] if numbered else len(lines)
    pre = [ln for i, ln in enumerate(lines[:preamble_end]) if i != title_idx]
    if _paragraphs(pre):
        out.append(("0", doc.title, _paragraphs(pre)))

    counters: list[int] = []
    trail: list[str] = []
    for n, (idx, level, title) in enumerate(numbered):
        depth = max(0, level - base_level)
        counters = counters[: depth + 1]
        trail = trail[:depth]
        while len(counters) < depth + 1:
            counters.append(0)
        counters[depth] += 1
        trail.append(title)
        end = numbered[n + 1][0] if n + 1 < len(numbered) else len(lines)
        number = ".".join(str(c) for c in counters if c > 0) or str(n + 1)
        out.append((number, " › ".join([doc.title, *trail]), _paragraphs(lines[idx + 1 : end])))
    return out


def chunk_document(doc: Document, target: int = 160, hard_max: int = 240) -> list[Chunk]:
    if doc.turns:
        return chunk_transcript(doc, target)
    raw: list[tuple[str, str, str]] = []
    for number, trail, paras in _sections(doc):
        packed = _pack(paras, target, hard_max)
        for part, text in enumerate(packed, start=1):
            if not number:
                section = str(part)
            elif len(packed) == 1:
                section = number
            else:
                section = f"{number} p{part}"
            raw.append((section, trail, text))
    return _finalise(doc, raw)


def chunk_transcript(doc: Document, target: int = 160) -> list[Chunk]:
    raw: list[tuple[str, str, str]] = []
    buf: list[tuple[str, float, float, str]] = []

    def flush() -> None:
        if not buf:
            return
        speakers = {t[0] for t in buf}
        start, end = buf[0][1], buf[-1][2]
        if len(speakers) == 1:
            section = f"{buf[0][0][:10].strip()} @{_clock(start)}"
        else:
            section = f"@{_clock(start)}-{_clock(end)}"
        heading = f"{doc.title} › {', '.join(sorted(speakers))} {_clock(start)}–{_clock(end)}"
        text = "\n".join(f"{s}: {t}" for s, _, _, t in buf)
        raw.append((section, heading, text))
        buf.clear()

    for turn in doc.turns:
        # Whole turns only: a turn is never split, even when it is long.
        if buf and _words(" ".join(t[3] for t in buf)) + _words(turn[3]) > target:
            flush()
        buf.append(turn)
    flush()
    return _finalise(doc, raw, kind="transcript")


def _finalise(doc: Document, raw: list[tuple[str, str, str]], kind: str | None = None) -> list[Chunk]:
    chunks = []
    used: set[str] = set()
    for i, (section, heading, text) in enumerate(raw):
        section = section.replace("]", ")").replace("\n", " ").strip()[:SECTION_MAX] or str(i + 1)
        candidate, k = section, 2
        while candidate.lower() in used:  # markers must resolve to exactly one chunk
            suffix = f"~{k}"
            candidate = section[: SECTION_MAX - len(suffix)] + suffix
            k += 1
        used.add(candidate.lower())
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}_{i}",
                doc_id=doc.doc_id,
                doc_label=doc.label,
                section=candidate,
                heading=heading,
                text=text.strip(),
                ordinal=i,
                kind=kind or doc.kind,
            )
        )
    return chunks
