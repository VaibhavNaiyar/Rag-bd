"""Corpus directory -> Documents.

Corpus-agnostic by contract: the graders' corpus is held out, so nothing here
assumes a filename convention, a schema or a domain. Supported: txt, md, json,
jsonl, pdf. Transcripts are detected by shape (speaker-prefixed lines), not by
name.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from slr.contracts import Document

log = logging.getLogger(__name__)

SUPPORTED = {".txt", ".md", ".markdown", ".json", ".jsonl", ".pdf"}

# "[00:05:02] Priya: ..."  "00:05 Priya: ..."  "Priya (00:05:02): ..."  "PRIYA: ..."
_TS = r"(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?"
_TURN_LINE = re.compile(
    rf"^\s*(?:\[?(?P<ts1>{_TS})\]?\s*[-–]?\s*)?"
    rf"(?P<speaker>[A-Z][\w .'\-]{{0,30}}?)\s*(?:\((?P<ts2>{_TS})\))?\s*:\s+(?P<text>\S.*)$"
)
_TEXT_KEYS = ("text", "content", "body", "transcript", "passage", "document", "answer")
_TITLE_KEYS = ("title", "name", "heading", "subject", "wikipage", "id")
_TURNS_KEYS = ("turns", "utterances", "segments", "dialogue", "messages")


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _seconds(ts: str | None) -> float | None:
    if not ts:
        return None
    parts = [float(p) for p in ts.split(":")]
    total = 0.0
    for p in parts:
        total = total * 60 + p
    return total


def parse_transcript(text: str) -> list[tuple[str, float, float, str]] | None:
    """Speaker turns, or None when the text does not look like a transcript."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return None
    turns: list[list[Any]] = []
    matched = 0
    for line in lines:
        m = _TURN_LINE.match(line)
        # A markdown heading or bullet is never a speaker turn.
        if m and not line.lstrip().startswith(("#", "-", "*")):
            matched += 1
            ts = _seconds(m.group("ts1") or m.group("ts2"))
            turns.append([m.group("speaker").strip(), ts, None, m.group("text").strip()])
        elif turns:
            turns[-1][3] += " " + line.strip()
    if matched / len(lines) < 0.6:
        return None
    # Fill missing timestamps from word counts at ~150 wpm; set end = next start.
    clock = 0.0
    for turn in turns:
        if turn[1] is None:
            turn[1] = clock
        clock = turn[1] + len(turn[3].split()) * 0.4
        turn[2] = clock
    for cur, nxt in zip(turns, turns[1:]):
        if nxt[1] >= cur[1]:
            cur[2] = nxt[1]
    return [(s, float(a), float(b), t) for s, a, b, t in turns]


def _record_text(record: Any) -> tuple[str, str, list | None]:
    """(title, text, turns) from an arbitrary JSON record."""
    if isinstance(record, str):
        return "", record, None
    if not isinstance(record, dict):
        return "", json.dumps(record, ensure_ascii=False), None
    title = next((str(record[k]) for k in _TITLE_KEYS if isinstance(record.get(k), (str, int))), "")
    for key in _TURNS_KEYS:
        value = record.get(key)
        if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            turns = []
            for i, v in enumerate(value):
                speaker = str(v.get("speaker") or v.get("role") or v.get("name") or f"S{i}")
                body = str(next((v[k] for k in _TEXT_KEYS if k in v), ""))
                start = v.get("start", v.get("start_s", v.get("timestamp")))
                end = v.get("end", v.get("end_s"))
                turns.append((speaker, start, end, body))
            return title, "\n".join(f"{s}: {b}" for s, _, _, b in turns), turns
    for key in _TEXT_KEYS:
        if isinstance(record.get(key), str) and record[key].strip():
            return title, record[key], None
    # Unknown shape: keep every string value, so nothing is silently dropped.
    strings = [f"{k}: {v}" for k, v in record.items() if isinstance(v, str) and v.strip()]
    return title, "\n".join(strings), None


def _norm_turns(turns: list) -> tuple[tuple[str, float, float, str], ...]:
    out = []
    clock = 0.0
    for speaker, start, end, body in turns:
        start_s = _seconds(start) if isinstance(start, str) else (float(start) if start is not None else clock)
        end_s = _seconds(end) if isinstance(end, str) else (float(end) if end is not None else None)
        if end_s is None:
            end_s = start_s + len(str(body).split()) * 0.4
        clock = end_s
        out.append((speaker, float(start_s), float(end_s), str(body)))
    return tuple(out)


def _iter_raw(path: Path, root: Path) -> Iterator[tuple[str, str, bytes, str, list | None]]:
    """(title, text, bytes_for_hash, source, turns)"""
    rel = path.relative_to(root).as_posix()
    suffix = path.suffix.lower()
    data = path.read_bytes()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "") for page in reader.pages]
        # Page markers survive into the chunker, which turns them into sections.
        text = "\n".join(f"\f[[page {i + 1}]]\n{p}" for i, p in enumerate(pages))
        yield path.stem, text, data, rel, None
    elif suffix == ".jsonl":
        for n, line in enumerate(data.decode("utf-8", errors="replace").splitlines()):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                log.warning("skip malformed line %s:%d", rel, n + 1)
                continue
            title, text, turns = _record_text(record)
            yield title or f"{path.stem} #{n + 1}", text, line.encode(), f"{rel}#{n + 1}", turns
    elif suffix == ".json":
        payload = json.loads(data.decode("utf-8", errors="replace"))
        records = payload if isinstance(payload, list) else [payload]
        for n, record in enumerate(records):
            title, text, turns = _record_text(record)
            blob = json.dumps(record, sort_keys=True, ensure_ascii=False).encode()
            source = rel if len(records) == 1 else f"{rel}#{n + 1}"
            yield title or path.stem, text, blob, source, turns
    else:
        text = data.decode("utf-8", errors="replace")
        heading = next((ln.lstrip("# ").strip() for ln in text.splitlines() if ln.startswith("# ")), "")
        yield heading or path.stem, text, data, rel, None


def load_corpus(corpus_dir: str | Path) -> list[Document]:
    root = Path(corpus_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {root}")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    docs: list[Document] = []
    seen: set[str] = set()
    for path in files:
        try:
            for title, text, blob, source, turns in _iter_raw(path, root):
                if not text.strip():
                    continue
                doc_id = _hash(blob)
                if doc_id in seen:  # identical content ingested once
                    continue
                seen.add(doc_id)
                parsed_turns: tuple = ()
                if turns:
                    parsed_turns = _norm_turns(turns)
                elif path.suffix.lower() in {".txt", ".md", ".markdown"}:
                    found = parse_transcript(text)
                    parsed_turns = tuple(found) if found else ()
                kind = "transcript" if parsed_turns else ("pdf" if path.suffix.lower() == ".pdf" else "document")
                docs.append(
                    Document(
                        doc_id=doc_id,
                        label=f"Doc_{len(docs) + 1}",
                        title=title.strip()[:120],
                        source=source,
                        kind=kind,
                        text=text,
                        turns=parsed_turns,
                    )
                )
        except Exception as exc:  # one bad file must not sink the ingest
            log.warning("skip %s: %s", path, exc)
    return docs
