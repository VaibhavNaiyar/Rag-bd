"""Transcript -> timestamped chunks.

Voice input is simulated from transcripts, as the theme allows. The grouping
matches the frontend's typed-input chunker exactly (3, 4, 5, 3, 4, 5 … words at
150 wpm), so a replayed fixture and a typed request reach the controller in
the same shape.

A fixture may instead give explicit ``chunks: [{text, atMs}]`` (``at_ms`` is also
accepted) — the format a real ASR partial stream would be recorded in.

Fixtures are data files under ``evals/fixtures/``; they are read by path at
runtime and never imported.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

WPM = 150
CHUNK_WORDS = (3, 4, 5)
_SAFE = re.compile(r"^[A-Za-z0-9_\-]{1,80}$")


def chunk_utterance(text: str, wpm: int = WPM) -> list[tuple[str, int]]:
    """[(chunk_text, delay_ms_before_chunk)]"""
    words = text.split()
    ms_per_word = 60_000 / wpm
    out = []
    i = g = 0
    while i < len(words):
        size = CHUNK_WORDS[g % len(CHUNK_WORDS)]
        piece = words[i : i + size]
        out.append((" ".join(piece), round(len(piece) * ms_per_word)))
        i += size
        g += 1
    return out


def timed_chunks(spec: dict[str, Any]) -> Iterator[tuple[str, int]]:
    if spec.get("chunks"):
        last = 0
        for c in spec["chunks"]:
            at = int(c["atMs"] if "atMs" in c else c["at_ms"])
            yield str(c["text"]), max(0, at - last)
            last = at
    else:
        yield from chunk_utterance(str(spec["utterance"]), int(spec.get("wpm", WPM)))


def utterance_of(spec: dict[str, Any]) -> str:
    if spec.get("utterance"):
        return str(spec["utterance"])
    return " ".join(str(c["text"]) for c in spec.get("chunks", []))


def list_fixtures(fixtures_dir: str) -> dict[str, Path]:
    root = Path(fixtures_dir)
    return {p.stem: p for p in sorted(root.rglob("*.json"))} if root.is_dir() else {}


def describe_fixtures(fixtures_dir: str, corpus: str) -> list[dict[str, Any]]:
    """What a client may replay against ``corpus``: id, family (its folder), description and what is said."""
    out = []
    for name, path in list_fixtures(fixtures_dir).items():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("corpus", "enterprise") != corpus or not data.get("turns"):
            continue
        out.append(
            {
                "id": name,
                "family": path.parent.name,
                "description": data.get("description", ""),
                "turns": [utterance_of(turn) for turn in data["turns"]],
            }
        )
    return out


def load_fixture(fixtures_dir: str, name: str) -> dict[str, Any]:
    if not _SAFE.match(name):
        raise ValueError(f"invalid fixture name {name!r}")
    found = list_fixtures(fixtures_dir).get(name)
    if found is None:
        raise FileNotFoundError(f"fixture {name!r} not found under {fixtures_dir}")
    data = json.loads(found.read_text(encoding="utf-8"))
    if "turns" not in data or not data["turns"]:
        raise ValueError(f"fixture {name!r} has no turns")
    return data
