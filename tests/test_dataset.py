"""The generated ASQA fixtures and transcripts agree with each other and with the replay clock.

G2 compares first-retrieval time with ``utterance_end_ms``. If a fixture's
chunks drifted from its transcript, or the recorded end did not match what
the simulator will actually play, the gate would be measured against the
wrong line.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slr.stream.simulator import timed_chunks

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "evals" / "fixtures"
FAMILIES = ("compound", "single", "late_detail", "suppression")

CASES = sorted(p for fam in FAMILIES for p in (FIXTURES / fam).glob("asqa_*.json"))


def test_every_family_has_asqa_cases():
    for family in FAMILIES:
        assert list((FIXTURES / family).glob("asqa_*.json")), f"no ASQA fixtures in {family}/"


@pytest.mark.parametrize("path", CASES, ids=[p.stem for p in CASES])
def test_fixture_matches_transcript_and_clock(path: Path):
    fixture = json.loads(path.read_text(encoding="utf-8"))
    transcript = json.loads((ROOT / fixture["transcript"]).read_text(encoding="utf-8"))
    assert fixture["corpus"] == "asqa"
    assert len(fixture["turns"]) == len(transcript["turns"])

    for turn, spoken in zip(fixture["turns"], transcript["turns"]):
        assert turn["chunks"] == spoken["chunks"]
        assert " ".join(c["text"] for c in turn["chunks"]) == turn["utterance"]
        assert all(3 <= len(c["text"].split()) <= 5 for c in turn["chunks"][:-1])

        # 150 wpm: a chunk lands when its last word has been spoken.
        words = 0
        for chunk in turn["chunks"]:
            words += len(chunk["text"].split())
            assert chunk["atMs"] == words * 400

        # What the simulator will play adds up to the recorded utterance end.
        played = sum(delay for _, delay in timed_chunks(turn)) + turn["end_pause_ms"]
        assert played == turn["utterance_end_ms"] == spoken["utterance_end_ms"]


def test_family_expectations():
    for path in CASES:
        fixture = json.loads(path.read_text(encoding="utf-8"))
        family = path.parent.name
        expects = [t["expect"] for t in fixture["turns"]]
        if family == "compound":
            assert len(expects[0]["gold_sub_intents"]) >= 2
        elif family == "single":
            assert len(expects[0]["gold_sub_intents"]) == 1
        elif family == "late_detail":
            assert [e["mode"] for e in expects] == ["retrieve", "refine"]
            assert expects[1]["gold_passages"]
        elif family == "suppression":
            assert [e["mode"] for e in expects] == ["retrieve", "suppress"]
