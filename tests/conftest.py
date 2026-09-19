"""Test fixtures.

The suite runs the real pipeline, but with the cheap arms selected (LSA
embedder, no cross-encoder, lexical verifier, no LLM) so it needs no network,
no GPU and no API key. The model-backed paths are covered by ``FakeChatModel``,
which scripts exactly what a provider would return.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

ENTERPRISE_CORPUS = ROOT / "evals" / "corpora" / "enterprise"


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("slr")
    os.environ.update(
        {
            "SLR_EMBEDDER": "lsa",
            "SLR_RERANKER": "none",
            "SLR_VERIFIER": "lexical",
            "SLR_LLM": "offline",
            "SLR_CORPUS_DIR": str(ENTERPRISE_CORPUS),
            "SLR_INDEX_DIR": str(tmp / "index"),
            "SLR_TRACE_PATH": str(tmp / "trace.jsonl"),
            "SLR_CONTROLLER": "rule",
        }
    )
    from slr.config import get_settings, reset_settings

    reset_settings()
    return get_settings()


@pytest.fixture(scope="session")
def index(settings):
    from slr.ingest.build_index import build
    from slr.retrieval.store import load_index

    build(settings.corpus_dir, settings.index_dir, "lsa")
    return load_index(settings.index_dir)


@pytest.fixture(scope="session")
def engine(settings, index):
    from slr.stream.engine import Engine

    return Engine.from_settings(settings, index)


@pytest.fixture
def llm(engine, settings):
    """Engine clone whose LLM is scripted per test (assign ``.model``)."""
    return engine.with_settings(settings.with_overrides(llm="openai"))


class FakeChatModel:
    """Scripted provider. `completions` and `streams` are consumed in order.

    A callable entry is handed the rendered user prompt, so a test can build a
    reply out of the evidence the engine actually retrieved instead of guessing
    chunk ids.
    """

    def __init__(self, completions: list | None = None, streams: list | None = None):
        self.name = "fake-model"
        self.completions = list(completions or [])
        self.streams = list(streams or [])
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system, user, *, ledger, step, json_mode=False, max_tokens=600) -> str:
        self.calls.append((step, user))
        ledger.record_llm(step, self.name, 100, 40, 12.0)
        if not self.completions:
            raise AssertionError(f"FakeChatModel: no scripted completion for step {step!r}")
        reply = self.completions.pop(0)
        return reply(user) if callable(reply) else reply

    async def stream(self, system, user, *, ledger, step, max_tokens=900) -> AsyncIterator[str]:
        self.calls.append((step, user))
        if not self.streams:
            raise AssertionError(f"FakeChatModel: no scripted stream for step {step!r}")
        text = self.streams.pop(0)
        text = text(user) if callable(text) else text
        ledger.record_llm(step, self.name, 300, 120, 30.0)
        for i in range(0, len(text), 7):  # arrive in provider-sized deltas
            yield text[i : i + 7]
            await asyncio.sleep(0)


class Recorder:
    """Collects emitted wire events."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    async def __call__(self, event: dict) -> None:
        self.events.append(event)

    def of(self, kind: str) -> list[dict]:
        return [e for e in self.events if e["type"] == kind]

    def one(self, kind: str) -> dict:
        found = self.of(kind)
        assert found, f"no {kind} event emitted"
        return found[-1]

    def answer(self, turn_id: str | None = None) -> str:
        return "".join(
            e["text"] for e in self.events if e["type"] == "answer.token" and (turn_id is None or e["turnId"] == turn_id)
        )


def markers_in(prompt: str) -> list[str]:
    """Citation markers the engine put in front of the model, in order."""
    import re

    seen: list[str] = []
    for m in re.finditer(r"\[[A-Za-z0-9_.:-]+ §[^\]\n]{1,24}\]", prompt):
        if m.group(0) not in seen:
            seen.append(m.group(0))
    return seen


def blocks_in(prompt: str) -> list[tuple[str, str]]:
    """The evidence blocks as the engine rendered them: [(marker, body), ...].

    Scripted answers are built from these, so a fake reply quotes real retrieved
    text and cites the marker that block was labelled with — exactly what the
    synthesise prompt asks a real model to do.
    """
    import re

    block = re.compile(
        r"^(\[[A-Za-z0-9_.:-]+ §[^\]\n]{1,24}\])\n"
        r"<document[^>\n]*>\n(?:source:[^\n]*\n)?(.*?)\n</document>",
        re.S | re.M,
    )
    return [(m.group(1), " ".join(m.group(2).split())) for m in block.finditer(prompt)]


def quote(prompt: str, index: int = 0, words: int = 18) -> tuple[str, str]:
    """(sentence, marker) taken verbatim from one evidence block."""
    marker, body = blocks_in(prompt)[index]
    first = body.split(". ")[0]
    return " ".join(first.split()[:words]).rstrip(".,"), marker


@pytest.fixture
def recorder():
    return Recorder()


@pytest.fixture
def runner(engine, recorder):
    from slr.stream.engine import SessionRunner

    return SessionRunner(engine, recorder)
