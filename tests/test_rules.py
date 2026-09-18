"""The hard rules from the theme guide, enforced as tests.

1. Corpus isolation      — answers come from the index, nothing else.
2. No hardcoding         — prompts live in prompts/, fixtures under evals/, and
                           neither is imported by src/.
3. Grounding             — citations are built from chunk records.
4. Session-bound state   — no cross-session profile.
5. Architectural parsimony — no agent framework; two model calls per turn.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "slr"
SOURCES = sorted(SRC.rglob("*.py"))

BANNED_IMPORTS = {
    "langchain", "langchain_core", "langchain_openai", "langgraph", "llama_index", "llamaindex",
    "haystack", "crewai", "autogen", "deepagents", "semantic_kernel", "dspy", "guidance",
    "chromadb", "pinecone", "weaviate", "qdrant_client", "faiss",
}


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_no_agent_framework_is_imported():
    for path in SOURCES:
        offending = imported_modules(path) & BANNED_IMPORTS
        assert not offending, f"{path.relative_to(ROOT)} imports {offending}"


def test_src_never_imports_evals_or_tests():
    for path in SOURCES:
        names = imported_modules(path)
        assert "evals" not in names and "tests" not in names, path.relative_to(ROOT)


def test_prompts_are_files_not_string_literals():
    """A prompt in code cannot be reviewed or swapped by a judge."""
    prompts = {p.stem for p in (ROOT / "prompts").glob("*.md")}
    assert prompts >= {"decompose", "synthesise", "refine", "refine_plan", "restructure", "controller"}

    telltale = re.compile(r"(You are a helpful|You answer|Answer the question|Respond with JSON|Rules:)", re.I)
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # Docstrings explain the code; only other long literals could be prompts.
        docstrings = {
            id(n.body[0].value)
            for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and n.body
            and isinstance(n.body[0], ast.Expr)
            and isinstance(n.body[0].value, ast.Constant)
            and isinstance(n.body[0].value.value, str)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) > 300:
                if id(node) in docstrings:
                    continue
                assert not telltale.search(node.value), f"{path.relative_to(ROOT)} inlines a prompt"


def test_no_fixture_utterance_or_corpus_sentence_appears_in_src():
    """The held-out replay is private: nothing in src/ may be tuned to the dev data."""
    blob = "\n".join(p.read_text(encoding="utf-8") for p in SOURCES).lower()

    for fixture in (ROOT / "evals" / "fixtures").rglob("*.json"):
        data = json.loads(fixture.read_text(encoding="utf-8"))
        for turn in data["turns"]:
            utterance = turn.get("utterance", "")
            assert utterance.lower() not in blob, f"{fixture.name}: utterance is embedded in src/"
            for intent in turn.get("expect", {}).get("gold_sub_intents", []):
                assert intent.lower() not in blob, f"{fixture.name}: a gold sub-intent is embedded in src/"

    for doc in (ROOT / "evals" / "corpora").rglob("*.md"):
        for line in doc.read_text(encoding="utf-8").splitlines():
            line = line.strip().lower()
            if len(line.split()) >= 8:
                assert line not in blob, f"a corpus sentence from {doc.name} is embedded in src/"


def test_no_corpus_specific_vocabulary_in_the_controller():
    """The controller must work on a corpus it has never seen."""
    text = (SRC / "controller" / "signals.py").read_text(encoding="utf-8").lower()
    for word in ["pune", "riverside", "baner", "hinjewadi", "reimbursement", "catering", "venue", "workshop"]:
        assert word not in text, f"controller vocabulary mentions corpus content: {word!r}"


def test_citations_are_built_from_the_chunk_record_only():
    """No code path constructs a marker from model output."""
    chunker = (SRC / "ingest" / "chunker.py").read_text(encoding="utf-8")
    contracts = (SRC / "contracts.py").read_text(encoding="utf-8")
    assert 'f"[{self.doc_label} §{self.section}]"' in contracts
    assert "SECTION_MAX" in chunker

    grounding = (SRC / "synthesis" / "grounding.py").read_text(encoding="utf-8")
    # the grounder resolves markers against the retrieved set; it never trusts one
    assert "self._cmap.get(normalise_marker(doc, section))" in grounding
    assert "fabricated_blocked += 1" in grounding


def test_two_model_calls_per_turn_is_the_design():
    engine = (SRC / "stream" / "engine.py").read_text(encoding="utf-8")
    # decompose (or refine planning) + synthesis (or refine/restructure). Nothing else.
    steps = set(re.findall(r'step="(\w+)"', "\n".join(p.read_text(encoding="utf-8") for p in SOURCES)))
    assert steps <= {"decompose", "synthesise", "refine", "refine_plan", "restructure", "controller"}
    assert "while True" not in engine or "queue" in engine, "no open-ended agent loop"


@pytest.mark.parametrize("path", [p for p in SOURCES], ids=lambda p: str(p.relative_to(SRC)))
def test_sources_are_utf8_and_parse(path):
    source = path.read_bytes().decode("utf-8")
    ast.parse(source)


def test_settings_are_env_overridable():
    from slr.config import Settings

    import dataclasses

    names = {f.name for f in dataclasses.fields(Settings)} - {"extra"}
    for knob in ["branches", "controller", "quota_per_intent", "top_k", "tau_stab", "support_min"]:
        assert knob in names, f"{knob} must be tunable without a code change"
