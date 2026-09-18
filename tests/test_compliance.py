"""Everything under evals/ is test data; src/ must not depend on it or contain it.

* No module under src/ imports from evals/.
* No gold answer string from evals/gold.jsonl appears anywhere under src/,
  including the built console bundle. That would mean an answer had been
  written into the engine instead of retrieved.

Answers are matched as whole-word sequences, case-insensitively. Only answers
with at least two content words are checked. Single tokens such as "1908",
"frame" or "Speaker" occur in ordinary code and UI text, so they would show up
without meaning anything, while a phrase like "Christine Sinclair" or "Mann
Plaza Theater" in src/ would be a real leak.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
GOLD = ROOT / "evals" / "gold.jsonl"

_BINARY = {".woff", ".woff2", ".ttf", ".otf", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".pyc"}
_STOP = {
    "the", "and", "for", "with", "from", "that", "this", "was", "were", "are", "his", "her",
    "its", "their", "all", "not", "but", "one", "two", "who", "has", "had", "have", "into",
    "than", "then", "after", "before", "over", "under", "about", "against", "between", "only",
}  # fmt: skip


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _content(tokens: tuple[str, ...]) -> int:
    return sum(1 for t in tokens if t.isalpha() and len(t) >= 3 and t not in _STOP)


def _src_files() -> list[Path]:
    return [
        p
        for p in SRC.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix.lower() not in _BINARY
    ]


def test_src_never_imports_evals():
    offenders = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            else:
                continue
            if any(n == "evals" or n.startswith("evals.") for n in names):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, f"src/ imports test data from evals/: {offenders}"


@pytest.mark.skipif(not GOLD.exists(), reason="evals/gold.jsonl not built (make dataset)")
def test_no_gold_answer_string_in_src():
    answers: set[tuple[str, ...]] = set()
    with GOLD.open(encoding="utf-8") as fh:
        for line in fh:
            for sub in json.loads(line)["sub_questions"]:
                for answer in sub["short_answers"]:
                    tokens = tuple(_tokens(answer))
                    if _content(tokens) >= 2:
                        answers.add(tokens)
    # Guard against the check silently checking nothing.
    assert len(answers) > 1000, f"only {len(answers)} checkable gold answers; is gold.jsonl complete?"

    longest = max(len(a) for a in answers)
    seen: dict[tuple[str, ...], Path] = {}
    for path in _src_files():
        try:
            tokens = _tokens(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue
        for n in range(2, longest + 1):
            for i in range(len(tokens) - n + 1):
                seen.setdefault(tuple(tokens[i : i + n]), path)

    leaks = sorted(f"{' '.join(a)!r} in {seen[a].relative_to(ROOT)}" for a in answers if a in seen)
    assert not leaks, f"gold answers found in src/: {leaks[:20]}"
