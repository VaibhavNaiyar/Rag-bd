"""Few-shot examples for the decomposer, chosen per utterance by nearest neighbour.

A bank is a JSONL file of ``{"question": ..., "readings": [...]}``: questions whose
readings are known, for the domain the engine serves. For each utterance the
decomposer is shown the ``k`` most similar questions and how they split. This is
how Tree of Clarifications (Kim et al., EMNLP 2023) and DIVA (In et al., NAACL 2025)
prompt for the readings of ambiguous questions: nearest-neighbour examples from a
training set, not a fixed handful.

An example too close to the utterance itself (``max_cos``) is skipped, so a
question can never be shown its own twin with the answer attached.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from slr.retrieval.embed import Embedder

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Example:
    question: str
    readings: tuple[str, ...]


class ExampleBank:
    def __init__(self, examples: list[Example], vecs: np.ndarray, max_cos: float = 0.90):
        self.examples = examples
        self.vecs = vecs
        self.max_cos = max_cos

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder, max_cos: float = 0.90) -> ExampleBank | None:
        path = Path(path)
        if not path.is_file():
            log.warning("decomposer example bank %s not found; decomposing without examples", path)
            return None
        examples = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    examples.append(Example(row["question"], tuple(row["readings"])))
        if not examples:
            return None
        # Embedding a few thousand questions takes a while on a CPU; keep them beside the bank.
        cache = path.with_suffix(f".{_slug(getattr(embedder, 'name', 'embedder'))}.npy")
        vecs = np.load(cache) if cache.is_file() else None
        if vecs is None or vecs.shape[0] != len(examples):
            vecs = embedder.embed([e.question for e in examples], kind="query")
            try:
                np.save(cache, vecs)
            except OSError:
                pass
        return cls(examples, vecs, max_cos)

    def nearest(self, utterance: str, embedder: Embedder, k: int, k_single: int = 2) -> list[Example]:
        """The ``k`` nearest questions that split and the ``k_single`` nearest that did not.

        Both kinds are needed: shown only questions that split, the model splits
        everything; shown the nearest of all, an ambiguous question's nearest neighbours
        are its own specific readings, and it splits nothing. The contrast between a
        similar question that split and a similar one that did not is the lesson.
        """
        if k <= 0 and k_single <= 0:
            return []
        sims = self.vecs @ embedder.embed([utterance], kind="query")[0]
        split: list[tuple[float, Example]] = []
        single: list[tuple[float, Example]] = []
        for i in np.argsort(-sims):
            if sims[i] >= self.max_cos:
                continue  # the utterance itself, or a paraphrase of it
            ex = self.examples[int(i)]
            pool, cap = (single, k_single) if len(ex.readings) == 1 else (split, k)
            if len(pool) < cap:
                pool.append((float(sims[i]), ex))
            if len(split) >= k and len(single) >= k_single:
                break
        return [ex for _, ex in sorted([*split, *single], key=lambda p: -p[0])]


def format_examples(examples: list[Example]) -> str:
    if not examples:
        return "(none)"
    lines = []
    for ex in examples:
        lines.append(f'"{ex.question}"')
        if len(ex.readings) == 1 and ex.readings[0] == ex.question:
            lines.append("  - one reading: already specific, not split")
        else:
            lines.extend(f"  - {r}" for r in ex.readings)
    return "\n".join(lines)


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")[:60] or "embedder"
