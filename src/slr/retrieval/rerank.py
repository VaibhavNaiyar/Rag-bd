"""Cross-encoder rerank with an adaptive margin cut.

Pattern borrowed from Lattice's ``doc/_search.py``:

* Rerank the fused top ``rerank_cap`` against *the sub-query that retrieved
  them*, not the whole utterance.
* Drop candidates more than ``margin`` below *this search's own* top score —
  a relative cut, because absolute relevance varies wildly between queries.
* Never keep fewer than ``min_keep``.
* On any reranker failure, degrade to fused order: a weaker ranking must never
  become no ranking.
"""

from __future__ import annotations

import logging
import math
import threading
from functools import lru_cache
from typing import Protocol

from slr.contracts import Hit

log = logging.getLogger(__name__)


class Reranker(Protocol):
    name: str

    def score(self, pairs: list[tuple[str, str]]) -> list[float]: ...


def _squash(x: float) -> float:
    """Map raw logits into [0, 1]; models that already emit probabilities pass through."""
    if 0.0 <= x <= 1.0:
        return x
    return 1.0 / (1.0 + math.exp(-x))


class CrossEncoderReranker:
    def __init__(self, model_name: str, max_length: int = 256, temperature: float = 4.0):
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self._model = CrossEncoder(model_name, max_length=max_length, device="cpu")
        self._temperature = temperature
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str], float] = {}
        # Some models already apply a sigmoid, some emit logits. Probe once.
        probe = self._model.predict([("probe", "probe text")], show_progress_bar=False)
        self._raw_logits = not (0.0 <= float(probe[0]) <= 1.0)

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        out: list[float | None] = [self._cache.get(p) for p in pairs]
        todo = [i for i, v in enumerate(out) if v is None]
        if todo:
            with self._lock:
                raw = predict_by_length(self._model, [pairs[i] for i in todo])
            for i, r in zip(todo, raw):
                value = float(r)
                value = 1.0 / (1.0 + math.exp(-value / self._temperature)) if self._raw_logits else _squash(value)
                out[i] = value
                if len(self._cache) < 50000:
                    self._cache[pairs[i]] = value
        return [float(v) for v in out]  # type: ignore[arg-type]


def predict_by_length(model, pairs: list[tuple[str, str]], batch_size: int = 16, **kwargs):
    """``model.predict`` with pairs batched by length, results back in the caller's order.

    A batch is padded to its longest pair, so one long passage in a batch of short ones
    makes every pair in it pay for the long one. Sorting first changes no score (each pair
    is still scored alone, with its own attention mask) and halved rerank time on a
    4-core laptop CPU: 192 pairs, 14.9 s to 7.1 s.
    """
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
    scored = model.predict([pairs[i] for i in order], batch_size=batch_size, show_progress_bar=False, **kwargs)
    out = [None] * len(pairs)
    for rank, i in enumerate(order):
        out[i] = scored[rank]
    return out


@lru_cache(maxsize=2)
def load_reranker(model_name: str, max_length: int, temperature: float = 4.0) -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name, max_length, temperature)


def margin_cut(hits: list[Hit], margin: float, min_keep: int, keep: int) -> list[Hit]:
    """Hits must already be sorted by score, descending."""
    if not hits:
        return []
    top = hits[0].score
    kept = [h for h in hits if h.score >= top - margin]
    if len(kept) < min_keep:
        kept = hits[:min_keep]
    return kept[:keep]


def apply_scores(hits: list[Hit], scores: list[float]) -> list[Hit]:
    for hit, s in zip(hits, scores):
        hit.score = s
    return sorted(hits, key=lambda h: (-h.score, -h.rrf))


def safe_score(reranker: Reranker | None, pairs: list[tuple[str, str]]) -> list[float] | None:
    if reranker is None or not pairs:
        return None
    try:
        return reranker.score(pairs)
    except Exception as exc:  # degrade, never fail the turn
        log.warning("reranker failed (%s); keeping fused order", exc)
        return None
