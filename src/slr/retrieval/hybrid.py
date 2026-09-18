"""BM25 + dense branches, fused with reciprocal rank fusion.

    score(d) = Σ_branch 1 / (k + rank_branch(d)),  k = 60

RRF needs no score calibration between branches, which is why it is used
rather than a weighted sum: BM25 and cosine scores are not on one scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slr.retrieval.store import Index
from slr.text import content_tokens


@dataclass
class Fused:
    idx: int
    rrf: float
    branches: list[str]


def bm25_search(index: Index, query: str, k: int) -> list[int]:
    toks = content_tokens(query)
    if not toks:
        return []
    scores = index.bm25.get_scores(toks)
    order = np.argsort(-scores)[:k]
    return [int(i) for i in order if scores[i] > 0]


def dense_search(index: Index, qvec: np.ndarray, k: int) -> list[int]:
    sims = index.embeddings @ qvec
    k = min(k, len(sims))
    top = np.argpartition(-sims, k - 1)[:k] if k < len(sims) else np.arange(len(sims))
    return [int(i) for i in top[np.argsort(-sims[top])]]


def hybrid_search(
    index: Index,
    query: str,
    qvec: np.ndarray | None,
    branches: tuple[str, ...],
    k: int,
    rrf_k: int,
) -> tuple[list[Fused], dict[str, int]]:
    rankings: dict[str, list[int]] = {}
    if "bm25" in branches:
        rankings["bm25"] = bm25_search(index, query, k)
    if "dense" in branches and qvec is not None:
        rankings["dense"] = dense_search(index, qvec, k)

    fused: dict[int, Fused] = {}
    for branch, ranked in rankings.items():
        for rank, idx in enumerate(ranked, start=1):
            entry = fused.setdefault(idx, Fused(idx, 0.0, []))
            entry.rrf += 1.0 / (rrf_k + rank)
            entry.branches.append(branch)
    ordered = sorted(fused.values(), key=lambda f: (-f.rrf, f.idx))
    return ordered, {b: len(r) for b, r in rankings.items()}
