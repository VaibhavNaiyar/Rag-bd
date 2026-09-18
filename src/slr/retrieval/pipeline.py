"""retrieve -> rerank -> (per sub-query) cut.

``search_many`` batches every sub-query's rerank pairs into one cross-encoder
call: on CPU that is the dominant cost of a turn, and one batch beats N
contending threads.
"""

from __future__ import annotations

import time

from slr.config import Settings
from slr.contracts import Hit, SubQuery, SubQueryResult
from slr.retrieval.hybrid import hybrid_search
from slr.retrieval.rerank import Reranker, apply_scores, margin_cut, safe_score
from slr.retrieval.store import Index, index_text


class Retriever:
    def __init__(self, index: Index, reranker: Reranker | None, settings: Settings):
        self.index = index
        self.reranker = reranker
        self.settings = settings

    def _candidates(self, sq: SubQuery, qvec) -> tuple[list[Hit], int, dict[str, int]]:
        s = self.settings
        fused, counts = hybrid_search(self.index, sq.text, qvec, s.branches, s.branch_k, s.rrf_k)
        top = fused[: s.rerank_cap]
        hits = [
            Hit(
                chunk=self.index.chunks[f.idx],
                score=f.rrf,
                branches=sorted(f.branches),
                sub_query_ids=[sq.id],
                rrf=f.rrf,
            )
            for f in top
        ]
        return hits, len(fused), counts

    def search_many(self, sub_queries: list[SubQuery], pair_budget: int | None = None) -> list[SubQueryResult]:
        if not sub_queries:
            return []
        started = time.perf_counter()
        s = self.settings
        qvecs = None
        if "dense" in s.branches:
            qvecs = self.index.embedder.embed([sq.text for sq in sub_queries], kind="query")
        staged = []
        pairs: list[tuple[str, str]] = []
        # Per-turn pair budget: a 4-intent turn costs the same rerank time as a 2-intent one.
        budget = pair_budget or s.rerank_pair_budget
        per_sq = max(s.min_keep * 3, budget // len(sub_queries))
        for i, sq in enumerate(sub_queries):
            hits, n, counts = self._candidates(sq, None if qvecs is None else qvecs[i])
            tail = hits[per_sq:]
            hits = hits[:per_sq]
            staged.append((sq, hits, tail, n, counts, len(pairs)))
            pairs.extend((sq.text, index_text(h.chunk)) for h in hits)

        scores = safe_score(self.reranker, pairs)
        elapsed = (time.perf_counter() - started) * 1000
        results = []
        for sq, hits, _tail, n, counts, offset in staged:
            if scores is not None:
                ranked = apply_scores(hits, scores[offset : offset + len(hits)])
                kept = margin_cut(ranked, s.rerank_margin, s.min_keep, s.per_query_keep)
            else:
                kept = hits[: max(s.min_keep, s.per_query_keep)]
            results.append(
                SubQueryResult(
                    sub_query=sq,
                    candidates=n,
                    kept=kept,
                    branch_counts=counts,
                    reranked=scores is not None,
                    elapsed_ms=elapsed,
                )
            )
        return results

    def search(self, sq: SubQuery) -> SubQueryResult:
        return self.search_many([sq])[0]

    def search_provisional(self, sq: SubQuery) -> SubQueryResult:
        return self.search_many([sq], self.settings.rerank_provisional_pairs)[0]
