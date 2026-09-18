"""Coverage-guaranteed fusion across sub-queries.

1. Every sub-query gets its top ``quota`` chunks a slot.
2. Remaining slots, up to ``top_k``, go by global score.
3. Chunks are deduplicated by id and their sub-query ids unioned.

Without step 1 one verbose sub-intent can take every slot and the answer
silently drops a question the user asked. Aggregate recall cannot see that
failure, which is why the quota has its own ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from slr.contracts import Hit, SubQueryResult


@dataclass
class FusionOutcome:
    hits: list[Hit]
    quota_applied: bool
    #: chunks that are in the final set only because of the quota
    quota_promoted: list[str] = field(default_factory=list)
    per_sub_query: dict[str, int] = field(default_factory=dict)


def _merge(into: dict[str, Hit], hit: Hit, sq_id: str) -> None:
    cid = hit.chunk.chunk_id
    existing = into.get(cid)
    if existing is None:
        into[cid] = Hit(hit.chunk, hit.score, list(hit.branches), [sq_id], hit.rrf)
        return
    if sq_id not in existing.sub_query_ids:
        existing.sub_query_ids.append(sq_id)
    for b in hit.branches:
        if b not in existing.branches:
            existing.branches.append(b)
    existing.score = max(existing.score, hit.score)


def fuse(
    results: list[SubQueryResult], top_k: int, quota: int, extras: list[SubQueryResult] | None = None
) -> FusionOutcome:
    """``extras`` compete for slots by score but get no quota (unreused provisional guesses)."""
    extras = extras or []
    pool: dict[str, Hit] = {}
    for res in [*results, *extras]:
        for hit in res.kept:
            _merge(pool, hit, res.sub_query.id)

    ranked = sorted(pool.values(), key=lambda h: (-h.score, -h.rrf, h.chunk.chunk_id))
    global_only = {h.chunk.chunk_id for h in ranked[:top_k]}

    chosen: list[str] = []
    if quota > 0:
        for res in results:
            for hit in res.kept[:quota]:
                if hit.chunk.chunk_id not in chosen:
                    chosen.append(hit.chunk.chunk_id)
    # A quota larger than the budget still has to respect it.
    chosen = chosen[: max(top_k, 0)] if len(chosen) > top_k else chosen
    for hit in ranked:
        if len(chosen) >= top_k:
            break
        if hit.chunk.chunk_id not in chosen:
            chosen.append(hit.chunk.chunk_id)

    hits = sorted((pool[c] for c in chosen), key=lambda h: (-h.score, h.chunk.chunk_id))
    promoted = [c for c in chosen if c not in global_only]
    per_sq = {r.sub_query.id: sum(r.sub_query.id in h.sub_query_ids for h in hits) for r in [*results, *extras]}
    return FusionOutcome(
        hits=hits,
        quota_applied=quota > 0 and len(results) > 1,
        quota_promoted=promoted,
        per_sub_query=per_sq,
    )


def union_prior(prior: list[Hit], fresh: list[Hit]) -> list[Hit]:
    """Refinement: prior evidence stays, delta evidence is appended."""
    pool: dict[str, Hit] = {}
    for h in prior:
        pool[h.chunk.chunk_id] = Hit(h.chunk, h.score, list(h.branches), list(h.sub_query_ids), h.rrf)
    for h in fresh:
        for sq in h.sub_query_ids:
            _merge(pool, h, sq)
    return list(pool.values())
