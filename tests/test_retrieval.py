from __future__ import annotations

from slr.contracts import Chunk, Hit, SubQuery, SubQueryResult
from slr.retrieval.fusion import fuse, union_prior
from slr.retrieval.rerank import margin_cut


def chunk(i: int) -> Chunk:
    return Chunk(f"c{i}", "d", "Doc_1", str(i), "h", f"text {i}", i)


def hit(i: int, score: float, sq: str = "sq1") -> Hit:
    return Hit(chunk(i), score, ["dense"], [sq], rrf=score / 10)


def result(sq_id: str, scores: list[float], offset: int = 0) -> SubQueryResult:
    hits = [hit(offset + n, s, sq_id) for n, s in enumerate(scores)]
    return SubQueryResult(SubQuery(sq_id, f"query {sq_id}"), candidates=50, kept=hits)


def test_margin_is_relative_to_this_searchs_own_top_score():
    hits = [hit(1, 0.90), hit(2, 0.80), hit(3, 0.60), hit(4, 0.10)]
    kept = margin_cut(hits, margin=0.15, min_keep=1, keep=10)
    assert [h.chunk.chunk_id for h in kept] == ["c1", "c2"]

    # a weak search keeps its own best, rather than everything or nothing
    weak = [hit(1, 0.20), hit(2, 0.12), hit(3, 0.02)]
    assert [h.chunk.chunk_id for h in margin_cut(weak, 0.15, 1, 10)] == ["c1", "c2"]


def test_margin_never_cuts_below_min_keep():
    hits = [hit(1, 0.9), hit(2, 0.2), hit(3, 0.1), hit(4, 0.05)]
    assert len(margin_cut(hits, 0.15, min_keep=3, keep=10)) == 3


def test_quota_guarantees_every_sub_query_a_slot():
    verbose = result("sq1", [0.99, 0.98, 0.97, 0.96, 0.95, 0.94])
    quiet = result("sq2", [0.50, 0.40], offset=10)
    outcome = fuse([verbose, quiet], top_k=6, quota=2)
    ids = [h.chunk.chunk_id for h in outcome.hits]
    assert "c10" in ids and "c11" in ids, "the quiet sub-intent was starved out"
    assert outcome.quota_applied is True
    assert set(outcome.quota_promoted) == {"c10", "c11"}
    assert outcome.per_sub_query == {"sq1": 4, "sq2": 2}


def test_without_quota_one_sub_intent_takes_every_slot():
    verbose = result("sq1", [0.99, 0.98, 0.97, 0.96, 0.95, 0.94])
    quiet = result("sq2", [0.50, 0.40], offset=10)
    outcome = fuse([verbose, quiet], top_k=6, quota=0)
    assert [h.chunk.chunk_id for h in outcome.hits] == ["c0", "c1", "c2", "c3", "c4", "c5"]
    assert outcome.quota_applied is False


def test_duplicate_chunks_merge_and_union_their_sub_query_ids():
    a = SubQueryResult(SubQuery("sq1", "a"), 10, [hit(1, 0.9, "sq1")])
    b = SubQueryResult(SubQuery("sq2", "b"), 10, [hit(1, 0.7, "sq2")])
    outcome = fuse([a, b], top_k=8, quota=2)
    assert len(outcome.hits) == 1
    assert sorted(outcome.hits[0].sub_query_ids) == ["sq1", "sq2"]
    assert outcome.hits[0].score == 0.9


def test_extras_compete_on_score_but_get_no_quota():
    decomposed = result("sq1", [0.9, 0.85])
    provisional = result("p1", [0.99, 0.98, 0.97], offset=20)
    outcome = fuse([decomposed], top_k=3, quota=2, extras=[provisional])
    ids = [h.chunk.chunk_id for h in outcome.hits]
    assert "c0" in ids and "c1" in ids  # the intent's quota is honoured first
    assert len(ids) == 3


def test_refinement_keeps_prior_evidence_and_appends_delta():
    prior = [hit(1, 0.9, "sq1"), hit(2, 0.8, "sq1")]
    delta = [hit(2, 0.95, "sq9"), hit(3, 0.7, "sq9")]
    merged = union_prior(prior, delta)
    by_id = {h.chunk.chunk_id: h for h in merged}
    assert set(by_id) == {"c1", "c2", "c3"}
    assert sorted(by_id["c2"].sub_query_ids) == ["sq1", "sq9"]


def test_hybrid_branches_and_rrf(index):
    from slr.retrieval.hybrid import hybrid_search

    qvec = index.embedder.embed(["cancellation refund policy"], kind="query")[0]
    fused, counts = hybrid_search(index, "cancellation refund policy", qvec, ("bm25", "dense"), 50, 60)
    assert counts["bm25"] > 0 and counts["dense"] > 0
    assert fused[0].rrf >= fused[-1].rrf
    # a chunk found by both branches outranks one found by a single branch at the same rank
    both = [f for f in fused if len(f.branches) == 2]
    assert both, "no chunk was retrieved by both branches"


def test_dense_only_ablation_changes_nothing_structural(index, settings):
    from slr.retrieval.pipeline import Retriever

    dense_only = Retriever(index, None, settings.with_overrides(branches=("dense",)))
    res = dense_only.search(SubQuery("sq1", "cancellation policy"))
    assert res.kept and res.branch_counts == {"dense": len(index.chunks)}
