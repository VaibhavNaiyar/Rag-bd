# System Architecture Brief

**Streaming Live RAG — Samsung PRISM GenAI Hackathon, Theme 04**

---

## 1. The problem, stated as an engineering constraint

A batch RAG turn is: wait for the full question, embed it, search, synthesise, answer. In a
full-duplex conversation that is wrong in three separate ways.

1. **Latency is structural, not incidental.** Waiting for the speaker to finish adds the
   length of their sentence to every answer. Retrieval cannot start at the end of the
   utterance if the answer is meant to feel immediate.
2. **One utterance is not one query.** "I need a venue in Pune for thirty people, and the
   cancellation policy, and the catering options" is three searches. Embedding it as one
   query returns the blurry average of three topics.
3. **A conversation edits itself.** "Actually the trip was international" does not restate
   the question; it narrows the answer that already exists. Re-running the pipeline throws
   away work that is still correct and produces a disjoint reply.

So the engine is built as four stages over a transcript stream, not as a function of a
question string. Each stage exists because one of those three constraints demands it.

---

## 2. Stage 1 — the retrieval controller

The controller consumes transcript chunks (3–5 words, ~150 wpm) and emits one decision per
chunk: `wait`, `retrieve`, `suppress` or `refine`. It runs four signals, no model call, in
about 20 ms per chunk (dominated by one embedding of the growing prefix).

**Semantic stability — the primary trigger.** The prefix is embedded on every chunk. When
`cos(e(p_t), e(p_{t-1})) ≥ 0.95` holds for two consecutive chunks, the *meaning* has stopped
moving and the utterance is safe to search on. This detects a semantic boundary rather than
guessing from a pause or a word count, which is what makes it work on an unpunctuated ASR
stream.

**Clause completion — the secondary trigger.** Speech does not always stabilise before it
ends. A closed clause (sentence punctuation or a comma) with enough content behind it also
fires a search. In practice this is what catches short requests, where the utterance ends
before three chunks exist.

**Content sufficiency — the gate on both.** At least four content-bearing tokens and at
least one *entity*, where an entity is a number, a capitalised non-initial word, or a token
the index itself marks as salient (document frequency ≤ 25% of chunks). Corpus-grounded
rather than NER-based: no extra model, and it transfers to a corpus the engine has never
seen. This is what stops "I need to plan a customer…" from searching on nothing.

**Suppression.** A presentation-only turn ("repeat that in two bullets", "make it shorter")
must not touch the retriever. It requires all three of: a presentation phrase, an anaphoric
reference to prior output, and no new content — where "new content" is measured on the text
*outside* the presentation phrase, so the verb in "make that shorter" is not mistaken for a
topic. Result: zero vector queries, prior citations retained.

**Refinement.** A late constraint is routed to the refinement path rather than a fresh
decomposition. Three bars by sentence form: a modifier cue on a statement needs only weak
similarity to the previous request (a constraint often shares little vocabulary with what it
narrows), a cue on a question needs more, and a bare new question is never a refinement
however similar it looks.

**Speculative retrieval with cancellation.** On `retrieve`, the search runs as an asyncio
task. If a later chunk moves the topic away — measured on the words spoken *since* the
trigger, not on the whole prefix, which dilutes a topic change until it disappears — the
task is cancelled and `retrieval_cancelled` is logged. The thrash guard is demonstrable in
the trace rather than asserted.

A clause that already carries two needs ("the cancellation policy and the catering options")
is split, without a model call, into two provisional searches. Both start before the speaker
finishes instead of one blended query that serves neither.

---

## 3. Stage 2 — multi-intent decomposition

One model call, strict JSON out: up to four sub-queries, each retrieval-shaped rather than
grammar-corrected, each carrying the shared context it needs ("Pune", "30 people") to stand
alone. `span` records which part of the utterance produced it, so a claim can be traced back
to the part of the request it answers.

**The over-fragmentation guard** (guide pitfall #5) caps at four, merges any pair above 0.90
cosine, and requires a single-intent utterance to return exactly one sub-query. Splitting
"what is the cancellation policy" into three near-identical variants pollutes the reranker
and burns tokens for nothing.

That guard has a real cost, and the benchmark measures it rather than hiding it: ASQA's
*ambiguous* questions are disambiguated into sub-questions that differ by one qualifier
("men's" vs "women's"), which sit above the merge threshold and collapse into one query. The
report scores compound utterances and ambiguous ones separately for exactly this reason.

**The provisional query is reused, not discarded.** A decomposed sub-query that matches a
provisional guess above 0.85 cosine takes its results directly; the rest are searched fresh,
and both run concurrently. Reuse is decided on the *queries*, which are known immediately, so
the fresh searches never wait behind the speculative ones. That overlap is what converts
early retrieval into a latency win instead of a wasted call.

Without an API key a deterministic clause splitter runs instead, and the trace records which
path ran (`decomposition.method`).

---

## 4. Stage 3 — retrieval and fusion

Per sub-query, in parallel:

1. **BM25** top 50 and **dense** top 50 (bge-small-en-v1.5, in-process numpy dot product —
   at this corpus size an ANN index would add a service and buy nothing).
2. **RRF fusion**, `1/(60 + rank)`. RRF needs no score calibration between branches, which a
   weighted sum would: BM25 scores and cosines are not on one scale.
3. **Cross-encoder rerank** of the fused top 20, scored **against that sub-query** — a
   candidate was retrieved for one sub-intent and should be judged against it, not against
   the whole utterance.
4. **Adaptive margin cut**: drop anything more than 0.15 below *this search's own* top score,
   never below `min_keep = 3`. A relative cut, because absolute relevance varies wildly
   between queries. Any reranker failure degrades to fused order — a weaker ranking must
   never become no ranking.

Then **coverage-guaranteed fusion** across sub-queries: every sub-query is guaranteed its top
2 chunks a slot, remaining slots to `top_k = 8` go by global score, duplicates merge and
union their sub-query ids.

That quota is the difference between an answer that addresses every question asked and one
that silently drops the quietest. Aggregate recall cannot see that failure, which is why it
has its own ablation arm and its own metric (*intents with evidence*).

**Latency, honestly.** Cross-encoder reranking dominates a CPU turn: `bge-reranker-base`
costs ~5.5 s per 30 pairs on the development machine, so the default is
`ms-marco-MiniLM-L-6-v2` at ~0.7 s, with the larger model one environment variable away for a
GPU host. A per-turn *pair budget* bounds the cost so a four-intent turn reranks no longer
than a two-intent one, and provisional searches get a smaller budget than real ones because
they are guesses.

---

## 5. Stage 4 — grounded synthesis, and refinement

**Generation is streamed, but the stream is released one validated sentence at a time.** For
each completed sentence:

1. Every citation marker is matched against the retrieved set. A marker that does not resolve
   is stripped and counted as a blocked fabrication. The shipped answer therefore carries zero
   fabricated citations *by construction*, not by asking the model nicely.
2. A sentence with no resolvable marker is attributed to the best-supporting retrieved chunk —
   but only above a higher bar (0.75) than validating a citation the model supplied, because
   choosing a source on the model's behalf is the stronger claim.
3. Support is scored by an NLI cross-encoder between the claim and its cited chunk. Below 0.5
   the claim is withheld and reported as uncertainty rather than shipped as fact.

Time-to-first-token is therefore time to first *verified* token, and that is what the report
measures.

One measured detail shaped this design: the NLI model scores a claim at 0.99 against the
single sentence that entails it, and 0.02 against that same sentence with one unrelated
sentence appended. Support is therefore scored over 1- and 2-sentence windows of the chunk,
not the whole chunk, with a lexical floor for near-verbatim claims.

**Refinement (stage 4b)** mutates claims instead of restarting:

1. Affected claims are found by similarity between the new detail and each claim plus the
   sub-query it answers.
2. Targeted delta sub-queries are issued **only** for the affected sub-intents.
   `full_corpus_search` stays `false` — and that flag in the trace is how this is *proven*,
   not asserted.
3. The model rewrites the answer as `KEEP` / `EDIT` / `ADD` / `DROP` instructions, so
   unaffected claims survive with their original ids and citations. A claim the model does not
   mention is kept: refinement never silently loses a fact. An `EDIT` that fails verification
   leaves the original claim standing.
4. The result is version *n+1* with parent *n*, and explicit `preserved` / `mutated` lists.

Session state is **ephemeral and scope-bound**: the store is constructed with its own id, no
method accepts a caller-supplied one (enforced by a test), and it is cleared when the socket
closes. There is no cross-session profile to leak because none exists.

---

## 6. Data provenance

- `doc_id = sha256(bytes)[:16]`, `chunk_id = f"{doc_id}_{i}"` — content-addressed, so
  re-ingesting rewrites in place instead of duplicating.
- **Documents** split on heading boundaries, then pack to a word bound; the heading trail
  becomes the chunk's heading and an outline number becomes its citation section:
  `[Doc_3 §2.1]`.
- **Transcripts** split on speaker turns and never mid-utterance; the section is the speaker
  and start time: `[Doc_7 §Priya @05:02]`.
- **PDFs** split per page: `[Doc_2 §page 4]`.
- Markers are assembled from the stored record in one place and are unique per chunk, so a
  marker in an answer resolves to exactly one chunk — or it is a fabrication and is stripped.

---

## 7. Trade-offs taken

| Decision | Bought | Paid |
|---|---|---|
| Rule controller by default | ~20 ms and $0 per chunk; deterministic and inspectable | Hand-calibrated thresholds; a model arm exists for comparison |
| Stability **or** clause completion as triggers | Short utterances still retrieve early | A closed clause fires on some incomplete thoughts |
| Merge guard at 0.90 cosine | No near-duplicate sub-queries | Genuinely distinct near-duplicate intents collapse (measured on ASQA ambiguous) |
| Sentence-level validation before streaming | Zero fabricated citations shipped | First token waits for the first sentence to be verified |
| Per-intent quota | No starved sub-intent | Two slots that global ranking would have spent elsewhere |
| MiniLM reranker on CPU | ~8x faster turns | Lower ranking quality than `bge-reranker-base` |
| numpy dot product, no ANN | No service, no index build | Would need an ANN index well beyond ~10⁵ chunks |
| No agent framework | Two model calls per turn, one readable turn loop | Orchestration is ours to maintain |

---

## 8. Failure modes and what mitigates them

| Failure | Mitigation | Where it is visible |
|---|---|---|
| Premature retrieval on noise (pitfall 1) | sufficiency gate + stability run + cancellation | `decisions[].reason`, `retrieval_cancelled` |
| Context lost on a late constraint (pitfall 2) | refinement path, claim preservation | `preserved`, `full_corpus_search: false` |
| Fabricated citations (pitfall 3) | validation against the retrieved set before shipping | `fabricated_citations_blocked` |
| Querying on presentation-only turns (pitfall 4) | suppression signal | `mode: suppress`, empty `retrieval` |
| Over-fragmented sub-queries (pitfall 5) | merge + cap guard | `decomposition.merged`, `capped` |
| Reranker unavailable or failing | degrade to fused order | `retrieval[].reranked: false` |
| LLM unavailable | deterministic offline arms; suite still passes | `decomposition.method`, `models.llm` |
| Corpus cannot answer part of the request | uncertainty instead of filler | `uncertainty[]` |

---

## 9. What is deliberately not here

No knowledge graph, no ontology layer, no multi-agent planner, no vector database service, no
cross-session memory, no auth. Each would add latency and operational surface for a theme
that grades architectural parsimony, and none is required by any acceptance gate. The engine
is two model calls and four stages per turn; that is the whole of it.
