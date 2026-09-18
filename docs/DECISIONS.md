# Decisions

Why things are the way they are, including the ones that were changed after measurement.
Each entry records what was chosen, what it was chosen over, and what the evidence was.

---

### D1 — WebSocket is the demo path; `POST /query` exists for testing only

Full-duplex is the theme. If the server ever receives a whole question in one message the
controller is bypassed and early retrieval becomes unmeasurable. `/query` therefore chunks
the text internally and runs the same turn loop; it is in the API for Swagger and for tests,
not for the demo.

### D1a — AG-UI on the wire, over the WebSocket; the input direction stays custom

Everything the server sends is a standard AG-UI event: a turn is a run, the four pipeline
stages are steps, each retrieval is a `corpus_search` tool call, the answer is a text message
per version, and the trace is shared state (snapshot, then JSON-Patch deltas). A standard
AG-UI client can read the console's stream, and `POST /agui` serves one over SSE. Chosen over
AG-UI's usual HTTP shape (one POST, one SSE stream per run) because AG-UI's input is a single
`RunAgentInput` sent at run start: it has no event for transcript chunks that arrive while the
run is already retrieving. A POST per chunk would add a round-trip to every chunk and shrink
the G2 lead. So the browser → server direction keeps three small input messages. The engine's
own events are unchanged; `slr/api/agui.py` translates at the boundary, so the trace and the
gates are measured exactly as before.

### D2 — Plain `asyncio`, no agent framework

The theme grades architectural parsimony explicitly. A turn is two model calls and four
stages; an orchestration framework would add a dependency, latency and indirection without
changing any gate. A test asserts that no banned framework appears in any `src/` import.

### D3 — Semantic stability as the primary trigger, clause completion as the secondary

Stability (`cos(e(p_t), e(p_{t-1})) ≥ 0.95`, twice) detects that the *meaning* stopped
moving, which works on unpunctuated ASR text. But it needs three chunks, and short requests
end before that. Measured on the demo fixtures, stability alone left every short
single-intent turn retrieving only at the utterance end. Adding a clause-completion trigger
(sentence punctuation or a comma, with sufficiency satisfied) brought early retrieval from
50% to 89% of eligible turns. Both are reported as separate reasons in the trace
(`intent_stable` vs `clause_complete`), so the split is visible.

### D4 — Corpus-grounded salience instead of NER

"Entity" is a number, a capitalised non-initial word, or a token with document frequency
≤ 25% of chunks. The first attempt used an IDF percentile, which broke on the small demo
corpus: most terms appear exactly once, so the 70th percentile marked almost nothing as
salient and "travel reimbursement rule employee trip" failed the sufficiency gate. Document
frequency is stable across corpus sizes and needs no model.

### D5 — Topic drift is measured on the words since the trigger

Comparing the whole prefix against the trigger embedding dilutes a topic change until it
disappears: by the time someone says "actually forget that, what encryption do laptops
need", the earlier two-thirds of the prefix keeps the cosine high. Measuring only the
uncovered segment against the segment that triggered the search makes the guard fire.

### D6 — Shared context is carried between clauses only when they are related

"Pune" should reach the cancellation sub-query in "a workshop in Pune, and the cancellation
policy". "Tokyo" must not reach the sick-leave sub-query in "the per diem in Tokyo, how many
sick days, and what if my laptop is stolen" — it was, and it poisoned that search. Context is
now carried only when the clause embeddings are similar (`SLR_CARRY_COS`, default 0.35).

### D7 — MiniLM reranker on CPU, `bge-reranker-base` one variable away

Measured on the development machine: `bge-reranker-base` takes ~5.5 s per 30 pairs,
`ms-marco-MiniLM-L-6-v2` ~0.7 s for the same work. The plan specified the former; on a CPU
host it would make a three-intent turn spend ~16 s reranking. The default is MiniLM with a
sigmoid temperature of 4 (its raw logits saturate, which flattens the adaptive margin cut so
that nothing is ever dropped), and `SLR_RERANK_MODEL` selects the larger model on a GPU host.

### D8 — A per-turn rerank pair budget

Reranking is the dominant CPU cost of a turn, and it grows with the number of intents. A
total pair budget per call (48 by default) means a four-intent turn reranks no longer than a
two-intent one. Provisional searches get a smaller budget (12) because they are guesses and
must not delay the real searches queued behind the same model lock.

### D9 — Support is scored over sentence windows, not whole chunks

Measured, and the single most surprising result in the build: `nli-deberta-v3-xsmall` scores
a claim at **0.986 entailment** against the sentence that states it, and **0.015** against
that same sentence with one unrelated sentence appended. These models are trained on
single-sentence premises. Support is therefore the maximum over 1- and 2-sentence windows of
the cited chunk (2-sentence windows cover claims that join adjacent facts), with a lexical
floor for near-verbatim claims. Without this, verbatim quotations from the corpus were being
withheld as unsupported.

### D10 — Auto-attribution has a higher bar than validation

When a model cites nothing, or cites something that does not exist, the engine may attribute
the sentence to the best-supporting chunk. Choosing a source on the model's behalf is a
stronger claim than checking one it supplied, so it needs 0.75 rather than 0.5. This was
added after a test caught an invented sentence ("parking costs 500 rupees per car") being
silently re-cited to a lexically similar chunk and shipped.

### D11 — Sentence-level release, so TTFT means time-to-first-*verified*-token

Tokens are buffered into a sentence, validated, and only then streamed. The alternative —
stream immediately and correct afterwards — cannot work: a fabricated citation would already
be on the reader's screen. The cost is that the first token waits for the first sentence;
the benefit is that `fabricated_citations` is zero by construction rather than by request.

### D12 — Absent vocabulary is the unanswerability signal

The offline extractive arm decides a sub-question is unanswerable when more than 40% of its
content words appear nowhere in the corpus. Earlier attempts weighted absent terms at maximum
IDF inside the coverage score, which conflated "the corpus does not discuss pets" with "the
corpus writes *booked* where the speaker said *book*", and suppressed answerable questions.
Light suffix folding (`book`/`booked`/`booking`) handles the second case; the absent-share
guard handles the first.

### D13 — The benchmark runs at speaking pace

At 20× replay the gap between the last word and the end-of-utterance signal shrinks from
250 ms to 12 ms, so retrieval that genuinely fired on the last chunk scored as late and G2
read 50% instead of 89%. The suite therefore defaults to `--speed 1.0`. Compressed replay is
available for development iteration, and the speed used is recorded in the report.

### D14 — Two dev corpora, and ambiguity is scored separately from composition

The guide's worked examples are *compound* requests (several unrelated needs in one breath).
ASQA's examples are *ambiguous* questions (one question with several readings, whose
disambiguations differ by a single qualifier). The over-fragmentation guard, which is
required by pitfall #5, merges near-duplicates and therefore collapses ASQA's ambiguity
variants by design. Both families are built and reported separately rather than picking the
flattering one.

### D15 — A deterministic offline mode, and it is the default without a key

`make eval` on a clean machine with no secrets must pass, so the engine has a heuristic
decomposer and an extractive synthesiser that are grounded by construction. They are not the
graded path — the report says which ran — but they make G1 real rather than conditional on a
judge having a key.

### D16 — The refinement bar depends on sentence form

A late constraint often shares little vocabulary with the request it narrows: "But we also
want to bring in an outside caterer" scored 0.478 against "Which Pune venue should I book for
a hands-on workshop with 24 people?", just under a flat 0.48 bar, and was routed as a new
search. A modifier cue on a *statement* is now enough at a much lower similarity; a cue on a
*question* needs more; a bare new question is never a refinement.

### D17 — Prices, including local compute

Cost per turn includes CPU time for the embedder, reranker and verifier at
`SLR_CPU_USD_PER_HOUR`. Reporting local inference as free would make the offline arm look
costless when it is the arm doing the most work.
