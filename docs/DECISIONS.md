# Decisions

Why things are the way they are, including the ones that were changed after measurement.
Each entry records what was chosen, what it was chosen over, and what the evidence was.

---

### D1 — WebSocket is the live path; `POST /query` exists for testing only

Full-duplex is the theme. If the server ever receives a whole question in one message the
controller is bypassed and early retrieval becomes unmeasurable. `/query` therefore chunks
the text internally and runs the same turn loop; it is in the API for Swagger and for tests,
not for the live console.

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
end before that. Measured on the enterprise fixtures, stability alone left every short
single-intent turn retrieving only at the utterance end. Adding a clause-completion trigger
(sentence punctuation or a comma, with sufficiency satisfied) brought early retrieval from
50% to 89% of eligible turns. Both are reported as separate reasons in the trace
(`intent_stable` vs `clause_complete`), so the split is visible.

### D4 — Corpus-grounded salience instead of NER

"Entity" is a number, a capitalised non-initial word, or a token with document frequency
≤ 25% of chunks. The first attempt used an IDF percentile, which broke on the small enterprise
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

### D14 — Two dev corpora; on ASQA a compound utterance is an ambiguous question

The guide's worked examples are *compound* requests (several unrelated needs in one breath).
ASQA's are *ambiguous* questions: one question with several readings, whose disambiguations
differ by a qualifier. The mechanics are the same (one utterance, N sub-queries, N
retrievals, one fused answer), so G3 is scored on ASQA's ambiguous questions against its gold
disambiguations, and on the enterprise corpus against the guide-style compounds. The
decomposer is told to split an underspecified question into its readings; the merge guard
(pitfall #5) merges paraphrases at cosine ≥ 0.95 but never two queries that differ in a
number, because "2014 winner" and "2018 winner" are two readings, not a duplicate.

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

### D17 — Prices, including local compute, per model

Cost per turn includes CPU time for the embedder, reranker and verifier at
`SLR_CPU_USD_PER_HOUR`. Reporting local inference as free would make the offline arm look
costless when it is the arm doing the most work. Model tokens are priced per model
(`slr/telemetry/cost.py: MODEL_RATES`, overridable with `SLR_PRICE_<MODEL>=in,out`): one
flat rate under-reports a gpt-4.1 decomposition thirteen-fold.

### D18 — Retrieved text is fenced as untrusted data

A corpus document is written by someone other than the user, so without a boundary every
document is an instruction channel into the model. Each evidence block's source and text sit
inside `<document>` … `</document>`, the synthesis and refine prompts declare fenced content
to be data and never instructions, a document cannot close its own fence, and text that
addresses a model ("ignore previous instructions", "the assistant must say") is marked
`flagged` and logged per turn (`fusion.flagged_chunk_ids`). The detector is a narrow pattern,
deliberately: an ordinary "you must" in a policy is not an attack, and a false flag costs
only a label. This is the cheapest real defence, not a complete one; the grounding verifier
behind it still withholds any claim the evidence does not support.

### D19 — A circuit breaker on the model, and every model step has an offline twin

If the provider is slow or rate-limiting, a turn must not wait out a timeout per call. Three
consecutive *health* failures (timeouts, dropped connections, 429s, 5xx) open the circuit for
30 s; one trial call then decides. A bad request or a rejected key never counts: retrying it
fails forever, but it is not an outage, and one malformed call must not take the model away
from every turn. While the circuit is open, or when a stream fails, each step takes its
offline strategy: decomposition the clause splitter, synthesis the extractive answer,
refinement the extractive refine, a reformat the offline regrouping. A stream that fails
after text was shipped keeps that text and says it was cut short, rather than gluing a
second answer onto half of the first. The trace records every fallback under `degraded`.

### D20 — The trace record is also exported as OpenTelemetry spans

G6 asks for structured logs *or dashboards*. The JSONL record stays the system of record;
with `SLR_OTEL_ENDPOINT` set, each record is also exported over OTLP/HTTP as one trace: a
`turn` span with `listen`, `plan`, `retrieve` and `synthesise` children rebuilt from the
record's own timings, decisions and searches as span events. Spans carry ids, counts,
timings, outcomes, costs and model names only; never the utterance, a query, chunk text or
the answer, because a collector is a third party. `tests/test_otel.py` asserts that.

### D21 — A baseline pipeline, run on the same fixtures

The guide asks for a comparison against the baseline pipeline. The baseline is a
conventional RAG turn, built from the same parts so the difference is only the design:
`SLR_CONTROLLER=batch` does nothing until the speaker stops and then always retrieves (no
suppression, no refinement), and `SLR_DECOMPOSE=off` makes the whole utterance one search.
`make eval` replays every fixture through it after the main run, and the report's §4 puts
the two side by side: retrieval before speech ends, searches on reformat turns, multi-intent,
refinement, grounding, TTFT, completion time and cost.

### D22 — A stronger model for decomposition only, and readings only where the world has them

Measured on ASQA's 40 compound fixtures with the same prompt and guard: gpt-4o-mini
isolated two or more gold readings in 13, gpt-4.1-mini in 11, gpt-4.1 in 25, gpt-5.4 in 22
(3.8 s), gpt-5-mini in 24 but split 14 of 20 single questions and took 7 s. Grounding the
decomposer in a BM25 look at the corpus did not help (10 and 21). So `SLR_DECOMPOSE_MODEL`
defaults to gpt-4.1 while the answer stays on gpt-4o-mini: about 0.3 US cents more per turn,
on the one call that decides G3.

gpt-4.1 then invented readings for the enterprise corpus: a hotel limit "per night" and "per
stay", a hall's "main room" and "breakout room", even a "[specific venue name]" placeholder.
Pitfall #5 is exactly that, so the prompt now says readings belong to questions about the
wider world, and a question about the organisation's own rules, prices or facilities has one
reading. That cost three ASQA splits (25 → 22 of 40) and bought every single question kept
single (17 → 20 of 20 on ASQA, 1 → 3 of 4 on the enterprise set). ASQA's ambiguous questions
sit below the 70% bar and the report states it rather than tuning the matcher. The fourth
enterprise "single" asks for a per diem rate *and* a hotel limit, two needs; its label is
left as authored rather than edited to match the system.

### D23 — When no reading is verified, ask which was meant

The guide's grounding rule allows two answers to missing evidence: an uncertainty indicator,
or a targeted clarification. When a request was split into several readings and the
documents confirmed none, the answer version carries those readings as `clarification`, and
the console offers them as choices; choosing one speaks "Sorry, I meant …", which the
controller routes as a refinement of the same answer (D16, D19). The idea comes from a
planner that asks rather than guesses when a request is ambiguous. One alternative was
measured and dropped: withholding a sub-question from the answer model when its evidence
looked weak. Across 149 ASQA sub-queries the top rerank score barely separates those that
ended with a verified claim (median 0.858) from those that did not (0.832), so any cut would
drop good evidence with the bad; the verifier remains the gate.
