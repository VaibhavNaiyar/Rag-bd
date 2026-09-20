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

### D24 — Every gate counts against the fixture's labels, and shipped claims get a second grader

An audit of the gate code found three places where a number flattered the system:

- **G4 averaged per turn, over turns that kept at least one claim.** A turn where the
  verifier withheld every sentence dropped out of the average, which is exactly the turn a
  grounding metric exists to catch. Support is now pooled: verified sentences over every
  factual sentence the model wrote, withheld ones included. On the same ASQA run this moved
  G4 from 88.0% to 77.9%. The old per-turn mean is still reported alongside, labelled as such.
- **G2's denominators came from the system's own routing.** A question the controller wrongly
  suppressed left the eligible set instead of counting as late, and a reformat request it
  wrongly searched on never reached the false-trigger count. Both now come from the fixture's
  `expect.mode`. G2 also reports the stricter "search began before the last word" share,
  next to the end-of-speech figure the gate is judged on.
- **Enterprise recall was blank**, because those fixtures label gold documents, not
  passages. Recall@k is now measured on whichever the fixture labels, and every table names
  the unit.

A shipped claim was approved by the engine's own verifier, so the verifier's verdict cannot
be the evidence for it. The benchmark therefore re-reads every shipped claim against the
chunks it cites with a second, larger NLI model the engine never uses
(`nli-deberta-v3-base`) and reports the share it agrees with, listing the disagreements.
That figure is a floor, not an error count: on inspection, the base model rejects true
claims read off table-shaped text ("Career Hank Aaron – 2,297").

Two engine changes came out of reading the 46 withheld ASQA sentences, and neither lowers a
bar:

- Sentences such as "there is no information available regarding …" state that the documents
  are silent. They are uncertainty, not claims about the world, and now go there. First person
  only for "cannot provide": "the venue cannot provide AV equipment" is still a claim.
- A sentence whose cited block does not support it, but whose fact another retrieved block
  states, now ships with that block's citation. It must clear the stricter bar used for
  citations the engine picks itself (`SLR_AUTO_CITE_MIN`), and is counted as `recited`.

An opening connective ("However,", "Additionally,") is also removed before scoring, since it
asserts nothing. A cascade to the larger NLI model inside the engine was measured and left
out: it accepted 13 of 40 withheld claims against 4 for the small model, but three of the 13
were wrong (a bare "2", and a World Cup claim matched to a sentence about a player), and it
would add a model load and a second pass to every rejected sentence. The remaining withheld
sentences are the answer model stating what it knows rather than what the passages say;
withholding them is the gate doing its job.

### D25 — Start the answer from the mid-utterance search while the decomposer runs

The benchmark put the streaming pipeline's time to first token about 1.7 s behind the
batch baseline. The traces showed where it went: after the end of speech a turn waited on
the decomposer (a gpt-4.1 call, median about 1.2 s) before the answer model was even asked.
The search itself was already done: the decomposed query almost always reuses the search
that started while the user was still speaking, so "retrieve" costs about 0 ms after the end.

So at the end of speech the engine now starts the answer from that search's evidence, with
the utterance as spoken as the question, while the decomposer runs. When the decomposer
returns one reading close enough to that search's query for the normal path to reuse it
(`SLR_REUSE_COS`, the same rule), the normal path would have answered from exactly this
evidence, and the answer already under way is kept. With several readings, or a rephrasing
that no longer matches, the speculative answer is cancelled and each reading is searched and
answered as before. Nothing reaches the user before the decomposer has decided, so the answer
never addresses a question the decomposer then splits.

A first version searched the whole utterance afresh at the end instead of reusing the
mid-utterance search. It was slower than no speculation (median 3.8 s against 2.5 s on 16
turns): a full rerank on CPU takes 2–4 s, longer than the decomposer it was meant to hide.

Measured on the 20 ASQA single-reading questions, each played twice with and twice without,
interleaved: median time to first validated token 1.88 s with, 2.57 s without; p90 4.4 s
against 5.3 s. A dropped speculation still costs its prompt tokens, and they are billed: a
stream stopped before the provider's usage report is charged an estimate (characters / 4)
rather than nothing.

A faster decomposer was also measured and rejected: gpt-4.1-mini was no faster in practice
(median 1.18 s against 1.23 s) and isolated two or more readings in 8 of 40 ASQA ambiguous
questions against 22 for gpt-4.1.

### D26 — What did not move G3 on ASQA

ASQA's ambiguous questions sit near 55% against the 70% bar. Every change was measured by
running only the decomposer over all 64 labelled utterances (enterprise and ASQA), scored
with the G3 matcher, before spending a benchmark run on it:

| Variant | ASQA compound, ≥2 readings | kept single (ASQA / enterprise) |
|---|---|---|
| current prompt | 22/40 | 19–20/20, 3/4 |
| more reading cues (who: person or party; change over time; breakdown by category) | 23/40 | 18/20, 3/4 |
| plus "several true answers at once" and "split on the line that changes the answer" | 23/40 | 17/20, 3/4 |
| plus the page titles the mid-utterance search had already found | 21/40 | 18/20, 2/4 |
| gpt-5-mini, minimal reasoning | 25/40 | 12/20, 0/4 (and 3.4 s a call) |

None beat the current prompt by more than run-to-run noise, and each cost single questions
that must stay single (pitfall #5), so the prompt is unchanged. Most remaining misses are
splits the decomposer does make along a different line than ASQA's annotators (a character
and its revival series, where ASQA wants the character and the actor), or facets the
annotators chose that the question does not signal ("how did the US buy it" alongside "from
whom"). The matcher threshold was not touched. An early draft put two ASQA questions into the
prompt as examples; they were replaced before any number was taken, because a fixture in the
prompt measures memory, not decomposition.

### D27 — Cross-encoder batches sorted by length

After D25 the slow turns were the ones with several readings: a fresh search per reading
after the decomposer, taking 2–3 s, although the same search timed alone took 0.1 s. That
figure was the reranker's cache answering; an uncached pass of 48 pairs through MiniLM-L6
takes about 3 s on the 4-core laptop CPU the benchmark runs on. A batch is padded to its
longest pair, and passages run from about 40 to 280 tokens, so one long passage made every
pair in its batch pay for it. Pairs are now sorted by length before batching, for the
reranker and for the NLI verifier, and the scores returned in the caller's order. Nothing
changes but the time: on 192 pairs the largest score difference against the unsorted pass
was 1.5e-7, and the time went from 14.9 s to 7.1 s.

Two faster options were measured and not taken, because they change rankings: int8 dynamic
quantisation (a further 30%, top-1 agreement 11 of 12 queries) and a 128-token input limit
(top-3 overlap 35 of 36 alone, top-1 agreement 10 of 12 combined with quantisation).

### D28 — Answer in the evidence's own terms

With D25 in place the enterprise set fell to 82.4% support, and the withheld sentences were
true: "the current hotel nightly rate limit for business trips in New York is USD 360 per
night" against a block that says "New York: … hotel limit USD 360 per night". The entailment
model is right to call that neutral: "current" and "for business trips" came from the request
(and the decomposer's sub-query), not from the block. The synthesis prompt now asks for each
fact in the block's own terms, without the request's framing, and says why: every sentence is
checked against the block it cites. It also asks that every name, date and number appear in
the cited block, and a sub-question naming a number no retrieved block contains is marked
so, so the model writes the UNCERTAIN line rather than a remembered answer.

Full benchmark after the change: enterprise support 89.5% (85 of 95; baseline 87.7%), all six
gates green, time to first token 2.49 s against the baseline's 2.29 s. ASQA stays at 78.8%
(156 of 198) against the baseline's 87.9%, and the split explains it: single-reading turns
reach 85.1% (57 of 67), turns split into readings 70.5% (62 of 88). A reading the corpus has
no passage for is where the model reaches for what it remembers, and the verifier withholds
it. The baseline asks one question per turn, so it never writes those sentences, and it also
answers none of the other readings (recall 62.2% against 78.8%). Raising G3 on ASQA would
move more turns into the weaker bucket; the report shows both numbers rather than trading
one for the other.

### D29 — Nearest-neighbour examples for the decomposer, from ASQA's train split

Tree of Clarifications (Kim et al., EMNLP 2023) and DIVA (In et al., NAACL 2025) both prompt
for the readings of an ambiguous question with examples chosen per question by nearest
neighbour from a training set. The dataset builder now writes such a bank from ASQA's train
split only (`data/decompose_examples.jsonl`: 4,353 questions with their readings, plus each
reading as a question that must not be split). Every fixture comes from the dev split, and
the engine skips any example 0.90 cosine or closer to the utterance, because ASQA itself has
near-duplicate questions across its splits (6 of our 75 dev fixture questions have a train
twin at 0.90 or above). The enterprise corpus has no bank and gets no examples.

Measured with the decomposer alone over all labelled utterances (G3 matcher, unchanged):
no examples 22/40; nearest questions that split only, 32/40 but single questions kept single
fell to 12/20; nearest of both kinds pooled, 20/40 (an ambiguous question's nearest
neighbours are its own readings); the nearest *k* that split plus the nearest *m* that did not,
27/40 at 5+2, 29/40 at 8+2 and 8+4. Adding the mid-utterance passages, ToC's other input,
did not help (26/40) and stays off (`SLR_DECOMPOSE_PASSAGES=0`). Full benchmark at 8+4:
ASQA G3 **72.5%** (29/40), single questions kept single 85%.

### D30 — Evidence first, and a trained fact-checker as the second reading

Two published ideas, both aimed at the ASQA sentences the verifier rejected.

*Attribute first, then generate* (Slobodkin et al., ACL 2024): the synthesis prompt now asks
for one hidden `EVIDENCE:` line per sub-question, copying the span that answers it, or
`EVIDENCE: NONE`, in which case the sub-question gets only an UNCERTAIN line. The line is kept
whole by the sentence splitter, never shown, never counted as a claim, and recorded in the
trace as `attributions`.

*MiniCheck* (Tang, Laban and Durrett, EMNLP 2024; MIT licence): a model trained for "is this
sentence supported by this document". Re-scoring the previous run's sentences offline showed
the small NLI model rejecting true ones ("France is the most recent winner, having won in
2018" against "The current champion is France, who won the title in 2018"). The verifier is now
a cascade: the NLI model decides, and a sentence it scores below 0.75 is read again by
MiniCheck-RoBERTa-Large; the grounder's bars (0.5 for the model's citation, 0.75 for one the
engine picks) are unchanged. The DeBERTa variant was slower and rejected the France sentence.

Reading the withheld sentences also found splitter bugs that had been counted as claims: cuts
at "St.", "Dr." and list numbers ("Community of St", "Dr .", "1 ."). Abbreviations and a list
number opening a line no longer end a sentence. "65.46%" and "65.46 percent" are read as one
quantity.

Full benchmark after D29 and D30: enterprise G4 **93.1%** (81/87), all six gates green; ASQA G4
**84.3%** (166/197), 0.7 points and two sentences under the bar, from 78.8%. What remains
withheld on ASQA is mostly the answer model adding what the quoted span does not say ("India
won in 1983 and 2011" from a span that names 1983). The independent audit agrees with 98.8% of
enterprise and 85.5% of ASQA shipped claims, down from 95.5%: reading the disagreements, most
are the audit model rejecting true claims ("the 2001 Finals opponent was the Philadelphia
76ers"), and about five of 166 are claims MiniCheck accepted that stretch their passage (a
Test-cricket record cited to a sentence about international cricket). The cost is time:
MiniCheck runs on CPU on every weak sentence, and first-token latency rose to 3.4 s
(enterprise) and 4.5 s (ASQA) against the baseline's 2.5 s.
