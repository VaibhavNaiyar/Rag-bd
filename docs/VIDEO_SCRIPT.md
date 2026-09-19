# Demonstration walkthrough (≤ 5 minutes)

What to show, in what order, and what to say while it is on screen. Every claim below is
visible on screen at the moment it is made — nothing is asserted that the trace does not show.

**Setup (before recording):** `docker compose up --build`, then open
<http://localhost:8000/?replay=1>. The fixture bar appears; keys `1`–`4` replay fixtures. Have
a second tab on <http://localhost:8000/trace?limit=1> for the raw record.

---

## 0:00 — The problem (20s)

> "A batch RAG system waits for you to stop speaking, then starts searching. In a real
> conversation that adds the length of your sentence to every answer — and one sentence often
> contains three questions."

Show the idle console: corpus size in the header, the trace rail empty on the right.

---

## 0:20 — Early retrieval (press `1`: compound_01) (70s)

The transcript streams in 3–5 word chunks. Point at the controller timeline as it fills.

> "The controller reads the transcript as it arrives. `wait` — the request is still
> incomplete. Here it says `clause_complete`, and a search starts **while the user is still
> speaking**."

Point at the retrieval marker sitting **left of** the utterance-end rule.

> "That marker is left of the end-of-utterance line. The trace records the lead in
> milliseconds — this turn started retrieving about 1.3 seconds before the speaker finished.
> That is gate G2."

Then the decomposition chips appear.

> "One sentence, three separate needs: venue capacity, cancellation policy, catering. One
> model call produced three search-ready queries — not paraphrases of one question, which is
> a failure mode the guide calls out explicitly."

Point at the evidence list grouped by sub-query.

> "Each sub-query has its own evidence, and the fusion step guarantees every intent a slot in
> the context window, so no question the user asked gets crowded out by a louder one."

Finally the answer streams with citation chips.

> "Every sentence ends with a citation that resolves to a real chunk — click one and you see
> the exact passage it came from."

---

## 1:30 — Refine, don't restart (press `2`: late_detail_01) (60s)

First turn answers the travel reimbursement question. Then the second utterance arrives.

> "Now a late detail: *the trip was international and the booking was made after travel*.
> This is not a new question — it narrows the answer that already exists."

Point at the decision pill: `refine · late_constraint`.

> "The controller routes it to refinement. It issues targeted queries for the affected
> sub-intent only."

Point at the version diff panel.

> "Version 2, from version 1. These claims are carried through untouched with their original
> citations. These are the ones the detail rewrote. And this line — **no full-corpus
> search** — is the flag the evaluation reads to prove the session state was reused rather
> than thrown away. That is gate G5."

---

## 2:30 — Suppression (press `3`: presentation_01) (40s)

> "Some turns need no search at all. *Please repeat your last answer in two bullets.*"

Point at the decision pill `suppress · presentation_restructure` and the empty retrieval row.

> "Zero vector queries. The answer is rebuilt from the previous one, keeping its original
> citations. Querying the index here would burn tokens and risk drifting away from what was
> already verified."

---

## 3:10 — Grounding and uncertainty (press `4`: unanswerable_01) (50s)

> "This asks something the corpus does not contain."

Point at the uncertainty note under the answer.

> "Rather than filling the gap with plausible prose, the engine says which part it could not
> verify. Generation is streamed, but each sentence is released only after its citations
> resolve against the retrieved set and its support clears a threshold — so a fabricated
> citation is never on screen to begin with. The counter reads zero fabricated citations, and
> that is structural, not a prompt instruction."

---

## 4:00 — Telemetry and cost (40s)

Switch to the trace tab, or scroll the metrics bar.

> "Every turn emits one record: each controller decision with its reason, when retrieval
> started relative to the utterance end, the sub-queries, the evidence per branch, the answer
> version lineage, citation support, time-to-first-token, and the cost of the turn —
> including local model time, priced rather than reported as free."

```bash
curl localhost:8000/trace?limit=1 | jq '{mode, before_utterance_end, citation_support_rate, cost}'
```

---

## 4:40 — Reproducibility (20s)

> "One command builds the image, indexes whatever corpus is mounted, and serves the console.
> `make eval` replays every fixture through the same WebSocket path and writes the benchmark
> report — six gates, three ablations, and the failures it found, measured on the machine
> it ran on."

Show `docs/BENCHMARK_REPORT.md` scrolling past the gate table.

---

## Notes

- Record the video with an `OPENAI_API_KEY` set if you want the model-backed answers; without one
  the engine uses its deterministic offline arms, which are grounded but blunter in wording.
  Either way the behaviour being demonstrated — early retrieval, decomposition, refinement,
  suppression, grounding — is identical.
- If a replay is too fast to narrate, the fixture bar has a speed control; 0.5× is comfortable.
