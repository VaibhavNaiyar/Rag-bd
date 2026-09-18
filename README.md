# Streaming Live RAG

**Samsung PRISM GenAI Hackathon · Theme 04** — an event-driven RAG engine that starts
retrieving before the speaker finishes, decomposes one spoken request into the several
questions it actually implies, and refines its answer when a late detail arrives instead
of starting over.

```bash
docker compose up --build      # the whole system, one command
open http://localhost:8000     # console, API docs at /docs
```

The operator console's source is in [Rag-fd](https://github.com/VaibhavNaiyar/Rag-fd). A
built copy ships in `src/slr/api/static/`, so this repository runs on its own without Node.
To rebuild it: `make web FD=../Rag-fd`.

That is the entire setup. No API key is required: without one the engine runs its
deterministic offline arms (heuristic decomposition, extractive synthesis) and the full
evaluation suite still passes. With `OPENAI_API_KEY` set it uses two model calls per
turn — one to decompose, one to synthesise.

---

## What it does

| The problem | What happens here |
|---|---|
| Waiting for the user to stop speaking costs seconds | A controller reads the transcript as it arrives and fires a **provisional search** the moment the request is searchable — typically **~1.3s before the utterance ends** |
| One utterance hides several questions | A single model call **decomposes** it into up to four search-ready sub-queries, guarded against splitting one question into near-duplicates |
| Multi-source evidence dilutes the answer | BM25 + dense branches per sub-query, fused with RRF, reranked by a cross-encoder, then merged with a **per-intent quota** so no question the user asked is starved of evidence |
| A late detail should sharpen the answer | The refinement path rewrites only the affected claims, **preserves the rest with their original citations**, and never re-runs a full-corpus search |
| Models invent citations | Every sentence is validated against the retrieved set **before it is streamed**. A marker that does not resolve is stripped and counted; an unsupported claim is withheld and reported as uncertainty |

Every one of those is visible in the trace, and measured by `make eval`.

---

## Running it

### Container (what a judge should use)

```bash
docker compose up --build                       # API + console on :8000
docker compose run --rm eval                    # the full benchmark
docker compose run --rm app ingest --corpus /data/my_corpus --out /data/index
```

The app serves the demo corpus shipped in the image. To serve your own, put it under
`./data/` and set `SLR_CORPUS_DIR` to it (not `./data/corpus/`, which holds the ASQA eval
corpus). It is indexed at startup: txt, md, json, jsonl and pdf, with transcripts detected
by shape. Nothing assumes a filename convention.

### Local

```bash
make venv          # virtualenv + pinned dependencies
make models        # pre-download the three local models
make ingest        # build an index from evals/corpora/demo
make serve         # http://localhost:8000
make eval          # the benchmark; rewrites docs/BENCHMARK_REPORT.md
make test          # 131 tests
make demo          # replay fixtures and print the event trace
```

---

## The shape of the system

```
transcript chunks ──▶ [1] Retrieval controller ──▶ provisional search (cancellable)
                            │ wait / retrieve / suppress / refine
        utterance end ──────┼──▶ [2] Multi-intent decomposer ── 1 model call
                            │           │
                            │           ▼
                            │    [3] Per sub-query: BM25 + dense ─ RRF ─ rerank ─ margin cut
                            │           │
                            │           ▼  coverage-guaranteed fusion (quota per intent)
                            │    [4] Grounded synthesis ── 1 model call, streamed
                            │           │  every sentence validated before it ships
                            ▼           ▼
                        trace record   answer + citations + uncertainty
```

- `src/slr/controller/` — the four signals: semantic stability, content sufficiency,
  presentation suppression, late-constraint refinement.
- `src/slr/retrieval/` — hybrid search, RRF, adaptive-margin rerank, fusion, the evidence package.
- `src/slr/synthesis/` — streamed generation, citation validation, claim-level refinement.
- `src/slr/stream/engine.py` — the turn loop. Plain `asyncio`; no agent framework.
- `src/slr/telemetry/` — the per-turn trace record and the cost ledger.
- `prompts/` — every prompt, loaded at runtime. None are inlined in code.
- `evals/` — fixtures, gates, ablations, report. Never imported by `src/`.

Full design rationale: [`docs/ARCHITECTURE_BRIEF.md`](docs/ARCHITECTURE_BRIEF.md).
Trace field reference: [`docs/TELEMETRY_SCHEMA.md`](docs/TELEMETRY_SCHEMA.md).
Measured results: [`docs/BENCHMARK_REPORT.md`](docs/BENCHMARK_REPORT.md).

---

## The API

| Route | What it is for |
|---|---|
| `WS /stream` | **The demo path.** Transcript chunks in, [AG-UI](https://docs.ag-ui.com) events out |
| `POST /agui` | AG-UI over SSE for any standard AG-UI client: a `RunAgentInput` in, one run streamed back |
| `POST /query` | Testing and Swagger only — it still streams the text through the controller internally |
| `GET /health` | Readiness, corpus size, which models are loaded |
| `GET /trace`, `GET /trace/{turn_id}` | The per-turn records the gates read |
| `GET /fixtures` | Replayable fixture names |

Typed input is chunked and released at speaking pace by the client, so the controller runs
identically whether the input was spoken, typed or replayed. If the server ever received a
whole question in one message the controller would be bypassed and early retrieval would
become unmeasurable — which is why `/query` chunks internally too.

---

## Evaluation

`make eval` runs every fixture through the same WebSocket turn loop the demo uses, then
writes `docs/BENCHMARK_REPORT.md` and `evals/results/latest.json`.

Two dev corpora: a small synthetic enterprise corpus (`evals/corpora/demo/`) that backs the
worked examples in the theme guide, and ASQA (`din0s/asqa`, Apache-2.0, all 5,301 samples).
`make dataset` (`evals/build_dataset.py`) builds four artifacts from ASQA:

| Artifact | What it is |
|---|---|
| `data/corpus/` | every passage from every sample, train and dev pooled: 8,366 pages, 15,488 passages. The pooling puts real distractors next to each gold passage. |
| `evals/gold.jsonl` | per sample: gold sub-questions (G3), the passage each answer came from (retrieval recall), reference long answer |
| `evals/transcripts/` | each case's turns in 3-5 word chunks at 150 wpm, each with `atMs`, plus the `utterance_end_ms` G2 measures against |
| `evals/fixtures/*/asqa_*` | dev-split cases: `compound/` (ambiguous question, 2-6 gold sub-questions), `single/` (one disambiguated sub-question), `late_detail/` (the question, then ASQA's own disambiguating condition), `suppression/` (a question, then a presentation-only turn) |

The raw dataset is vendored as `evals/data/asqa/*.jsonl.gz` so the suite runs offline;
`python -m evals.build_dataset --refresh` re-downloads it. ASQA's questions are compound
because they are *ambiguous* (several valid readings), not because they bundle unrelated
asks. The pipeline mechanics are the same: one utterance, N sub-questions, N retrievals,
one fused answer.

Everything under `evals/` is test data. `src/` never imports from it, and
`tests/test_compliance.py` fails if a gold answer string appears anywhere in `src/`.

The benchmark runs at **speaking pace by default**, because compressing replay time also
compresses the gap between the last word and the end-of-utterance signal — which is exactly
what the early-retrieval gate measures.

---

## Configuration

Everything is an environment variable; see [`.env.example`](.env.example). The knobs the
ablations flip:

| Variable | Default | Effect |
|---|---|---|
| `SLR_BRANCHES` | `bm25,dense` | retrieval branches |
| `SLR_CONTROLLER` | `rule` | `model` puts an LLM call on every transcript chunk |
| `SLR_QUOTA_PER_INTENT` | `2` | `0` removes the per-intent coverage guarantee |
| `SLR_LLM` | `auto` | `offline` forces the deterministic arms |
