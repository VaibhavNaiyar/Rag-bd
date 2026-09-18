# Telemetry & Observability Schema

**Streaming Live RAG — Samsung PRISM GenAI Hackathon, Theme 04**

Two streams carry the same facts:

- **Wire events** (`WS /stream`, `POST /agui`) — standard [AG-UI](https://docs.ag-ui.com)
  events: what the console, or any AG-UI client, renders live as a turn happens.
- **Trace records** (one JSON object per turn, appended to `SLR_TRACE_PATH`, served by
  `GET /trace`) — what the eval gates read.

Gate G6 requires that **every** turn emits **every** field of the trace record. The required
list lives in code (`slr/telemetry/trace.py: REQUIRED_FIELDS`) and `evals/gates.py` fails the
gate on a single missing field, so this document cannot drift from what is enforced.

---

## 1. Trace record

One object per turn. Suppressed and refined turns emit the same fields as retrieved ones;
fields that cannot apply are `null` rather than absent.

| Field | Type | Meaning |
|---|---|---|
| `trace_version` | int | schema version (currently 1) |
| `session_id` | string | ephemeral session; never reused, never persisted |
| `turn_id` | string | `t1`, `t2`, … within the session |
| `started_at` | int | epoch ms at `turn.start` |
| `mode` | `retrieve`\|`refine`\|`suppress` | the final controller route for this turn |
| `controller` | `rule`\|`model` | which controller arm ran |
| `utterance` | string | the full transcript as received |
| `utterance_end_ms` | int | ms from turn start to the end-of-utterance signal |
| `chunks` | `[{text, at_ms}]` | every transcript chunk with its arrival time |
| `decisions` | `[{decision, reason, confidence, at_ms, query, signals}]` | one entry per chunk plus the final decision; `signals` carries the raw signal values behind it |
| `retrieval_events` | `[{sub_query_id, query, trigger, at_ms, before_utterance_end}]` | every search launched, plus `retrieval_cancelled` entries |
| `first_retrieval_ms` | int\|null | when the first search began; `null` when nothing was searched |
| `before_utterance_end` | bool\|null | **the G2 fact**: did retrieval begin before the speaker finished |
| `sub_queries` | `[{id, text, source, span, confidence, reused_from}]` | provisional and decomposed queries |
| `decomposition` | object\|null | `method` (`llm`/`heuristic`/`heuristic_fallback`/`refine_*`), the raw model output, `merged`, `capped`, `ms`; on a refine turn also `affected_claims` and `claim_similarity` |
| `retrieval` | `[{sub_query_id, candidates, branch_counts, kept[], reranked, reused, ms}]` | per sub-query: how many candidates each branch returned and which chunks survived the margin cut |
| `fusion` | object\|null | `final_count`, `quota_applied`, `quota_promoted`, `per_sub_query`, `full_corpus_search`, `carried_from_session`, `chunk_ids` |
| `answer` | object | `version`, `parent`, `body`, `claims[]`, `preserved[]`, `mutated[]`, `full_corpus_search`, and a `grounding` block |
| `citations` | `[string]` | every marker the answer shipped, resolved against the index |
| `citation_support_rate` | float\|null | supported claims ÷ generated claims |
| `fabricated_citations` | int | markers in the **shipped** answer that do not resolve — 0 by construction |
| `fabricated_citations_blocked` | int | markers the model produced that were stripped before shipping |
| `uncertainty` | `[string]` | what could not be verified, in the user's terms |
| `latency_ms` | object | see §3 |
| `cost` | object | see §4 |
| `models` | object | embedder, reranker, verifier, llm, controller, branches, quota — the run's identity |
| `errors` | `[string]` | anything that degraded, e.g. a failed provisional search |

### `answer.grounding`

| Field | Meaning |
|---|---|
| `generated_claims` | sentences the model produced that asserted something |
| `supported_claims` | those that cleared `SLR_SUPPORT_MIN` against their cited chunk |
| `demoted_claims` | withheld and moved into `uncertainty` |
| `auto_cited` | sentences the engine attributed itself (above the stricter `SLR_AUTO_CITE_MIN`) |
| `fabricated_markers` | the exact markers that were stripped |
| `verifier` | which support scorer ran |

### `answer.claims[]`

`{id, text, chunkIds, subQueryId, support}` — the claim as shipped, the chunks that support
it, the sub-question it answers, and its support score. Claim ids are stable across versions,
which is what makes `preserved` meaningful: a preserved claim keeps its id, its text and its
citations from the previous version.

---

## 2. Wire events (AG-UI)

The engine emits its own event vocabulary (`slr/contracts.py: WIRE_SCHEMA`, validated before
anything is sent). `slr/api/agui.py` is the one place those become what leaves the server:
standard AG-UI events. `tests/test_agui.py` checks every stream is protocol-correct (runs and
steps balanced, tool calls started before their results, every state patch applies), and the
console validates each frame against `@ag-ui/core`'s schemas on arrival.

| Engine event | AG-UI events | Payload |
|---|---|---|
| `session.ready` | `RUN_STARTED` · `STATE_SNAPSHOT` · `RUN_FINISHED` (a short run of its own) | shared state `{session{id, corpus}, turns{}}` |
| `turn.start` | `RUN_STARTED` (`runId` = turn, `threadId` = session) · `STATE_DELTA` · `STEP_STARTED listen` | a blank turn at `/turns/<id>` |
| `transcript.chunk` | `STATE_DELTA` | `/turns/<id>/transcript/-` ← `{text, atMs}` |
| `controller.decision` | `STATE_DELTA` | `/turns/<id>/decisions/-` ← `{decision, reason, atMs, confidence}` |
| `retrieval.started` | `TOOL_CALL_START corpus_search` · `ARGS {query, trigger, atMs}` · `END` | **the G2 evidence** |
| `retrieval.cancelled` | `TOOL_CALL_RESULT` | `{cancelled: true, reason}` |
| `utterance.end` | `STATE_DELTA` · `STEP_FINISHED listen` · `STEP_STARTED plan` | `/turns/<id>/utteranceEndMs`, the line every lead is measured against |
| `subqueries` | `STATE_DELTA` · `STEP_STARTED retrieve` | `/turns/<id>/subQueries` ← `[{id, text, source}]` |
| `retrieval.result` | `TOOL_CALL_RESULT` (a reused sub-query first gets its own START/ARGS `{query, reused}`/END) | `{candidates, kept[Hit], reused}` |
| `fusion.final` | `STATE_DELTA` · `STEP_STARTED synthesise` | `/turns/<id>/fusion` ← `{hits[Hit], quotaApplied, fullCorpusSearch}` |
| `answer.token` | `TEXT_MESSAGE_START` (once) · `TEXT_MESSAGE_CONTENT` | `messageId` = `<turnId>:v<n>`; released only after the sentence validates |
| `answer.version` | `TEXT_MESSAGE_END` · `STATE_DELTA` · `STEP_FINISHED` | `/turns/<id>/versions/<n>` ← `{parent, claims, preserved, mutated, uncertainty, citationSupportRate, fabricatedCitations}` |
| `turn.complete` | `STATE_DELTA` · `RUN_FINISHED` with `usage` | `/turns/<id>/latencyMs`, `/turns/<id>/cost`; `usage` = `TokenUsage` per model, absent when no model ran |
| `error` | `RUN_ERROR` | `{code, message}` |

`Hit` = `{chunkId, docId, section, text, score, branches, subQueryIds, citation, heading}`.
`citation` is built server-side from the chunk record; the model never writes one.

Client → server (`WS /stream` only): `utterance.start`, `utterance.chunk{text}`,
`utterance.end`, `replay{fixture, speed}`, `session.new`. These stay custom because AG-UI has
no event for input arriving while a run is under way, which is what early retrieval needs.
`POST /agui` takes a standard `RunAgentInput` instead: earlier user messages replay silently
as earlier turns, and the last one is streamed back as one run.

---

## 3. Latency

Everything is measured from the **end of the utterance**, because that is the moment a batch
system would have started work.

| Field | Meaning |
|---|---|
| `first_retrieval_rel_end` | negative = retrieval began *before* the speaker finished. This is the headline number |
| `first_token_after_end` | time to first **verified** token |
| `complete_after_end` | time to the final answer version |
| `first_token_abs`, `complete_abs` | the same instants relative to turn start |

The shared state's `turns/<id>/latencyMs` carries `{firstRetrieval, firstToken, complete}` in
the same convention.

---

## 4. Cost

Per turn, drained from a usage ledger that every component writes into
(`slr/telemetry/cost.py`). Streamed calls take the **last** usage reading, never the sum:
providers report cumulative totals on the final chunk, and summing double-counts.

```json
"cost": {
  "turnUsd": 0.000412,
  "turnTokens": 1180,
  "steps": [{"step": "decompose", "usd": 0.00004}, {"step": "synthesise", "usd": 0.00031}],
  "entries": [
    {"step": "synthesise", "kind": "llm", "model": "gpt-4o-mini",
     "prompt_tokens": 1820, "completion_tokens": 140, "ms": 1450.0, "usd": 0.000357},
    {"step": "retrieve_rerank", "kind": "compute", "model": "3", "ms": 980.0, "usd": 0.0000136}
  ]
}
```

Local inference is priced by CPU time (`SLR_CPU_USD_PER_HOUR`) rather than reported as free,
so cost-per-turn does not flatter itself by ignoring the embedder, the reranker and the
verifier. Token prices are `SLR_PRICE_IN_PER_M` / `SLR_PRICE_OUT_PER_M`.

---

## 5. Reading it

```bash
curl localhost:8000/trace?limit=5 | jq '.traces[-1]'
curl localhost:8000/trace/t2 | jq '{mode, before_utterance_end, fusion, uncertainty}'
tail -f data/traces/trace.jsonl | jq -c '{turn: .turn_id, mode, lead: (.utterance_end_ms - .first_retrieval_ms)}'
```

The JSONL file is append-only and one object per line, so it can be shipped to any collector
without a vendor SDK in the hot path. An OTLP collector is available behind
`docker compose --profile observability up`; it is not on the default path because a judge
running the one-command start must not wait for a second image to boot.
