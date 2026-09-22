# Architecture, end to end

Three diagrams of the same system: the turn pipeline, the timeline of one utterance, and the
three routes a turn can take. Every box is a module in `src/slr/`; the numbers are measured
in [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md).

## 1. The pipeline

```mermaid
flowchart TB
    mic["Speaker<br/><i>transcript chunks, 3-5 words</i>"] -->|WS /stream| ctl

    subgraph stage1["1 · Retrieval controller — reads the transcript as it arrives"]
        ctl["Per chunk: semantic stability, content sufficiency,<br/>clause boundary, correction cues, presentation cues"]
        ctl --> verdict{"wait · retrieve<br/>suppress · refine"}
    end

    verdict -->|retrieve, mid-utterance| prov["Provisional search<br/><i>~260 ms before the speaker stops</i>"]
    verdict -->|suppress| restate
    verdict -->|refine| delta

    prov -.->|cancelled when the intent shifts| ctl

    subgraph stage2["2 · End of speech — the decomposer and the answer start together"]
        dec["Multi-intent decomposer (gpt-4.1)<br/>8 nearest split + 4 nearest unsplit examples<br/>merge guard, cap 4"]
        spec["Speculative answer from the provisional evidence<br/><i>kept when the decomposer finds one reading<br/>that reuses that search</i>"]
    end

    prov --> spec
    verdict -->|end of utterance| dec
    dec --> reuse{"one reading<br/>reusing the search?"}
    reuse -->|yes| keep["keep the answer already streaming"]
    reuse -->|no: several readings| fresh

    subgraph stage3["3 · Retrieval and fusion — per reading, plus the utterance as spoken"]
        fresh["BM25 + dense, per reading<br/>and one search for the whole utterance"]
        fresh --> rrf["RRF fusion"]
        rrf --> rr["Cross-encoder rerank<br/>MiniLM-L6, length-sorted batches"]
        rr --> cut["Margin cut (0.3)"]
        cut --> quota["Per-intent quota (3)<br/><i>no reading is starved</i>"]
        quota --> pack["Evidence package<br/>&lt;document&gt; fences, instruction-like text flagged"]
    end

    keep --> synth
    pack --> synth

    subgraph stage4["4 · Grounded synthesis — validated sentence by sentence"]
        synth["Answer model (gpt-5.4-mini)<br/>quotes its evidence per reading first"]
        synth --> split["Sentence splitter<br/><i>abbreviations and list numbers kept whole</i>"]
        split --> marker{"citation<br/>resolves?"}
        marker -->|no| strip["stripped and counted<br/><b>0 fabricated citations ship</b>"]
        marker -->|yes| nli["NLI check (deberta-v3-xsmall)"]
        strip --> attr
        nli -->|below 0.75| mini["MiniCheck-RoBERTa-Large<br/><i>second reading</i>"]
        nli -->|supported| figures
        mini -->|supported| figures
        mini -->|not supported| attr{"another retrieved block,<br/>best 6 by overlap?"}
        attr -->|yes, above 0.75| figures
        attr -->|no| held["withheld → uncertainty"]
        figures{"every figure in<br/>the cited block?"} -->|yes| ship["streamed to the reader"]
        figures -->|no| attr
    end

    subgraph stage5["5 · Session — ephemeral, one conversation"]
        store["Topic: utterance, sub-queries,<br/>evidence, answer version"]
    end

    ship --> store
    held --> unc["Uncertainty / 'which did you mean?'"]
    unc --> store

    delta["Refinement planner<br/><i>delta queries only</i>"] --> deltasearch["Search the delta"]
    deltasearch --> union["Session evidence ∪ delta<br/><b>never a full-corpus search</b>"]
    union --> ops["KEEP / EDIT / ADD / DROP per claim"]
    ops --> store
    restate["Restructure from the previous answer<br/><b>no retrieval at all</b>"] --> store

    store --> out["AG-UI events → console<br/>answer, citations, evidence, timeline"]
    store --> trace["Trace record per turn"]
    trace --> sinks["JSONL · OpenTelemetry spans + metrics → Grafana"]
```

## 2. One turn on the clock

```mermaid
sequenceDiagram
    autonumber
    participant S as Speaker
    participant C as Controller
    participant R as Retrieval
    participant D as Decomposer
    participant A as Answer model
    participant V as Verifier

    S->>C: "I need to plan a customer workshop in Pune…"
    C->>C: wait — intent still unstable
    S->>C: "…for 30 people, and I need…"
    C->>R: retrieve (clause complete)
    Note over R: provisional search runs while the speaker talks
    S->>C: "…the cancellation policy and the catering options."
    S->>C: [end of speech]
    par decomposer and answer start together
        C->>D: split the utterance
        C->>A: start answering from the provisional evidence
    end
    D-->>C: 3 readings — the speculative answer is dropped
    C->>R: one search per reading + one for the utterance
    R-->>A: fused, reranked, quota-balanced evidence
    loop per sentence
        A->>V: sentence + its citation
        V-->>A: supported → stream it
        V-->>A: unsupported → withhold, list as uncertainty
    end
    A-->>S: grounded answer, every claim cited
```

## 3. The three routes

```mermaid
flowchart LR
    u["utterance ends"] --> q{"controller"}
    q -->|a question| r["RETRIEVE<br/>decompose → search → answer<br/><i>G2, G3, G4</i>"]
    q -->|a late detail<br/>'I meant international'| f["REFINE<br/>delta search, claims updated in place<br/><i>G5 · 100%</i>"]
    q -->|'say that in two bullets'| p["SUPPRESS<br/>restructure the previous answer<br/><i>0 searches</i>"]
```
