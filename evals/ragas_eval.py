"""RAGAS scores for a finished run — the standard RAG metrics, on our own golden set.

The gates in ``evals/gates.py`` measure what this theme asks for (early retrieval,
decomposition, grounding, refinement). RAGAS measures the same answers the way the RAG
literature does, with an LLM judge, so the two can be read against each other:

    faithfulness        is every statement in the answer supported by the retrieved context
    answer_relevancy    does the answer address the question that was asked
    context_precision   are the retrieved chunks that matter ranked first (the reranker's job)
    context_recall      does the retrieved context cover the reference answer
    answer_correctness  does the answer agree with ASQA's own long answer

Inputs are a run's ``turns.jsonl`` and ``evals/gold.jsonl``; nothing here imports ``slr``,
and it runs in its own virtualenv (``.venv-eval``) so the engine's pinned dependencies are
untouched:

    python -m venv .venv-eval
    .venv-eval/Scripts/pip install ragas langchain-openai datasets
    .venv-eval/Scripts/python evals/ragas_eval.py evals/results/turns.jsonl --limit 30

The judge costs money, so ``--limit`` caps the sample and the default judge is a small model.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "evals" / "gold.jsonl"
CHUNKS = {
    "asqa": ROOT / "data" / "index_asqa" / "chunks.jsonl",
    "enterprise": ROOT / "data" / "index" / "chunks.jsonl",
}
MARKER = re.compile(r"\[[^\[\]\n]*§[^\[\]\n]*\]")


def load_chunks(corpus: str) -> dict[str, str]:
    path = CHUNKS[corpus]
    if not path.exists():
        raise SystemExit(f"{path} not found — build the index first (make ingest / make dataset)")
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                out[row["chunk_id"]] = f"{row.get('heading', '')}: {row['text']}".strip(": ")
    return out


def load_gold() -> dict[str, dict[str, Any]]:
    """ASQA's own long answer per question, keyed by the question text."""
    out = {}
    with GOLD.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            answers = row.get("long_answers") or ([row["long_answer"]] if row.get("long_answer") else [])
            if answers:
                out[" ".join(row["ambiguous_question"].split()).lower()] = {
                    "reference": max(answers, key=len),
                    "sub_questions": [s["question"] for s in row.get("sub_questions", [])],
                }
            for sub in row.get("sub_questions", []):
                short = "; ".join(sub.get("short_answers") or [])
                if short:
                    out.setdefault(" ".join(sub["question"].split()).lower(), {"reference": short, "sub_questions": []})
    return out


def samples(run: Path, corpus: str, limit: int) -> list[dict[str, Any]]:
    chunks = load_chunks(corpus)
    gold = load_gold()
    rows = []
    with run.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("corpus", corpus) != corpus:
                continue
            trace = record["trace"]
            answer = (trace.get("answer") or {}).get("body") or ""
            if trace.get("mode") != "retrieve" or not answer.strip():
                continue
            question = " ".join(trace["utterance"].split())
            reference = gold.get(question.lower())
            contexts = [chunks[c] for c in (trace.get("fusion") or {}).get("chunk_ids", []) if c in chunks]
            if not reference or not contexts:
                continue
            rows.append(
                {
                    "user_input": question,
                    "response": MARKER.sub("", answer).strip(),
                    "retrieved_contexts": contexts,
                    "reference": reference["reference"],
                }
            )
            if len(rows) >= limit:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="a run's turns.jsonl")
    parser.add_argument("--corpus", default="asqa", choices=sorted(CHUNKS))
    parser.add_argument("--limit", type=int, default=30, help="questions to judge (each one costs judge tokens)")
    parser.add_argument("--judge", default="gpt-4.1-mini")
    parser.add_argument("--embeddings", default="text-embedding-3-small")
    parser.add_argument("--out", type=Path, default=ROOT / "evals" / "results" / "ragas.json")
    args = parser.parse_args()

    rows = samples(args.run, args.corpus, args.limit)
    if not rows:
        raise SystemExit("no comparable turns: the run and evals/gold.jsonl do not overlap")
    print(f"[ragas] {len(rows)} answered turns from {args.run.name}, judged by {args.judge}", flush=True)

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerCorrectness,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
        ResponseRelevancy,
    )

    judge = LangchainLLMWrapper(ChatOpenAI(model=args.judge, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=args.embeddings))
    metrics = [
        Faithfulness(llm=judge),
        ResponseRelevancy(llm=judge, embeddings=embeddings),
        ContextPrecision(llm=judge),
        ContextRecall(llm=judge),
        AnswerCorrectness(llm=judge, embeddings=embeddings),
    ]
    result = evaluate(EvaluationDataset.from_list(rows), metrics=metrics, show_progress=True)
    scores = {k: round(float(v), 4) for k, v in result._repr_dict.items()}
    payload = {
        "run": str(args.run),
        "corpus": args.corpus,
        "turns": len(rows),
        "judge": args.judge,
        "embeddings": args.embeddings,
        "scores": scores,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
