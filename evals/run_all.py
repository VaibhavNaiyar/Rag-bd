"""The whole benchmark, unattended.

    python -m evals.run_all              # gates on both dev corpora + ablations
    python -m evals.run_all --quick      # enterprise corpus only, no ablations
    python -m evals.run_all --no-ablations

Writes evals/results/latest.json and docs/BENCHMARK_REPORT.md.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from evals import report as report_mod
from evals.ablations import compare_baseline, run_ablations
from evals.gates import AUDIT_NLI, all_gates, latency_and_cost
from evals.harness import (
    build_engine,
    ensure_asqa_corpus,
    ensure_index,
    load_fixtures,
    run,
    settings_for,
)
from slr.config import get_settings, reset_settings
from slr.synthesis.grounding import load_nli

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evals" / "results"

ENTERPRISE_FAMILIES = ["compound", "late_detail", "suppression", "single", "unanswerable"]
ASQA_FAMILIES = ["compound", "late_detail", "suppression", "single"]


def edge_cases(runs: dict[str, Any]) -> list[dict[str, Any]]:
    """Real failures, taken from the runs that just happened. Nothing invented."""
    found: list[dict[str, Any]] = []
    for corpus, result in runs.items():
        for turn in result.turns:
            trace = turn.trace
            utterance = trace["utterance"]
            fusion = trace.get("fusion") or {}
            answer = trace.get("answer") or {}
            decomposition = trace.get("decomposition") or {}

            if trace["mode"] in ("retrieve", "refine") and not trace.get("before_utterance_end"):
                found.append(
                    {
                        "category": "late_trigger",
                        "corpus": corpus,
                        "fixture": turn.fixture,
                        "utterance": utterance,
                        "what_happened": "No provisional search fired before the speaker finished; retrieval began at the utterance end.",
                        "evidence": {
                            "first_retrieval_ms": trace["first_retrieval_ms"],
                            "utterance_end_ms": trace["utterance_end_ms"],
                            "decisions": [(d["decision"], d["reason"]) for d in trace["decisions"]],
                        },
                    }
                )

            starved = [
                s["text"]
                for s in trace["sub_queries"]
                if s["source"] == "decomposed" and (fusion.get("per_sub_query") or {}).get(s["id"], 0) == 0
            ]
            if starved:
                found.append(
                    {
                        "category": "starved_sub_intent",
                        "corpus": corpus,
                        "fixture": turn.fixture,
                        "utterance": utterance,
                        "what_happened": "A sub-intent reached the answer with no evidence of its own.",
                        "evidence": {"sub_queries": starved, "quota_applied": fusion.get("quota_applied")},
                    }
                )

            gold = turn.expect.get("gold_sub_intents") or []
            predicted = [s["text"] for s in trace["sub_queries"] if s["source"] == "decomposed"]
            if len(gold) >= 2 and len(predicted) < len(gold):
                found.append(
                    {
                        "category": "under_decomposition",
                        "corpus": corpus,
                        "fixture": turn.fixture,
                        "utterance": utterance,
                        "what_happened": f"{len(gold)} gold sub-intents, {len(predicted)} sub-queries produced.",
                        "evidence": {
                            "gold": gold,
                            "predicted": predicted,
                            "merged_by_guard": decomposition.get("merged"),
                            "method": decomposition.get("method"),
                        },
                    }
                )
            if len(predicted) > max(len(gold), 1) and gold:
                found.append(
                    {
                        "category": "over_fragmentation",
                        "corpus": corpus,
                        "fixture": turn.fixture,
                        "utterance": utterance,
                        "what_happened": f"{len(predicted)} sub-queries for {len(gold)} gold sub-intents.",
                        "evidence": {"gold": gold, "predicted": predicted},
                    }
                )

            demoted = (answer.get("grounding") or {}).get("demoted_claims", 0)
            if demoted:
                found.append(
                    {
                        "category": "claim_withheld",
                        "corpus": corpus,
                        "fixture": turn.fixture,
                        "utterance": utterance,
                        "what_happened": f"{demoted} generated claim(s) failed verification and were withheld.",
                        "evidence": {
                            "uncertainty": trace["uncertainty"][:3],
                            "support_rate": trace["citation_support_rate"],
                            "verifier": (answer.get("grounding") or {}).get("verifier"),
                        },
                    }
                )

            if trace["mode"] == "suppress" and trace["retrieval_events"]:
                found.append(
                    {
                        "category": "false_trigger",
                        "corpus": corpus,
                        "fixture": turn.fixture,
                        "utterance": utterance,
                        "what_happened": "A presentation-only turn issued a vector query before suppression was detected.",
                        "evidence": {"retrieval_events": trace["retrieval_events"][:2]},
                    }
                )
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="enterprise corpus only, no ablations")
    parser.add_argument("--no-ablations", action="store_true")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="replay speed. 1.0 is speaking pace and is what the reported numbers use: "
        "compressing time shrinks the gap between the last word and the end-of-utterance "
        "signal, which is exactly what G2 measures.",
    )
    parser.add_argument("--out", default=str(RESULTS / "latest.json"))
    args = parser.parse_args()

    started = time.time()
    reset_settings()
    get_settings()
    steps: dict[str, Any] = {"manual_steps": [], "command": "make eval"}

    corpora = ["enterprise"] if args.quick else ["enterprise", "asqa"]
    runs, gates, indexes, played = {}, {}, {}, {}
    for corpus in corpora:
        if corpus == "asqa":
            steps.update(ensure_asqa_corpus())
        settings = settings_for(corpus)
        existed = (Path(settings.index_dir) / "manifest.json").exists()
        ensure_index(settings)
        steps[f"index_{corpus}"] = "reused" if existed else "built by the harness"
        engine = build_engine(settings)
        indexes[corpus] = engine.index
        families = ENTERPRISE_FAMILIES if corpus == "enterprise" else ASQA_FAMILIES
        fixtures = load_fixtures(corpus, families)
        if not fixtures:
            steps["manual_steps"].append(f"no fixtures for corpus {corpus}")
            continue
        print(f"[eval] {corpus}: {len(fixtures)} fixtures", flush=True)
        played[corpus] = (settings, fixtures)
        runs[corpus] = run(engine, fixtures, args.speed)
        print(f"[eval] {corpus}: {len(runs[corpus].turns)} turns in {runs[corpus].seconds:.0f}s", flush=True)

    steps["index_built_by_harness"] = True
    steps["fixtures_ran"] = sum(len(r.turns) for r in runs.values())

    try:
        auditor = load_nli(AUDIT_NLI)
    except Exception as exc:  # no copy of the audit model on this machine
        print(f"[eval] shipped-claim audit skipped: {exc}", flush=True)
        auditor = None
    for corpus, result in runs.items():
        index = indexes[corpus]
        gates[corpus] = [g.as_dict() for g in all_gates(result, index, index.embedder, steps, auditor)]

    baseline: dict[str, Any] = {}
    for corpus, (settings, fixtures) in played.items():
        print(f"[eval] baseline pipeline: {corpus}", flush=True)
        baseline[corpus] = compare_baseline(runs[corpus], settings, fixtures, indexes[corpus], args.speed)

    ablations: list[dict[str, Any]] = []
    if not args.quick and not args.no_ablations:
        print("[eval] ablations", flush=True)
        asqa = settings_for("asqa")
        fixtures = load_fixtures("asqa", ["compound"]) + load_fixtures("enterprise", ["compound"])
        ablations = run_ablations(
            asqa, fixtures, indexes["asqa"], args.speed, has_llm=bool(os.environ.get("OPENAI_API_KEY"))
        )

    payload = {
        "generated_at": int(time.time() * 1000),
        "seconds": round(time.time() - started, 1),
        "engine": {
            "models": next(iter(runs.values())).models if runs else {},
            "knobs": next(iter(runs.values())).settings if runs else {},
            "llm_configured": bool(os.environ.get("OPENAI_API_KEY")),
            "replay_speed": args.speed,
        },
        "gates": gates,
        "performance": {c: latency_and_cost(r) for c, r in runs.items()},
        "baseline": baseline,
        "ablations": ablations,
        "edge_cases": edge_cases(runs),
        "corpora": {
            c: {"documents": indexes[c].doc_count, "chunks": len(indexes[c].chunks)} for c in runs
        },
        "fixtures": {c: sorted({t.fixture for t in r.turns}) for c, r in runs.items()},
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # Every turn's full record, so a reader can recompute any number in the
    # report without re-running the suite.
    turns_path = out.parent / "turns.jsonl"
    with turns_path.open("w", encoding="utf-8") as fh:
        for corpus, result in runs.items():
            for turn in result.turns:
                fh.write(
                    json.dumps(
                        {"corpus": corpus, "fixture": turn.fixture, "family": turn.family,
                         "expect": turn.expect, "trace": turn.trace},
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )
    report_path = report_mod.write_report(payload)

    print(f"\n[eval] results -> {out}")
    print(f"[eval] report  -> {report_path}\n")
    for corpus, rows in gates.items():
        for g in rows:
            verdict = "PASS" if g["passed"] else ("n/a " if not g.get("applicable", True) else "FAIL")
            print(f"  {corpus:5} {g['id']} {g['name']:<28} {str(g['value']):>7}  {verdict}")
    failed = [g for rows in gates.values() for g in rows if g.get("applicable", True) and not g["passed"]]
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
