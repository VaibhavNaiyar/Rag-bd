"""Replay fixtures through the engine in-process and print a readable event log.

    python scripts/replay.py compound_01 late_detail_01 --speed 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from slr.config import get_settings
from slr.stream.engine import Engine, SessionRunner


def short(event: dict) -> str:
    t = event["type"]
    if t == "controller.decision":
        return f"  {event['atMs']:>6}ms decision {event['decision']:<8} {event['reason']} ({event.get('confidence')})"
    if t == "transcript.chunk":
        return f"  {event['atMs']:>6}ms chunk    {event['text']!r}"
    if t == "retrieval.started":
        return f"  {event['atMs']:>6}ms RETRIEVE {event['subQueryId']} [{event['trigger']}] {event.get('query')!r}"
    if t == "retrieval.cancelled":
        return f"          CANCEL   {event['subQueryId']} {event['reason']}"
    if t == "utterance.end":
        return f"  {event['atMs']:>6}ms --- utterance end ---"
    if t == "subqueries":
        return "          subqueries " + json.dumps([(i["id"], i["source"], i["text"]) for i in event["items"]])
    if t == "retrieval.result":
        return f"          result   {event['subQueryId']} cand={event['candidates']} kept={[h['citation'] for h in event['kept']]}"
    if t == "fusion.final":
        return f"          fusion   quota={event['quotaApplied']} full={event['fullCorpusSearch']} {[h['citation'] for h in event['hits']]}"
    if t == "answer.version":
        lines = [f"          VERSION v{event['version']} parent={event['parent']} support={event['citationSupportRate']} fabricated={event['fabricatedCitations']}"]
        lines.append(f"          preserved={event['preserved']} mutated={event['mutated']}")
        for u in event["uncertainty"]:
            lines.append(f"          UNCERTAIN {u}")
        return "\n".join(lines)
    if t == "turn.complete":
        return f"          complete latency={event['latencyMs']} cost=${event['cost']['turnUsd']} tokens={event['cost']['turnTokens']}"
    return f"          {t} {json.dumps({k: v for k, v in event.items() if k != 'type'})[:200]}"


async def run(names: list[str], speed: float) -> None:
    engine = Engine.from_settings(get_settings())
    print("models:", engine.models_info())
    for name in names:
        print(f"\n===== {name} =====")
        answer: dict[str, str] = {}

        async def emit(event: dict) -> None:
            if event["type"] == "answer.token":
                answer[event["turnId"]] = answer.get(event["turnId"], "") + event["text"]
                return
            if event["type"] == "turn.start":
                print(f"--- turn {event['turnId']}")
            print(short(event))
            if event["type"] == "answer.version":
                print("          ANSWER: " + answer.get(event["turnId"], "").replace("\n", "\n                  "))

        runner = SessionRunner(engine, emit)
        await runner.start()
        await runner.replay(name, speed)
        await runner.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("fixtures", nargs="+")
    p.add_argument("--speed", type=float, default=4.0)
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(run(a.fixtures, a.speed))
