"""The per-turn trace record, exported as OpenTelemetry spans (gate G6 dashboards).

Off unless ``SLR_OTEL_ENDPOINT`` is set (``docker compose --profile
observability up`` runs a collector at ``http://otel:4318``), and a no-op with
a log line if the OpenTelemetry SDK is not installed: the JSONL trace stays
the system of record either way, and this is a second sink for it.

One turn becomes one trace, rebuilt from the record's own timings:

    turn                    started_at … complete
    ├─ listen               0 … utterance end          events: every controller decision
    ├─ plan                 utterance end … + decomposition time
    ├─ retrieve             first retrieval … first token   events: every search, with its trigger
    └─ synthesise           … first token … complete   event: first_token

The payload rule is strict, because a collector is a third party: ids, counts,
timings, outcomes, costs and model names. Never the utterance, a query, chunk
text or the answer.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

SERVICE_NAME = "streaming-live-rag"


def _ns(started_at_ms: int, offset_ms: float | None) -> int:
    """Epoch nanoseconds, in integers: an epoch in float nanoseconds loses the low digits."""
    return int(started_at_ms) * 1_000_000 + round((offset_ms or 0) * 1_000_000)


def turn_attributes(record: dict[str, Any]) -> dict[str, Any]:
    """Scalar facts about a turn. Nothing here may carry text a user or a document wrote."""
    latency = record.get("latency_ms") or {}
    cost = record.get("cost") or {}
    answer = record.get("answer") or {}
    fusion = record.get("fusion") or {}
    attrs: dict[str, Any] = {
        "slr.session_id": record["session_id"],
        "slr.turn_id": record["turn_id"],
        "slr.mode": record.get("mode") or "unknown",
        "slr.controller": record.get("controller") or "",
        "slr.chunks": len(record.get("chunks") or []),
        "slr.sub_queries": len(record.get("sub_queries") or []),
        "slr.searches": len(record.get("retrieval_events") or []),
        "slr.before_utterance_end": bool(record.get("before_utterance_end")),
        "slr.claims": int(answer.get("claim_count") or 0),
        "slr.uncertain": len(record.get("uncertainty") or []),
        "slr.fabricated_citations": int(record.get("fabricated_citations") or 0),
        "slr.full_corpus_search": bool(fusion.get("full_corpus_search")),
        "slr.flagged_chunks": len(fusion.get("flagged_chunk_ids") or []),
        "slr.cost_usd": float(cost.get("turnUsd") or 0.0),
        "slr.tokens": int(cost.get("turnTokens") or 0),
        "slr.errors": len(record.get("errors") or []),
    }
    optional = {
        "slr.first_retrieval_rel_end_ms": latency.get("first_retrieval_rel_end"),
        "slr.ttft_ms": latency.get("first_token_after_end"),
        "slr.complete_ms": latency.get("complete_after_end"),
        "slr.citation_support_rate": record.get("citation_support_rate"),
    }
    attrs.update({k: v for k, v in optional.items() if v is not None})
    degraded = [d["step"] for d in record.get("degraded") or []]
    if degraded:
        attrs["slr.degraded_steps"] = degraded
    models = [m.get("model", "") for m in cost.get("models") or []]
    if models:
        attrs["gen_ai.request.model"] = models[0]
    return attrs


class OtelExporter:
    """Turns finished trace records into spans on an OpenTelemetry tracer provider."""

    def __init__(self, provider: Any) -> None:
        from opentelemetry import trace

        self._trace = trace
        self._provider = provider
        self._tracer = provider.get_tracer("slr.telemetry")

    @classmethod
    def from_settings(cls, endpoint: str) -> OtelExporter | None:
        if not endpoint:
            return None
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError:
            log.warning("SLR_OTEL_ENDPOINT is set but the OpenTelemetry SDK is not installed; spans are off")
            return None
        provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")))
        log.info("exporting turn traces to %s", endpoint)
        return cls(provider)

    def export(self, record: dict[str, Any]) -> None:
        try:
            self._export(record)
        except Exception:  # noqa: BLE001 - telemetry must never break a turn
            log.exception("span export failed for turn %s", record.get("turn_id"))

    def _export(self, record: dict[str, Any]) -> None:
        from opentelemetry.trace import Status, StatusCode

        t0 = int(record["started_at"])
        latency = record.get("latency_ms") or {}
        end_ms = record.get("utterance_end_ms")
        done_ms = latency.get("complete_abs", end_ms or 0)
        first_token = latency.get("first_token_abs")
        first_search = record.get("first_retrieval_ms")
        plan_ms = (record.get("decomposition") or {}).get("ms")

        root = self._tracer.start_span("turn", start_time=_ns(t0, 0), attributes=turn_attributes(record))
        ctx = self._trace.set_span_in_context(root)

        def child(name: str, start: float | None, end: float | None):
            if start is None or end is None:
                return None
            return self._tracer.start_span(name, context=ctx, start_time=_ns(t0, start)), end

        opened = []
        if end_ms is not None:
            listen = child("listen", 0, end_ms)
            if listen:
                for d in record.get("decisions") or []:
                    listen[0].add_event(
                        "decision",
                        {"decision": d["decision"], "reason": d["reason"], "confidence": float(d["confidence"])},
                        timestamp=_ns(t0, d["at_ms"]),
                    )
                opened.append(listen)
            if plan_ms is not None:
                opened.append(child("plan", end_ms, end_ms + plan_ms))
        if first_search is not None:
            retrieve = child("retrieve", first_search, first_token if first_token is not None else done_ms)
            if retrieve:
                for r in record.get("retrieval_events") or []:
                    retrieve[0].add_event(
                        "search",
                        {"sub_query_id": r["sub_query_id"], "trigger": r["trigger"],
                         "before_utterance_end": bool(r.get("before_utterance_end"))},
                        timestamp=_ns(t0, r["at_ms"]),
                    )
                opened.append(retrieve)
        if first_token is not None:
            start = (end_ms + plan_ms) if end_ms is not None and plan_ms is not None else first_token
            synth = child("synthesise", min(start, first_token), done_ms)
            if synth:
                synth[0].add_event("first_token", timestamp=_ns(t0, first_token))
                opened.append(synth)

        for item in opened:
            if item:
                span, end = item
                span.end(end_time=_ns(t0, end))
        if record.get("errors"):
            root.set_status(Status(StatusCode.ERROR, f"{len(record['errors'])} error(s)"))
        root.end(end_time=_ns(t0, done_ms))

    def shutdown(self) -> None:
        self._provider.shutdown()
