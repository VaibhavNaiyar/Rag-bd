"""Every threshold the engine uses, in one place, each overridable by env.

Ablations flip these without a code change (``SLR_BRANCHES=dense``,
``SLR_CONTROLLER=model``, ``SLR_QUOTA_PER_INTENT=0``), so the benchmark report
and the running service can never disagree about what a knob means.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from functools import lru_cache
from pathlib import Path

try:  # .env is a convenience for local runs; the container passes real env.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, tuple):
        return tuple(part.strip() for part in raw.split(",") if part.strip())
    return raw


@dataclass(frozen=True)
class Settings:
    # --- paths -----------------------------------------------------------
    corpus_dir: str = str(ROOT / "evals" / "corpora" / "demo")
    index_dir: str = str(ROOT / "data" / "index")
    trace_path: str = str(ROOT / "data" / "traces" / "trace.jsonl")
    fixtures_dir: str = str(ROOT / "evals" / "fixtures")
    static_dir: str = str(Path(__file__).resolve().parent / "api" / "static")
    prompts_dir: str = str(ROOT / "prompts")

    # --- models ----------------------------------------------------------
    embedder: str = "auto"  # auto | bge | lsa
    embed_model: str = "BAAI/bge-small-en-v1.5"
    reranker: str = "auto"  # auto | cross | none
    # bge-reranker-base costs ~5.5s per 30 pairs on a 4-thread CPU; MiniLM-L6
    # costs ~0.7s. CPU default is MiniLM; set SLR_RERANK_MODEL on a GPU host.
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_max_length: int = 256
    #: logit models are squashed as sigmoid(logit / T) so the margin cut has room to work
    rerank_temperature: float = 4.0
    #: total cross-encoder pairs per call — bounds latency however many intents a turn has
    rerank_pair_budget: int = 48
    #: a provisional search is a guess made mid-utterance: it gets a smaller
    #: budget so it never delays the real searches behind the same model lock
    rerank_provisional_pairs: int = 12
    verifier: str = "auto"  # auto | nli | lexical
    nli_model: str = "cross-encoder/nli-deberta-v3-xsmall"
    llm: str = "auto"  # auto | openai | offline
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = ""
    llm_timeout_s: float = 30.0
    controller: str = "rule"  # rule | model

    # --- cost (USD) --------------------------------------------------------
    price_in_per_m: float = 0.15
    price_out_per_m: float = 0.60
    #: Local inference is not free; CPU time is priced so cost-per-turn is honest.
    cpu_usd_per_hour: float = 0.05

    # --- retrieval -------------------------------------------------------
    branches: tuple[str, ...] = ("bm25", "dense")
    branch_k: int = 50
    rrf_k: int = 60
    rerank_cap: int = 20
    rerank_margin: float = 0.15
    min_keep: int = 3
    per_query_keep: int = 6
    quota_per_intent: int = 2
    top_k: int = 8
    context_char_budget: int = 9000

    # --- chunking --------------------------------------------------------
    chunk_words: int = 160
    chunk_max_words: int = 240

    # --- controller ------------------------------------------------------
    tau_stab: float = 0.95
    stab_run: int = 2
    tau_shift: float = 0.70
    tau_refine: float = 0.60
    min_content_tokens: int = 4
    max_provisional: int = 2

    # --- decomposition ---------------------------------------------------
    max_subqueries: int = 4
    merge_cos: float = 0.90
    reuse_cos: float = 0.85
    #: shared context is carried between clauses only when they are about the
    #: same thing; otherwise 'Tokyo' leaks from one intent into an unrelated one
    carry_cos: float = 0.35

    # --- grounding / refinement ----------------------------------------------
    support_min: float = 0.5
    #: Bar for a citation the ENGINE picks, rather than one the model supplied.
    #: Higher on purpose: choosing a source for an uncited (or falsely cited)
    #: sentence is a stronger claim than checking one the model named.
    auto_cite_min: float = 0.75
    affect_margin: float = 0.08

    extra: dict = field(default_factory=dict, compare=False, hash=False)

    def with_overrides(self, **changes) -> "Settings":
        return replace(self, **changes)


def load_settings() -> Settings:
    base = Settings()
    values = {}
    for f in fields(Settings):
        if f.name == "extra":
            continue
        values[f.name] = _env(f"SLR_{f.name.upper()}", getattr(base, f.name))
    return Settings(**values)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def reset_settings() -> None:
    get_settings.cache_clear()
