"""Download the local models so the image needs no network at run time.

Run at build time by the Dockerfile; also useful locally before a first run.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slr.config import get_settings  # noqa: E402


def main() -> None:
    s = get_settings()
    from sentence_transformers import CrossEncoder, SentenceTransformer

    print(f"embedder   {s.embed_model}", flush=True)
    SentenceTransformer(s.embed_model, device="cpu").encode(["warm up"])

    print(f"reranker   {s.rerank_model}", flush=True)
    CrossEncoder(s.rerank_model, max_length=s.rerank_max_length, device="cpu").predict([("a", "b")])

    print(f"verifier   {s.nli_model}", flush=True)
    CrossEncoder(s.nli_model, device="cpu").predict([("a", "b")], apply_softmax=True)

    if s.nli_fallback_model:
        print(f"checker    {s.nli_fallback_model}", flush=True)
        CrossEncoder(s.nli_fallback_model, device="cpu").predict([("a", "b")], apply_softmax=True)

    print("models cached", flush=True)


if __name__ == "__main__":
    main()
