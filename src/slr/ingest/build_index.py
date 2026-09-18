"""CLI: any corpus directory -> index.

    python -m slr.ingest.build_index --corpus /data/corpus --out /data/index

Content-addressed ids mean re-ingesting the same corpus rewrites the index in
place rather than duplicating it.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np

from slr.config import get_settings
from slr.ingest.chunker import chunk_document
from slr.ingest.loader import load_corpus
from slr.retrieval.embed import LsaEmbedder, load_bge, neural_available
from slr.retrieval.store import index_text

log = logging.getLogger("slr.ingest")


def build(corpus: str, out: str, embedder: str | None = None) -> dict:
    settings = get_settings()
    choice = (embedder or settings.embedder).lower()
    started = time.time()
    docs = load_corpus(corpus)
    if not docs:
        raise SystemExit(f"no ingestible documents under {corpus}")
    chunks = [c for d in docs for c in chunk_document(d, settings.chunk_words, settings.chunk_max_words)]
    texts = [index_text(c) for c in chunks]

    if choice == "auto":
        choice = "bge" if neural_available(settings.embed_model) else "lsa"

    tmp = Path(out + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    if choice == "lsa":
        model = LsaEmbedder.fit(texts)
        model.save(str(tmp / "lsa.npz"))
        name = "lsa"
    else:
        model = load_bge(settings.embed_model)
        name = "bge"
    vectors = model.embed(texts, kind="passage")
    np.save(tmp / "embeddings.npy", vectors.astype(np.float32))

    with (tmp / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    manifest = {
        "embedder": name,
        "embed_model": settings.embed_model if name == "bge" else "lsa",
        "dim": int(vectors.shape[1]),
        "docs": len(docs),
        "chunks": len(chunks),
        "built_at": int(time.time() * 1000),
        "build_seconds": round(time.time() - started, 2),
        "corpus_dir": str(Path(corpus).resolve()),
        "documents": [
            {"doc_id": d.doc_id, "label": d.label, "title": d.title, "source": d.source, "kind": d.kind}
            for d in docs
        ],
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Swap atomically-ish so a running server never loads a half-written index.
    dest = Path(out)
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)
    log.info("indexed %d docs / %d chunks with %s in %.1fs", len(docs), len(chunks), name, time.time() - started)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    settings = get_settings()
    parser.add_argument("--corpus", default=settings.corpus_dir)
    parser.add_argument("--out", default=settings.index_dir)
    parser.add_argument("--embedder", choices=["auto", "bge", "lsa"], default=None)
    parser.add_argument("--if-missing", action="store_true", help="skip when an index already exists")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.if_missing and (Path(args.out) / "manifest.json").exists():
        log.info("index already present at %s", args.out)
        return
    manifest = build(args.corpus, args.out, args.embedder)
    print(json.dumps({k: v for k, v in manifest.items() if k != "documents"}, indent=2))


if __name__ == "__main__":
    main()
