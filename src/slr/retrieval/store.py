"""A built index, loaded into memory.

Files under the index directory::

    manifest.json     embedder identity, counts, build time, document table
    chunks.jsonl      one Chunk per line
    embeddings.npy    float32, L2-normalised, row i == chunk i
    lsa.npz           only when the LSA fallback embedder was used
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from slr.contracts import Chunk
from slr.retrieval.embed import Embedder, LsaEmbedder, load_bge
from slr.text import content_tokens


def index_text(chunk: Chunk) -> str:
    """What the retrievers see: heading trail + body."""
    return f"{chunk.heading}\n{chunk.text}"


@dataclass
class Index:
    chunks: list[Chunk]
    embeddings: np.ndarray
    embedder: Embedder
    manifest: dict
    bm25: BM25Okapi = field(init=False)
    by_id: dict[str, int] = field(init=False)
    by_citation: dict[str, Chunk] = field(init=False)
    idf: dict[str, float] = field(init=False)
    df: dict[str, int] = field(init=False)
    salient_df: int = field(init=False)

    def __post_init__(self) -> None:
        tokenised = [content_tokens(index_text(c)) or ["_"] for c in self.chunks]
        self.bm25 = BM25Okapi(tokenised)
        self.by_id = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        self.by_citation = {c.citation.lower(): c for c in self.chunks}
        n = len(self.chunks)
        df: dict[str, int] = {}
        for toks in tokenised:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log((n + 1) / (d + 0.5)) for t, d in df.items()}
        self.df = df
        # Salient == the corpus does not use this word everywhere. An IDF
        # percentile fails on small corpora, where most terms are singletons.
        self.salient_df = max(2, int(n * 0.25))

    def chunk(self, chunk_id: str) -> Chunk:
        return self.chunks[self.by_id[chunk_id]]

    @property
    def doc_count(self) -> int:
        return int(self.manifest.get("docs", 0))

    def is_salient(self, token: str) -> bool:
        d = self.df.get(token)
        return d is not None and d <= self.salient_df


def load_index(index_dir: str | Path) -> Index:
    root = Path(index_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    with (root / "chunks.jsonl").open(encoding="utf-8") as fh:
        chunks = [Chunk.from_dict(json.loads(line)) for line in fh if line.strip()]
    embeddings = np.load(root / "embeddings.npy")
    if manifest["embedder"] == "lsa":
        embedder: Embedder = LsaEmbedder.load(str(root / "lsa.npz"))
    else:
        embedder = load_bge(manifest["embed_model"])
    if embeddings.shape[0] != len(chunks):
        raise ValueError("index is inconsistent: embeddings and chunks differ in length")
    return Index(chunks=chunks, embeddings=embeddings, embedder=embedder, manifest=manifest)
