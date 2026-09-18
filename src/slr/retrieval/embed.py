"""Embedder protocol with a local neural model and an offline fallback.

``bge-small-en-v1.5`` is the benchmark embedder. ``LsaEmbedder`` (TF-IDF +
truncated SVD, fitted on the corpus at ingest) needs no download, so a
development machine without network still runs the full pipeline.

Vectors are L2-normalised, so cosine similarity is a dot product.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import Counter
from functools import lru_cache
from typing import Protocol

import numpy as np

from slr.text import content_tokens

log = logging.getLogger(__name__)

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        """kind: passage | query | text (no instruction prefix)."""


def _normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


class BgeEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    def embed(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out: list[np.ndarray | None] = [self._cache.get((kind, t)) for t in texts]
        todo = [i for i, v in enumerate(out) if v is None]
        if todo:
            prefix = BGE_QUERY_PREFIX if kind == "query" else ""
            batch = [prefix + texts[i] for i in todo]
            with self._lock:
                vecs = self._model.encode(batch, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
            for i, v in zip(todo, vecs):
                out[i] = v.astype(np.float32)
                if kind != "passage" and len(self._cache) < 20000:
                    self._cache[(kind, texts[i])] = out[i]
        return np.stack(out)  # type: ignore[arg-type]


class LsaEmbedder:
    """TF-IDF + truncated SVD. Deterministic, dependency-light, offline."""

    name = "lsa"

    def __init__(self, vocab: dict[str, int], idf: np.ndarray, components: np.ndarray):
        self.vocab = vocab
        self.idf = idf.astype(np.float32)
        self.components = components.astype(np.float32)  # (k, V)
        self.dim = int(components.shape[0])

    @classmethod
    def fit(cls, texts: list[str], dim: int = 256) -> "LsaEmbedder":
        docs = [Counter(content_tokens(t)) for t in texts]
        df: Counter[str] = Counter()
        for d in docs:
            df.update(d.keys())
        vocab = {w: i for i, w in enumerate(sorted(w for w, c in df.items() if c >= 1))}
        n = max(1, len(docs))
        idf = np.zeros(len(vocab), dtype=np.float32)
        for w, i in vocab.items():
            idf[i] = math.log((1 + n) / (1 + df[w])) + 1
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import svds

        rows, cols, vals = [], [], []
        for r, d in enumerate(docs):
            for w, c in d.items():
                rows.append(r)
                cols.append(vocab[w])
                vals.append((1 + math.log(c)) * idf[vocab[w]])
        mat = csr_matrix((vals, (rows, cols)), shape=(n, max(1, len(vocab))), dtype=np.float32)
        k = max(1, min(dim, min(mat.shape) - 1))
        if min(mat.shape) <= 2:
            components = np.eye(1, max(1, len(vocab)), dtype=np.float32)
        else:
            _, _, vt = svds(mat, k=k, random_state=0)
            components = vt[::-1]
        return cls(vocab, idf, components)

    def _tfidf(self, text: str) -> np.ndarray:
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for w, c in Counter(content_tokens(text)).items():
            i = self.vocab.get(w)
            if i is not None:
                vec[i] = (1 + math.log(c)) * self.idf[i]
        return vec

    def embed(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        mat = np.stack([self._tfidf(t) for t in texts])
        return _normalise(mat @ self.components.T)

    def save(self, path: str) -> None:
        words = np.array(sorted(self.vocab, key=self.vocab.get))
        np.savez_compressed(path, words=words, idf=self.idf, components=self.components)

    @classmethod
    def load(cls, path: str) -> "LsaEmbedder":
        data = np.load(path, allow_pickle=False)
        vocab = {str(w): i for i, w in enumerate(data["words"])}
        return cls(vocab, data["idf"], data["components"])


@lru_cache(maxsize=4)
def load_bge(model_name: str) -> BgeEmbedder:
    return BgeEmbedder(model_name)


def neural_available(model_name: str) -> bool:
    try:
        load_bge(model_name)
        return True
    except Exception as exc:  # no network, no torch, no cache
        log.warning("neural embedder unavailable (%s); using LSA fallback", exc)
        return False
