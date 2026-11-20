"""Embedding providers behind one interface.

`embed(texts, kind)` returns L2-normalised float vectors. `kind` is "document" or
"query"; some local models use different prefixes for each.
"""

import hashlib
import re
from functools import lru_cache

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from rag.config import get_settings


class Usage:
    """Process-wide token counter so the eval can report embedding cost."""

    embedding_tokens = 0


def _normalise(vectors: np.ndarray) -> list[list[float]]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32).tolist()


class FakeEmbedder:
    """Deterministic hashed bag-of-words. Only for tests and CI: no semantics, no API key."""

    def __init__(self, dim: int):
        self.dim = dim

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
                out[i, h % self.dim] += 1.0 if (h >> 63) else -1.0
        return _normalise(out)


class OpenAIEmbedder:
    def __init__(self, model: str, dim: int):
        from openai import OpenAI

        self.client, self.model, self.dim = OpenAI(), model, dim

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    def _call(self, batch: list[str]):
        return self.client.embeddings.create(model=self.model, input=batch, dimensions=self.dim)

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        resp = self._call(texts)
        Usage.embedding_tokens += resp.usage.total_tokens
        return _normalise(np.array([d.embedding for d in resp.data], dtype=np.float32))


class LocalEmbedder:
    """sentence-transformers model, e.g. BAAI/bge-small-en-v1.5 (dim 384). pip install sentence-transformers."""

    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)
        self.query_prefix = (
            "Represent this sentence for searching relevant passages: " if "bge" in model else ""
        )

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        if kind == "query" and self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        return _normalise(np.asarray(self.model.encode(texts, batch_size=32)))


@lru_cache
def get_embedder():
    s = get_settings()
    if s.embedding_provider == "openai":
        return OpenAIEmbedder(s.embedding_model, s.embedding_dim)
    if s.embedding_provider == "local":
        return LocalEmbedder(s.embedding_model)
    return FakeEmbedder(s.embedding_dim)


def embed_batched(texts: list[str], kind: str = "document") -> list[list[float]]:
    s, embedder = get_settings(), get_embedder()
    out: list[list[float]] = []
    for i in range(0, len(texts), s.embedding_batch_size):
        out.extend(embedder.embed(texts[i : i + s.embedding_batch_size], kind))
    return out
