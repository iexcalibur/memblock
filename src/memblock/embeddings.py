"""Embedding providers — optional vector embedding layer for semantic search.

Supports:
  - Local embeddings via fastembed (no API key, runs on CPU)
  - OpenAI text-embedding-3-small (API key required)
  - Gemini text-embedding-004 (API key required)
  - Custom callable provider

Install:
  pip install memblock[embeddings]        # fastembed (local)
  pip install memblock[embeddings-openai]  # OpenAI API
  pip install memblock[embeddings-gemini]  # Gemini API
"""

from __future__ import annotations

import math
import struct
from abc import ABC, abstractmethod
from typing import Callable


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Convert texts to embedding vectors."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Number of dimensions in the embedding vectors."""
        ...


class FastEmbedProvider(EmbeddingProvider):
    """
    Local embedding provider using FastEmbed (ONNX Runtime).

    No API key required. Downloads model (~80MB) on first use.
    Default model: all-MiniLM-L6-v2 (384 dimensions, fast, good quality).
    """

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError(
                "FastEmbed is required for local embeddings. "
                "Install with: pip install memblock[embeddings]"
            )
        self._model_name = model
        self._model = TextEmbedding(model_name=model)
        # Detect dimensions from a test embed
        self._dims: int | None = None

    @property
    def dimensions(self) -> int:
        if self._dims is None:
            test = list(self._model.embed(["test"]))[0]
            self._dims = len(test)
        return self._dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        results = list(self._model.embed(texts))
        if self._dims is None and results:
            self._dims = len(results[0])
        return [list(r) for r in results]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI text-embedding-3-small (1536 dimensions)."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        self._api_key = api_key
        self._model = model
        self._dims = 1536

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        import httpx

        results: list[list[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = httpx.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self._model, "input": batch},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            results.extend(item["embedding"] for item in data["data"])
        return results


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Google Gemini text-embedding-004 (768 dimensions)."""

    def __init__(self, api_key: str, model: str = "text-embedding-004") -> None:
        self._api_key = api_key
        self._model = model
        self._dims = 768

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        import httpx

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:batchEmbedContents?key={self._api_key}"
        )
        results: list[list[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            body = {
                "requests": [
                    {
                        "model": f"models/{self._model}",
                        "content": {"parts": [{"text": t}]},
                    }
                    for t in batch
                ]
            }
            resp = httpx.post(url, json=body, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            for emb_obj in data.get("embeddings", []):
                results.append(emb_obj["values"])
        return results


class CallableEmbeddingProvider(EmbeddingProvider):
    """Wraps any function(texts) -> list[list[float]] as an embedding provider."""

    def __init__(self, fn: Callable[[list[str]], list[list[float]]], dims: int) -> None:
        self._fn = fn
        self._dims = dims

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._fn(texts)


# ─── Utility functions ────────────────────────────────────────────────────


def pack_embedding(values: list[float]) -> bytes:
    """Pack a list of floats into a compact bytes blob."""
    return struct.pack(f"{len(values)}f", *values)


def unpack_embedding(data: bytes) -> list[float]:
    """Unpack a bytes blob back into a list of floats."""
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (norm_a * norm_b)


def rrf_merge(
    fts_ids: list[str],
    vec_ids: list[str],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion — merge two ranked result lists.

    Returns list of (block_id, rrf_score) sorted by combined score.

    RRF score = sum( 1 / (k + rank) ) across lists where the item appears.
    k=60 is the standard constant (from the original RRF paper).
    """
    scores: dict[str, float] = {}
    for rank, block_id in enumerate(fts_ids):
        scores[block_id] = scores.get(block_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, block_id in enumerate(vec_ids):
        scores[block_id] = scores.get(block_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
