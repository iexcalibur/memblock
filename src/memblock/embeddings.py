"""Embedding providers — optional vector embedding layer for semantic search.

Supports:
  - Local embeddings via fastembed (no API key, runs on CPU)
  - OpenAI text-embedding-3-small (API key required)
  - Gemini gemini-embedding-001 / gemini-embedding-2 (API key required)
  - Custom callable provider

Install:
  pip install memblock[embeddings]        # fastembed (local)
  pip install memblock[embeddings-openai]  # OpenAI API
  pip install memblock[embeddings-gemini]  # Gemini API
"""

from __future__ import annotations

import math
import struct
import sys
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
    """Google Gemini embedding provider.

    Supported models (all return 3072-dim vectors):
      - gemini-embedding-001    (default, GA, recommended for most workloads)
      - gemini-embedding-2      (newer GA model)
      - gemini-embedding-2-preview (preview channel — may change)

    Note: the prior-default `text-embedding-004` was removed by Google
    in 2026 and now returns HTTP 404. Existing callers that hardcoded
    that name should switch to `gemini-embedding-001`.
    """

    # Known default dimensions per model.
    _MODEL_DIMS: dict[str, int] = {
        "gemini-embedding-001": 3072,
        "gemini-embedding-2": 3072,
        "gemini-embedding-2-preview": 3072,
        # Legacy alias kept in the dict so existing callers that
        # passed the old name still get a usable default. The actual
        # API call will 404 — caller should migrate.
        "text-embedding-004": 768,
    }

    def __init__(
        self, api_key: str, model: str = "gemini-embedding-001",
    ) -> None:
        self._api_key = api_key
        self._model = model
        # Default to 3072 (the dim of every current Gemini embedding
        # model) when the user passes an unknown name.
        self._dims = self._MODEL_DIMS.get(model, 3072)

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


def _brute_force_similarity_python(
    query_vec: list[float],
    embeddings: list[tuple[str, bytes]],
) -> list[tuple[str, float]]:
    """Pure-Python reference path for `brute_force_similarity`.

    One `cosine_similarity` call per row; stable sort so ties keep
    their input order.
    """
    scored = []
    for block_id, blob in embeddings:
        score = cosine_similarity(query_vec, unpack_embedding(blob))
        if score != score:  # NaN (corrupt embedding): rank last, like the numpy path
            score = float("-inf")
        scored.append((block_id, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _brute_force_similarity_numpy(
    query_vec: list[float],
    embeddings: list[tuple[str, bytes]],
) -> list[tuple[str, float]] | None:
    """Vectorised path for `brute_force_similarity`.

    Returns None when numpy is unavailable or the input cannot be
    stacked into one matrix (mixed blob lengths, or a blob whose
    length does not match the query dimension); the caller then falls
    back to the pure-Python loop. numpy is an optional accelerator,
    never a hard dependency.
    """
    # `pack_embedding` uses struct '{n}f' — NATIVE float32 byte order.
    # We read the blobs back as explicit little-endian '<f4', which is
    # only equivalent on little-endian hosts; anything else takes the
    # portable Python path.
    if sys.byteorder != "little":
        return None
    try:
        import numpy as np
    except ImportError:
        return None

    dims = len(query_vec)
    if dims == 0:
        return None
    expected_len = dims * 4
    blobs = [blob for _, blob in embeddings]
    if any(len(blob) != expected_len for blob in blobs):
        return None

    # One contiguous buffer -> (n, dims) matrix. Equivalent to stacking a
    # per-row np.frombuffer, but a single allocation instead of n. We
    # widen to float64 so the arithmetic matches the Python reference
    # (which sums float32 values as Python floats) to ~1e-15, keeping
    # both paths' orderings identical.
    matrix = (
        np.frombuffer(b"".join(blobs), dtype="<f4")
        .reshape(len(blobs), dims)
        .astype(np.float64)
    )
    query = np.asarray(query_vec, dtype=np.float64)

    # Same zero-norm guard as `cosine_similarity`: a zero vector gets
    # norm 1.0 (score 0.0) rather than a division by zero.
    row_norms = np.sqrt(np.einsum("ij,ij->i", matrix, matrix))
    row_norms[row_norms == 0.0] = 1.0
    query_norm = float(np.sqrt(query @ query)) or 1.0

    scores = (matrix @ query) / (row_norms * query_norm)
    # NaN (corrupt embedding) ranks last on both paths.
    scores = np.where(np.isnan(scores), -np.inf, scores)

    # Descending, stable: ties keep input order exactly like the Python
    # path's `list.sort(reverse=True)`. `.tolist()` hands back plain
    # Python floats/ints in one shot (per-element numpy scalar indexing
    # was the single most expensive step at 10k rows).
    order = np.argsort(-scores, kind="stable").tolist()
    score_list = scores.tolist()
    ids = [block_id for block_id, _ in embeddings]
    return [(ids[i], score_list[i]) for i in order]


def brute_force_similarity(
    query_vec: list[float],
    embeddings: list[tuple[str, bytes]],
) -> list[tuple[str, float]]:
    """Cosine similarity of `query_vec` against every packed embedding.

    Returns ALL `(block_id, score)` pairs sorted by score descending
    (stable on ties). Uses a single numpy matmul when numpy is importable
    and every blob has the query's dimension; otherwise (numpy missing,
    big-endian host, or mixed-length blobs) it runs the pure-Python
    `cosine_similarity` loop. Both paths agree to within 1e-5 and
    produce the same ordering.

    This is the Python-side fallback used by the query engines when the
    storage adapter has no server-side vector search (i.e. not pgvector).
    """
    if not embeddings:
        return []
    result = _brute_force_similarity_numpy(query_vec, embeddings)
    if result is None:
        result = _brute_force_similarity_python(query_vec, embeddings)
    return result


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


def weighted_rrf_merge(
    fts_ids: list[str],
    vec_ids: list[str],
    vec_scores: dict[str, float] | None = None,
    k: int = 60,
    fts_weight: float = 0.4,
    vec_weight: float = 0.6,
) -> list[tuple[str, float]]:
    """
    Weighted Reciprocal Rank Fusion — merge FTS and vector results with tunable weights.

    Returns list of (block_id, rrf_score) sorted by combined score.
    Also blends raw cosine similarity when available for better scoring.
    """
    scores: dict[str, float] = {}
    for rank, block_id in enumerate(fts_ids):
        scores[block_id] = scores.get(block_id, 0.0) + fts_weight / (k + rank + 1)
    for rank, block_id in enumerate(vec_ids):
        scores[block_id] = scores.get(block_id, 0.0) + vec_weight / (k + rank + 1)

    # Blend raw cosine similarity if available (normalized to same scale)
    if vec_scores:
        max_rrf = max(scores.values()) if scores else 1.0
        for block_id, cos_sim in vec_scores.items():
            # Add cosine similarity as a bonus (scaled to ~half of max RRF)
            scores[block_id] = scores.get(block_id, 0.0) + cos_sim * max_rrf * 0.5

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
