"""Reranker support — improve retrieval quality by reranking search results.

Supports:
  - BM25Reranker: Pure Python BM25 scoring (no dependencies)
  - CohereReranker: Cohere rerank API (requires: pip install cohere)
  - CrossEncoderReranker: HuggingFace cross-encoder (requires: pip install sentence-transformers)
  - CallableReranker: Wrap any function as a reranker

Usage:
    from memblock import MemBlock
    from memblock.rerankers import BM25Reranker, CohereReranker

    # BM25 — zero dependencies
    mem = MemBlock(storage="sqlite:///./memory.db", reranker=BM25Reranker())

    # Cohere — best quality
    mem = MemBlock(
        storage="sqlite:///./memory.db",
        reranker=CohereReranker(api_key="co-...")
    )
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from typing import Any, Callable

from memblock.block import Block


class Reranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    def rerank(self, query: str, blocks: list[Block], top_k: int = 10) -> list[Block]:
        """
        Rerank a list of blocks by relevance to the query.

        Args:
            query: The search query
            blocks: Candidate blocks to rerank
            top_k: Maximum results to return

        Returns:
            Reranked list of blocks (most relevant first).
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Reranker identifier."""
        ...


class BM25Reranker(Reranker):
    """
    Pure Python BM25 reranker — zero external dependencies.

    Uses BM25 (Best Matching 25) scoring to rerank blocks by term
    relevance. Good baseline that works without API calls.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """
        Args:
            k1: Term frequency saturation parameter (default 1.5)
            b: Length normalization parameter (default 0.75)
        """
        self._k1 = k1
        self._b = b

    @property
    def name(self) -> str:
        return "bm25"

    def rerank(self, query: str, blocks: list[Block], top_k: int = 10) -> list[Block]:
        if not blocks or not query:
            return blocks[:top_k]

        query_terms = self._tokenize(query)
        if not query_terms:
            return blocks[:top_k]

        # Build corpus stats
        docs = [self._tokenize(b.content) for b in blocks]
        avg_dl = sum(len(d) for d in docs) / len(docs) if docs else 1.0
        n = len(docs)

        # Document frequency for each query term
        df: dict[str, int] = {}
        for term in query_terms:
            df[term] = sum(1 for doc in docs if term in doc)

        # Score each document
        scores: list[tuple[int, float]] = []
        for i, doc in enumerate(docs):
            score = 0.0
            dl = len(doc)
            tf_map: dict[str, int] = {}
            for term in doc:
                tf_map[term] = tf_map.get(term, 0) + 1

            for term in query_terms:
                if term not in tf_map:
                    continue
                tf = tf_map[term]
                idf = math.log((n - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
                tf_norm = (tf * (self._k1 + 1)) / (
                    tf + self._k1 * (1 - self._b + self._b * dl / avg_dl)
                )
                score += idf * tf_norm

            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [blocks[i] for i, _ in scores[:top_k]]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercase tokenization."""
        return re.findall(r"\w+", text.lower())


class CohereReranker(Reranker):
    """
    Cohere rerank API — high-quality neural reranking.

    Requires: pip install memblock[reranker-cohere]
    """

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-v3.5",
        top_k: int | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._default_top_k = top_k

    @property
    def name(self) -> str:
        return f"cohere/{self._model}"

    def rerank(self, query: str, blocks: list[Block], top_k: int = 10) -> list[Block]:
        if not blocks or not query:
            return blocks[:top_k]

        try:
            import httpx
        except ImportError:
            raise ImportError(
                "CohereReranker requires httpx. Install with: pip install httpx"
            )

        documents = [b.content for b in blocks]
        effective_top_k = self._default_top_k or top_k

        resp = httpx.post(
            "https://api.cohere.com/v2/rerank",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "query": query,
                "documents": documents,
                "top_n": effective_top_k,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        # Map results back to blocks
        reranked: list[Block] = []
        for result in data.get("results", []):
            idx = result["index"]
            if 0 <= idx < len(blocks):
                reranked.append(blocks[idx])

        return reranked


class CrossEncoderReranker(Reranker):
    """
    Cross-encoder reranker using sentence-transformers.

    Requires: pip install memblock[reranker-cross-encoder]
    """

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "CrossEncoderReranker requires sentence-transformers. "
                "Install with: pip install memblock[reranker-cross-encoder]"
            )
        self._model_name = model
        self._model = CrossEncoder(model)

    @property
    def name(self) -> str:
        return f"cross-encoder/{self._model_name}"

    def rerank(self, query: str, blocks: list[Block], top_k: int = 10) -> list[Block]:
        if not blocks or not query:
            return blocks[:top_k]

        pairs = [[query, b.content] for b in blocks]
        scores = self._model.predict(pairs)

        scored = list(zip(blocks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [b for b, _ in scored[:top_k]]


class CallableReranker(Reranker):
    """
    Wrap any function as a reranker.

    Usage:
        def my_reranker(query: str, blocks: list[Block], top_k: int) -> list[Block]:
            # custom logic
            return sorted_blocks

        reranker = CallableReranker(fn=my_reranker, name="custom")
    """

    def __init__(
        self,
        fn: Callable[[str, list[Block], int], list[Block]],
        reranker_name: str = "custom",
    ) -> None:
        self._fn = fn
        self._name = reranker_name

    @property
    def name(self) -> str:
        return self._name

    def rerank(self, query: str, blocks: list[Block], top_k: int = 10) -> list[Block]:
        return self._fn(query, blocks, top_k)
