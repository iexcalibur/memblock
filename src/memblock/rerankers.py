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


class HeuristicReranker(Reranker):
    """
    Post-retrieval heuristic reranker — keyword overlap, quoted phrase
    matching, and person name boosting.

    Blends semantic ranking (preserved via input position) with lexical
    heuristics that embedding models miss: exact keyword matches, quoted
    phrases, and proper noun detection.

    Zero external dependencies. Pure Python.

    Usage:
        mem = MemBlock(
            storage="sqlite:///./memory.db",
            embeddings=True,
            reranker=HeuristicReranker(),
        )
    """

    _STOP_WORDS = frozenset({
        "what", "when", "where", "who", "how", "which", "did", "do",
        "was", "were", "have", "has", "had", "is", "are", "the", "a",
        "an", "my", "me", "i", "you", "your", "their", "it", "its",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "ago", "last", "that", "this", "there", "about", "get", "got",
        "give", "gave", "buy", "bought", "made", "make", "said",
        "and", "but", "not", "all", "can", "her", "his", "one", "our",
        "out", "new", "now", "old", "see", "way", "too", "use", "yes",
        "also", "been", "call", "come", "each", "just", "like", "much",
        "must", "only", "over", "such", "take", "than", "them", "then",
        "they", "very", "well", "will", "some", "would", "could",
        "should", "shall", "may", "might", "been", "does", "going",
        "think", "know", "want", "need", "tell", "told", "asked",
        "something", "anything", "everything", "nothing", "someone",
        "anyone", "everyone",
    })

    _NOT_NAMES = frozenset({
        "What", "When", "Where", "Who", "How", "Which", "Did", "Do",
        "Was", "Were", "Have", "Has", "Had", "Is", "Are", "The", "My",
        "Our", "Their", "Can", "Could", "Would", "Should", "Will",
        "Shall", "May", "Might", "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday", "January",
        "February", "March", "April", "June", "July", "August",
        "September", "October", "November", "December", "In", "On",
        "At", "For", "To", "Of", "With", "By", "From", "And", "But",
        "It", "Its", "This", "That", "These", "Those", "Also", "Just",
        "Very", "More", "Said", "Speaker", "Person", "Time", "Date",
        "Year", "Day", "Previously", "Recently",
    })

    def __init__(
        self,
        keyword_weight: float = 0.50,
        phrase_weight: float = 0.60,
        name_weight: float = 0.20,
    ) -> None:
        """
        Args:
            keyword_weight: Max boost for keyword overlap (0-1 scale).
            phrase_weight: Boost for quoted phrase exact match.
            name_weight: Boost for person name match.
        """
        self._kw_weight = keyword_weight
        self._phrase_weight = phrase_weight
        self._name_weight = name_weight

    @property
    def name(self) -> str:
        return "heuristic"

    def rerank(self, query: str, blocks: list[Block], top_k: int = 10) -> list[Block]:
        if not blocks or not query:
            return blocks[:top_k]

        keywords = self._extract_keywords(query)
        phrases = self._extract_quoted_phrases(query)
        names = self._extract_person_names(query)

        scored = []
        for rank, block in enumerate(blocks):
            # Base score from semantic ranking (input order = pre-sorted)
            base = 1.0 / (rank + 1)

            text_lower = block.content.lower()

            # Keyword overlap boost
            kw_boost = 0.0
            if keywords:
                hits = sum(1 for kw in keywords if kw in text_lower)
                kw_boost = (hits / len(keywords)) * self._kw_weight

            # Quoted phrase boost
            phrase_boost = 0.0
            if phrases:
                for p in phrases:
                    if p.lower() in text_lower:
                        phrase_boost = self._phrase_weight
                        break

            # Person name boost
            name_boost = 0.0
            if names:
                hits = sum(1 for n in names if n.lower() in text_lower)
                if hits:
                    name_boost = min(hits / len(names), 1.0) * self._name_weight

            total = base + kw_boost + phrase_boost + name_boost
            scored.append((block, total))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [b for b, _ in scored[:top_k]]

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract non-stopword tokens (3+ chars) from text."""
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
        return [w for w in words if w not in self._STOP_WORDS]

    def _extract_quoted_phrases(self, text: str) -> list[str]:
        """Extract quoted phrases from text."""
        phrases = []
        for pat in [r"'([^']{3,60})'", r'"([^"]{3,60})"']:
            phrases.extend(re.findall(pat, text))
        return [p.strip() for p in phrases if len(p.strip()) >= 3]

    def _extract_person_names(self, text: str) -> list[str]:
        """Extract likely person names (capitalized, not common words)."""
        words = re.findall(r"\b([A-Z][a-z]{2,15})\b", text)
        return list(set(w for w in words if w not in self._NOT_NAMES))


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
