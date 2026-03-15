"""Deduplication system for MemBlock — content hash and semantic near-duplicate detection."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from enum import Enum
from typing import Any, TYPE_CHECKING

from memblock.block import Block

if TYPE_CHECKING:
    from memblock.storage.base import StorageAdapter


class DuplicatePolicy(str, Enum):
    """How to handle duplicate blocks."""

    ERROR = "error"
    SKIP = "skip"
    RETURN_EXISTING = "return_existing"
    MERGE = "merge"


class ContentHasher:
    """Computes normalized content hashes for deduplication."""

    @staticmethod
    def hash(content: str) -> str:
        """
        Normalize content and compute SHA-256 hash.

        Normalization: Unicode NFC → strip → lowercase → collapse whitespace.
        """
        normalized = unicodedata.normalize("NFC", content)
        normalized = normalized.strip().lower()
        normalized = re.sub(r"\s+", " ", normalized)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class DuplicateChecker:
    """
    Checks for duplicate blocks using exact content hash and optional semantic similarity.

    Exact dedup works without embeddings. Semantic dedup only runs when
    an embedding provider is configured.
    """

    def __init__(
        self,
        storage: StorageAdapter,
        embedding_provider: Any = None,
        similarity_threshold: float = 0.95,
    ) -> None:
        self.storage = storage
        self.embedding_provider = embedding_provider
        self.similarity_threshold = similarity_threshold

    def check_exact(self, content_hash: str) -> Block | None:
        """Check for an exact content match by hash."""
        return self.storage.get_block_by_content_hash(content_hash)

    def check_semantic(self, content: str) -> Block | None:
        """Check for a semantic near-duplicate using embedding cosine similarity."""
        if self.embedding_provider is None:
            return None

        try:
            from memblock.embeddings import cosine_similarity, pack_embedding, unpack_embedding

            vectors = self.embedding_provider.embed([content])
            if not vectors:
                return None

            query_vec = vectors[0]
            all_embeddings = self.storage.get_all_embeddings()

            best_block_id = None
            best_score = 0.0

            for block_id, emb_bytes in all_embeddings:
                stored_vec = unpack_embedding(emb_bytes)
                score = cosine_similarity(query_vec, stored_vec)
                if score >= self.similarity_threshold and score > best_score:
                    best_score = score
                    best_block_id = block_id

            if best_block_id:
                return self.storage.get_block(best_block_id)
        except Exception:
            pass  # Semantic dedup failure should not block storage

        return None

    def check(self, content: str, content_hash: str) -> Block | None:
        """Check for duplicates: exact first, then semantic if available."""
        # Exact match first (fast, no embeddings needed)
        existing = self.check_exact(content_hash)
        if existing is not None:
            return existing

        # Semantic match (slower, requires embeddings)
        return self.check_semantic(content)
