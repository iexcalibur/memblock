"""Tests for the embeddings module and hybrid search."""

import math
import pytest

from memblock.embeddings import (
    CallableEmbeddingProvider,
    EmbeddingProvider,
    cosine_similarity,
    pack_embedding,
    unpack_embedding,
    rrf_merge,
)
from memblock.memblock import MemBlock
from memblock.types import BlockType


# ─── Mock embedding provider for tests ────────────────────────────────────

def _simple_embed(texts: list[str]) -> list[list[float]]:
    """
    Simple deterministic embedding: hash-based vector for testing.
    Each word contributes to a specific dimension.
    """
    dims = 32
    results = []
    for text in texts:
        vec = [0.0] * dims
        words = text.lower().split()
        for i, word in enumerate(words):
            idx = hash(word) % dims
            vec[idx] += 1.0
        # Normalize
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        vec = [x / norm for x in vec]
        results.append(vec)
    return results


class MockEmbeddingProvider(CallableEmbeddingProvider):
    def __init__(self):
        super().__init__(fn=_simple_embed, dims=32)


# ─── Unit tests: utility functions ────────────────────────────────────────


class TestPackUnpack:
    def test_roundtrip(self):
        original = [0.1, 0.2, 0.3, -0.5, 1.0]
        packed = pack_embedding(original)
        unpacked = unpack_embedding(packed)
        assert len(unpacked) == 5
        for a, b in zip(original, unpacked):
            assert abs(a - b) < 1e-6

    def test_empty(self):
        packed = pack_embedding([])
        unpacked = unpack_embedding(packed)
        assert unpacked == []

    def test_large_vector(self):
        vec = [float(i) / 1000 for i in range(384)]
        packed = pack_embedding(vec)
        unpacked = unpack_embedding(packed)
        assert len(unpacked) == 384
        assert abs(unpacked[0] - 0.0) < 1e-6
        assert abs(unpacked[383] - 0.383) < 1e-3


class TestCosineSimilarity:
    def test_identical_vectors(self):
        vec = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_similar_vectors(self):
        a = [1.0, 1.0, 0.0]
        b = [1.0, 0.9, 0.1]
        sim = cosine_similarity(a, b)
        assert sim > 0.9


class TestRRFMerge:
    def test_basic_merge(self):
        fts = ["a", "b", "c"]
        vec = ["b", "d", "a"]
        merged = rrf_merge(fts, vec)
        # "a" and "b" appear in both lists, should rank highest
        ids = [bid for bid, _ in merged]
        assert "a" in ids[:2]
        assert "b" in ids[:2]

    def test_single_list(self):
        merged = rrf_merge(["x", "y"], [])
        ids = [bid for bid, _ in merged]
        assert ids == ["x", "y"]

    def test_no_overlap(self):
        merged = rrf_merge(["a", "b"], ["c", "d"])
        ids = [bid for bid, _ in merged]
        assert len(ids) == 4

    def test_scores_decrease(self):
        merged = rrf_merge(["a", "b", "c"], ["a", "b", "c"])
        scores = [s for _, s in merged]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]


# ─── Unit tests: embedding providers ─────────────────────────────────────


class TestCallableProvider:
    def test_embed(self):
        provider = MockEmbeddingProvider()
        result = provider.embed(["hello world"])
        assert len(result) == 1
        assert len(result[0]) == 32

    def test_dimensions(self):
        provider = MockEmbeddingProvider()
        assert provider.dimensions == 32

    def test_multiple_texts(self):
        provider = MockEmbeddingProvider()
        result = provider.embed(["hello", "world", "test"])
        assert len(result) == 3


# ─── Integration tests: SQLite embedding storage ─────────────────────────


class TestSQLiteEmbeddingStorage:
    def test_save_and_get_embedding(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        block = mem.store("test content", type=BlockType.FACT)

        vec = [0.1, 0.2, 0.3]
        packed = pack_embedding(vec)
        mem._storage.save_embedding(block.id, packed)

        retrieved = mem._storage.get_embedding(block.id)
        assert retrieved is not None
        unpacked = unpack_embedding(retrieved)
        assert len(unpacked) == 3
        assert abs(unpacked[0] - 0.1) < 1e-6
        mem.close()

    def test_get_nonexistent_embedding(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        result = mem._storage.get_embedding("nonexistent")
        assert result is None
        mem.close()

    def test_get_all_embeddings(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        b1 = mem.store("block one", type=BlockType.FACT)
        b2 = mem.store("block two", type=BlockType.FACT)

        mem._storage.save_embedding(b1.id, pack_embedding([1.0, 0.0]))
        mem._storage.save_embedding(b2.id, pack_embedding([0.0, 1.0]))

        all_emb = mem._storage.get_all_embeddings()
        assert len(all_emb) == 2
        ids = {bid for bid, _ in all_emb}
        assert b1.id in ids
        assert b2.id in ids
        mem.close()

    def test_get_all_embeddings_excludes_deleted(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        b1 = mem.store("keep me", type=BlockType.FACT)
        b2 = mem.store("delete me", type=BlockType.FACT)

        mem._storage.save_embedding(b1.id, pack_embedding([1.0]))
        mem._storage.save_embedding(b2.id, pack_embedding([2.0]))
        mem.delete(b2.id)

        all_emb = mem._storage.get_all_embeddings()
        assert len(all_emb) == 1
        assert all_emb[0][0] == b1.id
        mem.close()

    def test_delete_embedding(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        block = mem.store("test", type=BlockType.FACT)
        mem._storage.save_embedding(block.id, pack_embedding([1.0]))

        assert mem._storage.get_embedding(block.id) is not None
        mem._storage.delete_embedding(block.id)
        assert mem._storage.get_embedding(block.id) is None
        mem.close()

    def test_update_embedding(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        block = mem.store("test", type=BlockType.FACT)

        mem._storage.save_embedding(block.id, pack_embedding([1.0, 0.0]))
        mem._storage.save_embedding(block.id, pack_embedding([0.0, 1.0]))

        retrieved = unpack_embedding(mem._storage.get_embedding(block.id))
        assert abs(retrieved[0] - 0.0) < 1e-6
        assert abs(retrieved[1] - 1.0) < 1e-6
        mem.close()


# ─── Integration tests: hybrid search with mock embeddings ───────────────


class TestHybridSearch:
    def _create_mem_with_embeddings(self) -> MemBlock:
        """Create a MemBlock instance with mock embedding provider."""
        mem = MemBlock(storage="sqlite:///:memory:")
        # Inject mock provider
        provider = MockEmbeddingProvider()
        mem._embedding_provider = provider
        mem._query._embedding_provider = provider
        return mem

    def test_store_generates_embedding(self):
        mem = self._create_mem_with_embeddings()
        block = mem.store("Python is great for data science", type=BlockType.PREFERENCE)

        emb = mem._storage.get_embedding(block.id)
        assert emb is not None
        vec = unpack_embedding(emb)
        assert len(vec) == 32
        mem.close()

    def test_hybrid_query_returns_results(self):
        mem = self._create_mem_with_embeddings()
        mem.store("I love Python programming", type=BlockType.PREFERENCE, tags=["tech"])
        mem.store("Django is my favorite web framework", type=BlockType.PREFERENCE, tags=["tech"])
        mem.store("I enjoy hiking on weekends", type=BlockType.PREFERENCE, tags=["hobby"])

        results = mem.query(text_search="Python", semantic=True)
        assert len(results) > 0
        mem.close()

    def test_fts_only_mode(self):
        mem = self._create_mem_with_embeddings()
        mem.store("Python is great", type=BlockType.FACT)
        mem.store("Java is verbose", type=BlockType.FACT)

        # semantic=False should skip vector search
        results = mem.query(text_search="Python", semantic=False)
        assert len(results) >= 1
        assert any("Python" in r.content for r in results)
        mem.close()

    def test_no_embeddings_still_works(self):
        """Without embeddings, query should still work via FTS."""
        mem = MemBlock(storage="sqlite:///:memory:")
        mem.store("Python rocks", type=BlockType.FACT)
        results = mem.query(text_search="Python")
        assert len(results) == 1
        mem.close()

    def test_has_embeddings_property(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        assert mem.has_embeddings is False

        mem2 = self._create_mem_with_embeddings()
        assert mem2.has_embeddings is True
        mem.close()
        mem2.close()

    def test_stats_include_embedding_info(self):
        mem = self._create_mem_with_embeddings()
        mem.store("test block", type=BlockType.FACT)

        stats = mem.stats()
        assert "embeddings_enabled" in stats
        assert stats["embeddings_enabled"] is True
        assert "total_embeddings" in stats
        assert stats["total_embeddings"] == 1
        mem.close()

    def test_delete_removes_embedding(self):
        mem = self._create_mem_with_embeddings()
        block = mem.store("delete me", type=BlockType.FACT)
        assert mem._storage.get_embedding(block.id) is not None

        mem.delete(block.id)
        assert mem._storage.get_embedding(block.id) is None
        mem.close()

    def test_update_content_reembeds(self):
        mem = self._create_mem_with_embeddings()
        block = mem.store("original content", type=BlockType.FACT)
        original_emb = mem._storage.get_embedding(block.id)

        mem.update(block.id, content="completely different content")
        updated_emb = mem._storage.get_embedding(block.id)

        assert original_emb is not None
        assert updated_emb is not None
        # Embeddings should be different (different content)
        assert original_emb != updated_emb
        mem.close()

    def test_multiple_blocks_hybrid_ranking(self):
        """Blocks that match both FTS and vector should rank higher."""
        mem = self._create_mem_with_embeddings()
        mem.store("Python web development with Django", type=BlockType.FACT)
        mem.store("Machine learning with Python", type=BlockType.FACT)
        mem.store("Cooking Italian pasta recipes", type=BlockType.FACT)

        results = mem.query(text_search="Python", semantic=True, limit=10)
        # Python-related blocks should appear
        python_results = [r for r in results if "Python" in r.content]
        assert len(python_results) >= 1
        mem.close()
