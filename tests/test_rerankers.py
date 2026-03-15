"""Tests for reranker support — BM25, Callable, and integration."""

import pytest

from memblock import MemBlock, BlockType
from memblock.block import Block
from memblock.rerankers import BM25Reranker, CallableReranker, Reranker
from memblock.types import BlockMetadata, SourceType


def _make_block(content: str, block_id: str = "blk_test") -> Block:
    """Helper to create a Block for testing."""
    return Block(
        id=block_id,
        content=content,
        type=BlockType.FACT,
        tags=[],
        metadata=BlockMetadata(confidence=1.0, source=SourceType.EXPLICIT),
    )


class TestBM25Reranker:
    def test_basic_reranking(self):
        reranker = BM25Reranker()
        blocks = [
            _make_block("JavaScript is for web development", "blk_1"),
            _make_block("Python is great for machine learning and AI", "blk_2"),
            _make_block("Rust is a systems programming language", "blk_3"),
        ]
        results = reranker.rerank("Python machine learning", blocks, top_k=3)
        assert len(results) == 3
        # Python block should rank first
        assert results[0].id == "blk_2"

    def test_top_k_limit(self):
        reranker = BM25Reranker()
        blocks = [_make_block(f"Block {i}", f"blk_{i}") for i in range(10)]
        results = reranker.rerank("Block", blocks, top_k=3)
        assert len(results) == 3

    def test_empty_blocks(self):
        reranker = BM25Reranker()
        results = reranker.rerank("query", [], top_k=5)
        assert results == []

    def test_empty_query(self):
        reranker = BM25Reranker()
        blocks = [_make_block("Some content")]
        results = reranker.rerank("", blocks, top_k=5)
        assert len(results) == 1

    def test_no_matching_terms(self):
        reranker = BM25Reranker()
        blocks = [
            _make_block("cats and dogs", "blk_1"),
            _make_block("fish and birds", "blk_2"),
        ]
        results = reranker.rerank("quantum physics", blocks, top_k=5)
        assert len(results) == 2  # All returned, just with 0 scores

    def test_name_property(self):
        reranker = BM25Reranker()
        assert reranker.name == "bm25"

    def test_custom_params(self):
        reranker = BM25Reranker(k1=2.0, b=0.5)
        blocks = [_make_block("test content")]
        results = reranker.rerank("test", blocks)
        assert len(results) == 1

    def test_single_block(self):
        reranker = BM25Reranker()
        blocks = [_make_block("Only block with Python")]
        results = reranker.rerank("Python", blocks, top_k=5)
        assert len(results) == 1

    def test_relevance_ordering(self):
        reranker = BM25Reranker()
        blocks = [
            _make_block("The cat sat on the mat", "blk_1"),
            _make_block("Python Python Python programming", "blk_2"),
            _make_block("A single mention of Python", "blk_3"),
        ]
        results = reranker.rerank("Python programming", blocks, top_k=3)
        # Block with most Python mentions should rank highest
        assert results[0].id == "blk_2"


class TestCallableReranker:
    def test_basic(self):
        def reverse_reranker(query, blocks, top_k):
            return list(reversed(blocks[:top_k]))

        reranker = CallableReranker(fn=reverse_reranker, reranker_name="reverse")
        blocks = [_make_block(f"B{i}", f"blk_{i}") for i in range(3)]
        results = reranker.rerank("query", blocks, top_k=3)
        assert results[0].id == "blk_2"
        assert results[-1].id == "blk_0"

    def test_name_property(self):
        reranker = CallableReranker(fn=lambda q, b, k: b, reranker_name="my_ranker")
        assert reranker.name == "my_ranker"

    def test_default_name(self):
        reranker = CallableReranker(fn=lambda q, b, k: b)
        assert reranker.name == "custom"


class TestRerankerIntegration:
    def test_memblock_with_bm25_reranker(self):
        """BM25 reranker integrates with MemBlock query."""
        reranker = BM25Reranker()
        mem = MemBlock(storage="sqlite:///:memory:", reranker=reranker)

        mem.store("Python is great for data science", type=BlockType.FACT)
        mem.store("JavaScript is for frontend development", type=BlockType.FACT)
        mem.store("Python machine learning frameworks", type=BlockType.FACT)

        results = mem.query(text_search="Python data science", limit=3)
        assert len(results) >= 1
        # The Python-related blocks should be ranked higher
        assert "Python" in results[0].content
        mem.close()

    def test_memblock_reranker_with_no_text_search(self):
        """Reranker should not affect non-text queries."""
        reranker = BM25Reranker()
        mem = MemBlock(storage="sqlite:///:memory:", reranker=reranker)

        mem.store("Fact 1", type=BlockType.FACT)
        mem.store("Pref 1", type=BlockType.PREFERENCE)

        results = mem.query(type=BlockType.FACT)
        assert len(results) >= 1
        assert all(b.type == BlockType.FACT for b in results)
        mem.close()

    def test_callable_reranker_integration(self):
        """Custom callable reranker works with MemBlock."""
        def always_reverse(query, blocks, top_k):
            return list(reversed(blocks[:top_k]))

        reranker = CallableReranker(fn=always_reverse)
        mem = MemBlock(storage="sqlite:///:memory:", reranker=reranker)

        mem.store("First stored", type=BlockType.FACT)
        mem.store("Second stored", type=BlockType.FACT)

        results = mem.query(text_search="stored", limit=10)
        assert len(results) >= 1
        mem.close()
