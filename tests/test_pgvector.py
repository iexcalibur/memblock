"""Tests for pgvector integration in PostgreSQLAdapter and QueryEngine.

Uses mocks since these tests don't require a real PostgreSQL + pgvector instance.
"""

import struct
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from memblock.embeddings import pack_embedding, unpack_embedding, cosine_similarity
from memblock.storage.base import StorageAdapter
from memblock.query import QueryEngine


# ─── Base class stub ──────────────────────────────────────────────────────


class TestBaseAdapterSearchStub:
    """search_similar_embeddings returns [] by default."""

    def test_default_returns_empty(self):
        class DummyAdapter(StorageAdapter):
            def save_block(self, block): pass
            def get_block(self, block_id): return None
            def delete_block(self, block_id): pass
            def update_block(self, block_id, updates): pass
            def query_blocks(self, filters): return []
            def get_all_blocks(self, include_deleted=False): return []
            def save_edge(self, edge): pass
            def get_edges(self, block_id, direction="both"): return []
            def delete_edge(self, edge_id): pass
            def delete_edges_for_block(self, block_id): pass
            def save_operation(self, op): pass
            def get_operations(self, block_id=None): return []
            def get_last_operation(self): return None
            def close(self): pass
            def initialize(self): pass

        adapter = DummyAdapter()
        result = adapter.search_similar_embeddings(b"dummy", limit=10)
        assert result == []


# ─── PostgreSQLAdapter pgvector fields ────────────────────────────────────


class TestPostgreSQLAdapterPgvector:

    def _make_adapter(self):
        """Create a PostgreSQLAdapter with mocked psycopg."""
        with patch.dict("sys.modules", {
            "psycopg": MagicMock(),
            "psycopg.rows": MagicMock(),
        }):
            import importlib
            import memblock.storage.postgresql as pg_mod
            importlib.reload(pg_mod)
            pg_mod.HAS_PSYCOPG = True
            adapter = pg_mod.PostgreSQLAdapter(
                dsn="postgresql://test:test@localhost/test",
                user_id="test_user",
            )
            return adapter

    def test_pgvector_fields_initialized(self):
        adapter = self._make_adapter()
        assert adapter._has_pgvector is False
        assert adapter._embedding_dims is None

    def test_search_similar_returns_empty_without_pgvector(self):
        adapter = self._make_adapter()
        vec = pack_embedding([0.1, 0.2, 0.3])
        result = adapter.search_similar_embeddings(vec, limit=5)
        assert result == []

    def _setup_mock_conn(self, adapter):
        """Attach a mock connection that passes the conn property check."""
        mock_conn = MagicMock()
        mock_conn.closed = False  # must be explicitly False, not MagicMock
        adapter._conn = mock_conn
        return mock_conn

    def test_ensure_pgvector_index_sets_dims(self):
        adapter = self._make_adapter()
        adapter._has_pgvector = True
        mock_conn = self._setup_mock_conn(adapter)

        adapter._ensure_pgvector_index(384)
        assert adapter._embedding_dims == 384
        # CREATE INDEX was called
        assert mock_conn.cursor.return_value.__enter__.return_value.execute.called

    def test_ensure_pgvector_index_noop_on_second_call(self):
        adapter = self._make_adapter()
        adapter._has_pgvector = True
        adapter._embedding_dims = 384  # already set
        mock_conn = self._setup_mock_conn(adapter)

        adapter._ensure_pgvector_index(384)
        # Should be a no-op, no execute calls
        mock_conn.cursor.return_value.__enter__.return_value.execute.assert_not_called()

    def test_save_embedding_dual_write_when_pgvector(self):
        adapter = self._make_adapter()
        adapter._has_pgvector = True
        adapter._embedding_dims = 3
        mock_conn = self._setup_mock_conn(adapter)

        vec = pack_embedding([0.1, 0.2, 0.3])
        adapter.save_embedding("blk_1", vec)

        cur = mock_conn.cursor.return_value.__enter__.return_value
        # Should have at least 2 execute calls (BYTEA insert + pgvector insert)
        assert cur.execute.call_count >= 2

    def test_delete_embedding_deletes_from_both_tables(self):
        adapter = self._make_adapter()
        adapter._has_pgvector = True
        mock_conn = self._setup_mock_conn(adapter)

        adapter.delete_embedding("blk_1")

        cur = mock_conn.cursor.return_value.__enter__.return_value
        # Should have 2 delete calls (BYTEA + pgvector)
        assert cur.execute.call_count == 2


# ─── QueryEngine hybrid search with server-side results ──────────────────


class TestHybridSearchWithPgvector:

    def _make_mock_block(self, block_id, content="test"):
        block = MagicMock()
        block.id = block_id
        block.content = content
        block.deleted = False
        block.metadata.confidence = 0.9
        block.metadata.access_count = 1
        block.metadata.created_at.isoformat.return_value = "2025-01-01T00:00:00"
        return block

    def _make_provider(self, embed_result=None):
        """Create a mock embedding provider that passes isinstance check."""
        from memblock.embeddings import EmbeddingProvider

        class MockProvider(EmbeddingProvider):
            def __init__(self, result):
                self._result = result or [[0.1, 0.2, 0.3]]

            def embed(self, texts):
                return self._result

            @property
            def dimensions(self):
                return len(self._result[0]) if self._result else 3

        return MockProvider(embed_result)

    def test_hybrid_search_prefers_server_side(self):
        """When search_similar_embeddings returns results, skip brute force."""
        mock_storage = MagicMock()
        mock_storage.search_similar_embeddings.return_value = [
            ("blk_1", 0.95),
            ("blk_2", 0.80),
        ]
        mock_storage.get_all_embeddings.return_value = []

        mock_graph = MagicMock()
        mock_decay = MagicMock()
        mock_decay.calculate_strength.return_value = 0.8

        provider = self._make_provider()

        engine = QueryEngine(
            mock_storage, mock_graph, mock_decay,
            embedding_provider=provider,
        )

        fts_candidates = [
            self._make_mock_block("blk_1"),
            self._make_mock_block("blk_3"),
        ]

        result = engine._hybrid_search("test query", fts_candidates, limit=10)

        mock_storage.search_similar_embeddings.assert_called_once()
        mock_storage.get_all_embeddings.assert_not_called()
        assert "blk_1" in result
        assert "blk_2" in result

    def test_hybrid_search_falls_back_when_no_pgvector(self):
        """When search_similar_embeddings returns [], use brute force."""
        mock_storage = MagicMock()
        mock_storage.search_similar_embeddings.return_value = []

        vec1 = pack_embedding([0.1, 0.2, 0.3])
        vec2 = pack_embedding([0.4, 0.5, 0.6])
        mock_storage.get_all_embeddings.return_value = [
            ("blk_1", vec1),
            ("blk_2", vec2),
        ]

        mock_graph = MagicMock()
        mock_decay = MagicMock()
        mock_decay.calculate_strength.return_value = 0.8

        provider = self._make_provider()

        engine = QueryEngine(
            mock_storage, mock_graph, mock_decay,
            embedding_provider=provider,
        )

        fts_candidates = [self._make_mock_block("blk_1")]
        result = engine._hybrid_search("test query", fts_candidates, limit=10)

        mock_storage.get_all_embeddings.assert_called_once()
        assert len(result) > 0


# ─── Utility function tests ──────────────────────────────────────────────


class TestEmbeddingUtilities:

    def test_pack_unpack_roundtrip(self):
        vec = [0.1, 0.2, 0.3, 0.4, 0.5]
        packed = pack_embedding(vec)
        unpacked = unpack_embedding(packed)
        assert len(unpacked) == 5
        for a, b in zip(vec, unpacked):
            assert abs(a - b) < 1e-6

    def test_cosine_similarity_identical(self):
        vec = [1.0, 0.0, 0.0]
        assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(a, b)) < 1e-6
