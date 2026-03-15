"""Tests for custom metadata filtering on queries."""

import pytest

from memblock import MemBlock, BlockType


class TestMetadataFiltering:
    @pytest.fixture
    def mem(self):
        m = MemBlock(storage="sqlite:///:memory:")
        yield m
        m.close()

    def test_store_with_custom_metadata(self, mem):
        block = mem.store(
            "With metadata",
            type=BlockType.FACT,
            metadata={"env": "production", "priority": "high"},
        )
        assert block.metadata.custom_metadata is not None
        assert block.metadata.custom_metadata["env"] == "production"
        assert block.metadata.custom_metadata["priority"] == "high"

    def test_store_without_metadata(self, mem):
        block = mem.store("No metadata", type=BlockType.FACT)
        # custom_metadata should be None or empty
        assert block.metadata.custom_metadata is None or block.metadata.custom_metadata == {}

    def test_query_with_single_filter(self, mem):
        mem.store("Prod fact", type=BlockType.FACT, metadata={"env": "production"})
        mem.store("Dev fact", type=BlockType.FACT, metadata={"env": "development"})
        mem.store("No env", type=BlockType.FACT)

        results = mem.query(metadata_filters={"env": "production"})
        assert len(results) == 1
        assert results[0].content == "Prod fact"

    def test_query_with_multiple_filters(self, mem):
        mem.store("Match", type=BlockType.FACT, metadata={"env": "prod", "team": "backend"})
        mem.store("Wrong team", type=BlockType.FACT, metadata={"env": "prod", "team": "frontend"})
        mem.store("Wrong env", type=BlockType.FACT, metadata={"env": "dev", "team": "backend"})

        results = mem.query(metadata_filters={"env": "prod", "team": "backend"})
        assert len(results) == 1
        assert results[0].content == "Match"

    def test_query_no_filters_returns_all(self, mem):
        mem.store("A", type=BlockType.FACT, metadata={"x": "1"})
        mem.store("B", type=BlockType.FACT, metadata={"x": "2"})
        mem.store("C", type=BlockType.FACT)

        results = mem.query()
        assert len(results) == 3

    def test_filter_with_numeric_value(self, mem):
        mem.store("Version 1", type=BlockType.FACT, metadata={"version": "1"})
        mem.store("Version 2", type=BlockType.FACT, metadata={"version": "2"})

        results = mem.query(metadata_filters={"version": "1"})
        assert len(results) == 1
        assert results[0].content == "Version 1"

    def test_metadata_combined_with_type_filter(self, mem):
        mem.store("Prod fact", type=BlockType.FACT, metadata={"env": "prod"})
        mem.store("Prod pref", type=BlockType.PREFERENCE, metadata={"env": "prod"})

        results = mem.query(type=BlockType.FACT, metadata_filters={"env": "prod"})
        assert len(results) == 1
        assert results[0].content == "Prod fact"

    def test_metadata_combined_with_tags(self, mem):
        mem.store("Tagged + meta", type=BlockType.FACT, tags=["important"], metadata={"src": "api"})
        mem.store("Meta only", type=BlockType.FACT, metadata={"src": "api"})
        mem.store("Tag only", type=BlockType.FACT, tags=["important"])

        results = mem.query(tags=["important"], metadata_filters={"src": "api"})
        assert len(results) == 1
        assert results[0].content == "Tagged + meta"

    def test_metadata_combined_with_scope(self, mem):
        mem.store("Scoped + meta", type=BlockType.FACT, org_id="acme", metadata={"tier": "paid"})
        mem.store("Just scoped", type=BlockType.FACT, org_id="acme", metadata={"tier": "free"})
        mem.store("Just meta", type=BlockType.FACT, metadata={"tier": "paid"})

        results = mem.query(org_id="acme", metadata_filters={"tier": "paid"})
        assert len(results) == 1
        assert results[0].content == "Scoped + meta"

    def test_build_context_with_metadata_filters(self, mem):
        mem.store("Prod context", type=BlockType.FACT, metadata={"env": "prod"})
        mem.store("Dev context", type=BlockType.FACT, metadata={"env": "dev"})

        ctx = mem.build_context(query="context", metadata_filters={"env": "prod"})
        assert "Prod" in ctx
        assert "Dev" not in ctx

    def test_filter_no_match(self, mem):
        mem.store("Data", type=BlockType.FACT, metadata={"key": "value"})
        results = mem.query(metadata_filters={"key": "nonexistent"})
        assert len(results) == 0

    def test_metadata_persists_after_retrieval(self, mem):
        """Verify custom_metadata survives store → get round-trip."""
        block = mem.store("Persist test", type=BlockType.FACT, metadata={"foo": "bar"})
        retrieved = mem.get(block.id)
        assert retrieved is not None
        assert retrieved.metadata.custom_metadata is not None
        assert retrieved.metadata.custom_metadata["foo"] == "bar"
