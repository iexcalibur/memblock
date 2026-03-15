"""Tests for session_id scoping across the MemBlock stack."""

from __future__ import annotations

import pytest

from memblock import MemBlock, BlockType


@pytest.fixture
def mem():
    """In-memory MemBlock instance."""
    m = MemBlock(storage="sqlite:///:memory:")
    yield m
    m.close()


class TestSessionStore:
    """Test storing blocks with session_id."""

    def test_store_with_session_id(self, mem: MemBlock):
        block = mem.store("hello", type=BlockType.FACT, session_id="s1")
        assert block.metadata.session_id == "s1"

    def test_store_without_session_id(self, mem: MemBlock):
        block = mem.store("hello", type=BlockType.FACT)
        assert block.metadata.session_id is None

    def test_store_default_session_id(self):
        mem = MemBlock(storage="sqlite:///:memory:", session_id="default_sess")
        block = mem.store("hello", type=BlockType.FACT)
        assert block.metadata.session_id == "default_sess"
        # Per-call override takes precedence
        block2 = mem.store("world", type=BlockType.FACT, session_id="override")
        assert block2.metadata.session_id == "override"
        mem.close()

    def test_get_preserves_session_id(self, mem: MemBlock):
        block = mem.store("hello", type=BlockType.FACT, session_id="s1")
        retrieved = mem.get(block.id)
        assert retrieved is not None
        assert retrieved.metadata.session_id == "s1"


class TestSessionQuery:
    """Test querying with session_id filter."""

    def test_query_scoped_to_session(self, mem: MemBlock):
        mem.store("msg 1", type=BlockType.EVENT, session_id="chat_001")
        mem.store("msg 2", type=BlockType.EVENT, session_id="chat_001")
        mem.store("other chat", type=BlockType.EVENT, session_id="chat_002")

        results = mem.query(session_id="chat_001")
        assert len(results) == 2

        results = mem.query(session_id="chat_002")
        assert len(results) == 1

    def test_query_without_session_returns_all(self, mem: MemBlock):
        mem.store("msg 1", type=BlockType.EVENT, session_id="chat_001")
        mem.store("msg 2", type=BlockType.EVENT, session_id="chat_002")
        mem.store("msg 3", type=BlockType.EVENT)

        results = mem.query(limit=100)
        assert len(results) == 3

    def test_query_session_with_type_filter(self, mem: MemBlock):
        mem.store("fact 1", type=BlockType.FACT, session_id="s1")
        mem.store("event 1", type=BlockType.EVENT, session_id="s1")
        mem.store("fact 2", type=BlockType.FACT, session_id="s2")

        results = mem.query(type=BlockType.FACT, session_id="s1")
        assert len(results) == 1
        assert results[0].content == "fact 1"

    def test_query_nonexistent_session(self, mem: MemBlock):
        mem.store("hello", type=BlockType.FACT, session_id="s1")
        results = mem.query(session_id="nonexistent")
        assert len(results) == 0


class TestSessionConvenience:
    """Test get_sessions() and get_session_history()."""

    def test_get_sessions(self, mem: MemBlock):
        mem.store("a", type=BlockType.FACT, session_id="s2")
        mem.store("b", type=BlockType.FACT, session_id="s1")
        mem.store("c", type=BlockType.FACT, session_id="s2")
        mem.store("d", type=BlockType.FACT)  # no session

        sessions = mem.get_sessions()
        assert sessions == ["s1", "s2"]

    def test_get_sessions_empty(self, mem: MemBlock):
        mem.store("a", type=BlockType.FACT)
        assert mem.get_sessions() == []

    def test_get_session_history(self, mem: MemBlock):
        mem.store("first", type=BlockType.EVENT, session_id="chat_001")
        mem.store("second", type=BlockType.EVENT, session_id="chat_001")
        mem.store("other", type=BlockType.EVENT, session_id="chat_002")

        history = mem.get_session_history("chat_001")
        assert len(history) == 2
        # Should be chronological
        assert history[0].content == "first"
        assert history[1].content == "second"

    def test_get_session_history_limit(self, mem: MemBlock):
        for i in range(5):
            mem.store(f"msg {i}", type=BlockType.EVENT, session_id="s1")

        history = mem.get_session_history("s1", limit=3)
        assert len(history) == 3

    def test_get_session_history_empty(self, mem: MemBlock):
        history = mem.get_session_history("nonexistent")
        assert history == []


class TestSessionBuildContext:
    """Test build_context with session_id."""

    def test_build_context_scoped_to_session(self, mem: MemBlock):
        mem.store("Python is great for programming", type=BlockType.PREFERENCE, session_id="s1")
        mem.store("Java is fast for programming", type=BlockType.PREFERENCE, session_id="s2")

        context = mem.build_context(query="Python", session_id="s1")
        assert "Python" in context
        # s2 content should not appear
        assert "Java" not in context


class TestSessionBackwardsCompat:
    """Verify backwards compatibility — existing code works unchanged."""

    def test_no_session_id_works(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        block = mem.store("hello", type=BlockType.FACT)
        assert block.metadata.session_id is None
        results = mem.query()
        assert len(results) == 1
        mem.close()

    def test_migration_existing_blocks_have_none(self, mem: MemBlock):
        # Blocks stored without session_id should have None
        block = mem.store("test", type=BlockType.FACT)
        retrieved = mem.get(block.id)
        assert retrieved.metadata.session_id is None
