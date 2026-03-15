"""Tests for opt-in auto-extraction via add_message / flush_extraction."""

import json
import pytest

from memblock import MemBlock, BlockType
from memblock.errors import ExtractionError
from memblock.extraction import ExtractionResult


def mock_llm_good(system: str, user: str) -> str:
    """Mock LLM that returns valid extraction results."""
    return json.dumps([
        {
            "content": "User prefers Python",
            "type": "preference",
            "confidence": 0.9,
            "source": "explicit",
            "tags": ["language"],
        },
        {
            "content": "User works at Acme Corp",
            "type": "fact",
            "confidence": 0.8,
            "source": "explicit",
            "tags": ["work"],
        },
    ])


def mock_llm_empty(system: str, user: str) -> str:
    """Mock LLM that returns no extractions."""
    return "[]"


def mock_llm_fail(system: str, user: str) -> str:
    """Mock LLM that raises an error."""
    raise RuntimeError("LLM API error")


class TestAddMessage:
    """Test message buffering and triggered extraction."""

    def test_buffer_accumulates_messages(self):
        mem = MemBlock(
            storage="sqlite:///:memory:",
            auto_extract=True,
            extract_provider=mock_llm_good,
            extract_every=5,
        )
        mem.add_message("user", "Hello")
        mem.add_message("assistant", "Hi!")
        assert len(mem._message_buffer) == 2
        assert mem._message_count == 2
        mem.close()

    def test_extraction_triggers_at_interval(self):
        mem = MemBlock(
            storage="sqlite:///:memory:",
            auto_extract=True,
            extract_provider=mock_llm_good,
            extract_every=3,
        )
        r1 = mem.add_message("user", "I love Python")
        assert r1 is None
        r2 = mem.add_message("assistant", "Great choice!")
        assert r2 is None
        r3 = mem.add_message("user", "I work at Acme Corp")
        # 3rd message should trigger extraction
        assert r3 is not None
        assert isinstance(r3, ExtractionResult)
        assert r3.blocks_created >= 1
        mem.close()

    def test_buffer_cleared_after_successful_extraction(self):
        mem = MemBlock(
            storage="sqlite:///:memory:",
            auto_extract=True,
            extract_provider=mock_llm_good,
            extract_every=2,
        )
        mem.add_message("user", "I love Python")
        mem.add_message("assistant", "Noted!")
        # Buffer should be cleared after extraction
        assert len(mem._message_buffer) == 0
        mem.close()

    def test_no_extraction_when_auto_extract_false(self):
        mem = MemBlock(
            storage="sqlite:///:memory:",
            auto_extract=False,
            extract_provider=mock_llm_good,
            extract_every=1,
        )
        result = mem.add_message("user", "I love Python")
        assert result is None
        assert len(mem._message_buffer) == 1
        mem.close()


class TestFlushExtraction:
    """Test manual flush_extraction."""

    def test_flush_extracts_buffer(self):
        mem = MemBlock(
            storage="sqlite:///:memory:",
            auto_extract=False,
            extract_provider=mock_llm_good,
        )
        mem.add_message("user", "I love Python")
        mem.add_message("assistant", "Great!")
        result = mem.flush_extraction()
        assert isinstance(result, ExtractionResult)
        assert result.blocks_created >= 1
        assert len(mem._message_buffer) == 0
        mem.close()

    def test_flush_empty_buffer_returns_error(self):
        mem = MemBlock(
            storage="sqlite:///:memory:",
            auto_extract=False,
            extract_provider=mock_llm_good,
        )
        result = mem.flush_extraction()
        assert len(result.errors) > 0
        mem.close()


class TestExtractionFailure:
    """Test that failed extraction preserves the buffer."""

    def test_failed_extraction_preserves_buffer(self):
        mem = MemBlock(
            storage="sqlite:///:memory:",
            auto_extract=True,
            extract_provider=mock_llm_fail,
            extract_every=2,
        )
        mem.add_message("user", "Hello")
        mem.add_message("assistant", "Hi")  # Triggers extraction which fails
        # Buffer should be preserved
        assert len(mem._message_buffer) == 2
        mem.close()

    def test_retry_after_failure(self):
        call_count = 0

        def mock_fail_then_succeed(system: str, user: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Temporary failure")
            return mock_llm_good(system, user)

        mem = MemBlock(
            storage="sqlite:///:memory:",
            auto_extract=False,
            extract_provider=mock_fail_then_succeed,
        )
        mem.add_message("user", "I love Python")

        # First attempt fails
        result1 = mem.flush_extraction()
        assert len(result1.errors) > 0
        assert len(mem._message_buffer) == 1  # Preserved

        # Reset extractor for retry (force re-init)
        mem._extractor = None
        # Second attempt succeeds
        result2 = mem.flush_extraction()
        assert result2.blocks_created >= 1
        assert len(mem._message_buffer) == 0  # Cleared
        mem.close()


class TestExtractionProviderErrors:
    """Test extraction provider configuration errors."""

    def test_missing_provider_returns_error(self):
        mem = MemBlock(
            storage="sqlite:///:memory:",
            auto_extract=True,
            extract_provider=None,
            extract_every=1,
        )
        result = mem.add_message("user", "test")
        assert result is not None
        assert len(result.errors) > 0
        mem.close()

    def test_callable_provider_works(self):
        mem = MemBlock(
            storage="sqlite:///:memory:",
            auto_extract=False,
            extract_provider=mock_llm_good,
        )
        mem.add_message("user", "I love Python and work at Acme")
        result = mem.flush_extraction()
        assert result.blocks_created >= 1
        mem.close()
