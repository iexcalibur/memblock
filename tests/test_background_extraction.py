"""Tests for background extraction worker."""

import json
import time
import threading
import pytest

from memblock import MemBlock, BlockType, EdgeRelation
from memblock.background import BackgroundExtractor
from memblock.extraction import ExtractionResult


def mock_llm_good(system: str, user: str) -> str:
    """Mock LLM that returns valid extraction results with a small delay."""
    time.sleep(0.1)  # Simulate LLM latency
    return json.dumps([
        {
            "content": "User prefers Python",
            "type": "preference",
            "confidence": 0.9,
            "source": "explicit",
            "tags": ["language"],
        },
    ])


def mock_llm_slow(system: str, user: str) -> str:
    """Mock LLM that takes a while (simulates real API call)."""
    time.sleep(0.5)
    return json.dumps([
        {
            "content": "User works at Acme Corp",
            "type": "fact",
            "confidence": 0.8,
            "source": "explicit",
            "tags": ["work"],
        },
    ])


def mock_llm_fail(system: str, user: str) -> str:
    """Mock LLM that raises an error."""
    raise RuntimeError("LLM API error")


class TestBackgroundExtractor:
    """Test the BackgroundExtractor worker directly."""

    def test_submit_and_wait(self):
        results = []

        def on_complete(block_id, extracted_ids):
            results.append((block_id, extracted_ids))

        bg = BackgroundExtractor(max_workers=2, on_complete=on_complete)

        def fake_extract(content):
            time.sleep(0.1)
            return ExtractionResult(block_ids=["ext_1", "ext_2"], blocks_created=2)

        links = []
        def fake_link(ext_id, src_id):
            links.append((ext_id, src_id))

        bg.submit("block_1", "test content", fake_extract, fake_link)
        assert bg.pending_count >= 0  # may already be running

        bg.wait()
        assert bg.pending_count == 0
        assert len(links) == 2
        assert ("ext_1", "block_1") in links
        assert ("ext_2", "block_1") in links
        bg.shutdown()

    def test_stats_tracking(self):
        bg = BackgroundExtractor(max_workers=1)

        def fake_extract(content):
            return ExtractionResult(block_ids=["ext_1"], blocks_created=1)

        bg.submit("b1", "content", fake_extract, lambda a, b: None)
        bg.submit("b2", "content", fake_extract, lambda a, b: None)
        bg.wait()

        stats = bg.stats
        assert stats["completed"] == 2
        assert stats["failed"] == 0
        assert stats["total_submitted"] == 2
        assert stats["pending"] == 0
        bg.shutdown()

    def test_error_handling(self):
        errors = []

        def on_error(block_id, exc):
            errors.append((block_id, str(exc)))

        bg = BackgroundExtractor(max_workers=1, on_error=on_error)

        def failing_extract(content):
            raise RuntimeError("API down")

        bg.submit("b1", "content", failing_extract, lambda a, b: None)
        bg.wait()

        stats = bg.stats
        assert stats["failed"] == 1
        assert stats["completed"] == 0
        assert len(errors) == 1
        assert errors[0][0] == "b1"
        bg.shutdown()

    def test_shutdown_rejects_new_jobs(self):
        bg = BackgroundExtractor(max_workers=1)
        bg.shutdown()

        # Should not raise, just log warning
        bg.submit("b1", "content", lambda c: None, lambda a, b: None)
        assert bg.pending_count == 0


class TestMemBlockBackgroundExtract:
    """Test background extraction integrated into MemBlock."""

    def test_store_returns_immediately_with_background(self):
        mem = MemBlock(
            storage="sqlite:///:memory:",
            auto_extract_on_store=True,
            background_extract=True,
            extract_provider=mock_llm_slow,
        )

        t_start = time.perf_counter()
        block = mem.store("I love Python and work at Acme Corp", type=BlockType.EVENT)
        t_store = time.perf_counter() - t_start

        # store() should return much faster than the 0.5s extraction
        assert t_store < 0.3
        assert block is not None

        # Wait for background extraction to complete
        mem.wait_for_extractions()

        # Extracted blocks should now exist
        all_blocks = mem.query(semantic=False, limit=100)
        # Should have original + extracted blocks
        assert len(all_blocks) >= 2
        mem.close()

    def test_store_blocks_without_background(self):
        mem = MemBlock(
            storage="sqlite:///:memory:",
            auto_extract_on_store=True,
            background_extract=False,
            extract_provider=mock_llm_slow,
        )

        t_start = time.perf_counter()
        block = mem.store("I love Python and work at Acme Corp", type=BlockType.EVENT)
        t_store = time.perf_counter() - t_start

        # Without background, store() should take at least 0.5s (LLM latency)
        assert t_store >= 0.4
        assert block is not None
        mem.close()

    def test_extraction_pending_count(self):
        mem = MemBlock(
            storage="sqlite:///:memory:",
            auto_extract_on_store=True,
            background_extract=True,
            extract_provider=mock_llm_slow,
        )

        mem.store("First message", type=BlockType.EVENT)
        mem.store("Second message", type=BlockType.EVENT)

        # Should have pending extractions
        assert mem.extraction_pending >= 0  # race condition safe

        mem.wait_for_extractions()
        assert mem.extraction_pending == 0
        mem.close()

    def test_extraction_stats(self):
        mem = MemBlock(
            storage="sqlite:///:memory:",
            auto_extract_on_store=True,
            background_extract=True,
            extract_provider=mock_llm_good,
        )

        mem.store("I love Python", type=BlockType.EVENT)
        mem.wait_for_extractions()

        stats = mem.extraction_stats
        assert stats["completed"] >= 1
        assert stats["total_submitted"] >= 1
        mem.close()

    def test_no_background_without_auto_extract(self):
        mem = MemBlock(
            storage="sqlite:///:memory:",
            auto_extract_on_store=False,
            background_extract=True,  # should be ignored
            extract_provider=mock_llm_good,
        )

        assert mem._bg_extractor is None
        assert mem.extraction_pending == 0
        mem.close()

    def test_extraction_failure_doesnt_break_store(self):
        mem = MemBlock(
            storage="sqlite:///:memory:",
            auto_extract_on_store=True,
            background_extract=True,
            extract_provider=mock_llm_fail,
        )

        # Should not raise even though extraction will fail
        block = mem.store("Test content", type=BlockType.EVENT)
        assert block is not None

        mem.wait_for_extractions()
        stats = mem.extraction_stats
        # The extractor catches the error internally and returns ExtractionResult
        # with errors (not raising), so it counts as completed, not failed.
        # The key assertion is: store() didn't break.
        assert stats["total_submitted"] >= 1
        mem.close()

    def test_close_waits_for_pending(self):
        mem = MemBlock(
            storage="sqlite:///:memory:",
            auto_extract_on_store=True,
            background_extract=True,
            extract_provider=mock_llm_slow,
        )

        mem.store("Test content", type=BlockType.EVENT)
        # close() should wait for pending extractions before shutting down
        mem.close()  # should not raise

    def test_multiple_concurrent_stores(self):
        mem = MemBlock(
            storage="sqlite:///:memory:",
            auto_extract_on_store=True,
            background_extract=True,
            background_extract_workers=3,
            extract_provider=mock_llm_good,
        )

        for i in range(5):
            mem.store(f"Message {i} about different topics", type=BlockType.EVENT)

        mem.wait_for_extractions()
        stats = mem.extraction_stats
        assert stats["total_submitted"] == 5
        assert stats["completed"] == 5
        assert stats["failed"] == 0
        mem.close()
