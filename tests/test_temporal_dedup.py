"""Tests for temporal-aware deduplication (G1, v0.12.0).

The same content stored at different `happened_at` timestamps must
coexist as two distinct blocks (time-series data) — pure-content
dedup would collapse them, breaking IIXR/portfolio-value snapshot
recall.

The opposite — same content + same timestamp — must still dedupe so
double-writes of identical-time events don't bloat the store.

Non-EVENT writes keep the legacy content-only hash so re-stating the
same fact ("my risk profile is moderate") still dedupes correctly.
"""

from datetime import datetime, timedelta, timezone

import pytest

from memblock import MemBlock, BlockType, DuplicatePolicy
from memblock.dedup import ContentHasher


class TestHashTemporal:
    """Pure hash-level tests — no storage backend involved."""

    def test_same_content_different_time_different_hash(self):
        ts1 = "2026-05-11T00:00:00"
        ts2 = "2026-05-18T00:00:00"
        h1 = ContentHasher.hash_temporal("IIXR=7.2", ts1)
        h2 = ContentHasher.hash_temporal("IIXR=7.2", ts2)
        assert h1 != h2

    def test_same_content_same_time_same_hash(self):
        ts = "2026-05-11T00:00:00"
        h1 = ContentHasher.hash_temporal("IIXR=7.2", ts)
        h2 = ContentHasher.hash_temporal("IIXR=7.2", ts)
        assert h1 == h2

    def test_datetime_and_iso_string_produce_same_hash(self):
        dt = datetime(2026, 5, 11, 0, 0, 0)
        h_dt = ContentHasher.hash_temporal("IIXR=7.2", dt)
        h_str = ContentHasher.hash_temporal("IIXR=7.2", dt.isoformat())
        assert h_dt == h_str

    def test_none_happened_at_falls_back_to_content_hash(self):
        h_temporal = ContentHasher.hash_temporal("IIXR=7.2", None)
        h_content = ContentHasher.hash("IIXR=7.2")
        assert h_temporal == h_content

    def test_temporal_hash_differs_from_content_hash_when_ts_present(self):
        h_temporal = ContentHasher.hash_temporal("IIXR=7.2", "2026-05-11T00:00:00")
        h_content = ContentHasher.hash("IIXR=7.2")
        assert h_temporal != h_content


class TestStoreTemporalDedup:
    """Storage-level: EVENT writes with happened_at must coexist
    at different timestamps; FACT writes must still dedupe by
    content alone."""

    @pytest.fixture
    def mem(self):
        return MemBlock(
            storage="sqlite:///:memory:",
            on_duplicate=DuplicatePolicy.ERROR,
        )

    def test_event_same_content_different_time_both_stored(self, mem):
        day1 = datetime(2026, 5, 11, tzinfo=timezone.utc)
        day8 = datetime(2026, 5, 18, tzinfo=timezone.utc)
        b1 = mem.store(
            "IIXR=7.2", type=BlockType.EVENT, happened_at=day1,
        )
        b2 = mem.store(
            "IIXR=7.2", type=BlockType.EVENT, happened_at=day8,
        )
        assert b1 is not None
        assert b2 is not None
        assert b1.id != b2.id
        # Both readable
        assert mem.get(b1.id) is not None
        assert mem.get(b2.id) is not None

    def test_event_same_content_same_time_dedupes(self, mem):
        day1 = datetime(2026, 5, 11, tzinfo=timezone.utc)
        mem.store("IIXR=7.2", type=BlockType.EVENT, happened_at=day1)
        with pytest.raises(Exception):
            # ERROR policy + identical (content, happened_at) hash
            # should collide.
            mem.store("IIXR=7.2", type=BlockType.EVENT, happened_at=day1)

    def test_fact_same_content_still_dedupes(self, mem):
        """FACT writes ignore happened_at for dedup (per-type policy)."""
        mem.store("risk profile: moderate", type=BlockType.FACT)
        with pytest.raises(Exception):
            mem.store("risk profile: moderate", type=BlockType.FACT)

    def test_event_without_happened_at_dedupes_by_content(self, mem):
        """EVENT writes that omit happened_at fall back to content-
        only dedup so the old behaviour is preserved when callers
        don't supply a timestamp."""
        mem.store("Login attempt", type=BlockType.EVENT)
        with pytest.raises(Exception):
            mem.store("Login attempt", type=BlockType.EVENT)
