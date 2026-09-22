"""
Adapter-level tests for SQLite full-text ranking, filtered LIMIT, and the
FTS-first query plan (SQLiteAdapter.query_blocks / _build_query_sql).

These guard the retrieval fix where:
- text_search results are ordered by FTS5 rank (best match first) unless an
  explicit sort_by is given,
- LIMIT is applied after every WHERE condition (no under-fill), and
- the blocks_fts MATCH is the outermost loop of the plan (evaluated once),
  never driven from idx_blocks_deleted (which re-ran MATCH per row).
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone

import pytest

from memblock.block import Block
from memblock.storage.sqlite import SQLiteAdapter
from memblock.types import BlockMetadata, BlockType


@pytest.fixture
def db():
    adapter = SQLiteAdapter(":memory:")
    adapter.initialize()
    yield adapter
    adapter.close()


T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _at(minutes: int) -> datetime:
    return T0 + timedelta(minutes=minutes)


def _save(
    db: SQLiteAdapter,
    content: str,
    *,
    type: BlockType = BlockType.FACT,
    minutes: int = 0,
    confidence: float = 1.0,
    session_id: str | None = None,
    access_count: int = 0,
    tags: list[str] | None = None,
) -> Block:
    block = Block(
        content=content,
        type=type,
        tags=tags or [],
        metadata=BlockMetadata(
            created_at=_at(minutes),
            confidence=confidence,
            session_id=session_id,
            access_count=access_count,
        ),
    )
    db.save_block(block)
    return block


def _plan(db: SQLiteAdapter, filters: dict) -> tuple[list[str], str, list]:
    """Run EXPLAIN QUERY PLAN on the exact SQL the adapter builds."""
    sql, params = db._build_query_sql(filters)
    rows = db.conn.execute("EXPLAIN QUERY PLAN " + sql, params).fetchall()
    return [row[3] for row in rows], sql, params


# ─── (a) relevance ordering ──────────────────────────────────────────────────


class TestRelevanceOrdering:
    QUERY = "python asyncio event loop"

    def _seed(self, db: SQLiteAdapter) -> tuple[Block, list[Block]]:
        # One OLD block matching every query word (strong bm25 score)...
        strong = _save(
            db,
            "python asyncio event loop: the asyncio event loop schedules python coroutines",
            minutes=0,
        )
        # ...vs 30 NEWER blocks each sharing exactly one weak token.
        weak = [
            _save(
                db,
                f"note {i}: a long unrelated sentence about deployment pipelines that "
                f"mentions python once among many other tokens for padding purposes",
                minutes=10 + i,
            )
            for i in range(30)
        ]
        return strong, weak

    def test_sort_by_relevance_puts_strong_match_first(self, db: SQLiteAdapter):
        strong, weak = self._seed(db)
        results = db.query_blocks(
            {"text_search": self.QUERY, "sort_by": "relevance", "limit": 10}
        )
        assert len(results) == 10
        assert results[0].id == strong.id

    def test_sort_by_absent_defaults_to_relevance_for_text_search(self, db: SQLiteAdapter):
        strong, weak = self._seed(db)
        results = db.query_blocks({"text_search": self.QUERY, "limit": 10})
        assert len(results) == 10
        assert results[0].id == strong.id

        # Also without a limit: full match set, still best-first.
        all_results = db.query_blocks({"text_search": self.QUERY})
        assert len(all_results) == 31
        assert all_results[0].id == strong.id

    def test_explicit_created_at_sort_still_newest_first(self, db: SQLiteAdapter):
        strong, weak = self._seed(db)
        results = db.query_blocks(
            {"text_search": self.QUERY, "sort_by": "created_at", "limit": 10}
        )
        assert len(results) == 10
        assert results[0].id == weak[-1].id  # newest block
        assert strong.id not in {b.id for b in results}
        created = [b.metadata.created_at for b in results]
        assert created == sorted(created, reverse=True)

    def test_explicit_confidence_and_access_count_sorts_still_apply(self, db: SQLiteAdapter):
        low = _save(db, "kafka consumer lag", confidence=0.2, access_count=50, minutes=0)
        high = _save(db, "kafka consumer lag", confidence=0.9, access_count=1, minutes=1)

        by_conf = db.query_blocks({"text_search": "kafka", "sort_by": "confidence"})
        assert [b.id for b in by_conf] == [high.id, low.id]

        by_access = db.query_blocks({"text_search": "kafka", "sort_by": "access_count"})
        assert [b.id for b in by_access] == [low.id, high.id]

    def test_relevance_on_non_text_query_falls_back_to_created_at(self, db: SQLiteAdapter):
        older = _save(db, "first", minutes=0)
        newer = _save(db, "second", minutes=5)
        results = db.query_blocks({"sort_by": "relevance"})
        assert [b.id for b in results] == [newer.id, older.id]


# ─── (b) no under-fill with filters ──────────────────────────────────────────


class TestNoUnderfillWithFilters:
    QUERY = "kubernetes deployment rollout strategy"

    def _seed(self, db: SQLiteAdapter) -> tuple[list[Block], list[Block]]:
        # 40 FACT blocks that are the strongest matches (all four query words, repeated)
        facts = [
            _save(
                db,
                "kubernetes deployment rollout strategy; kubernetes deployment rollout strategy",
                type=BlockType.FACT,
                minutes=i,
                confidence=0.3,
                session_id="sess-facts",
            )
            for i in range(40)
        ]
        # 20 PREFERENCE blocks that match only weakly (one query word)
        prefs = [
            _save(
                db,
                f"user prefers kubernetes for hosting side projects number {i}",
                type=BlockType.PREFERENCE,
                minutes=100 + i,
                confidence=0.9,
                session_id="sess-prefs",
            )
            for i in range(20)
        ]
        return facts, prefs

    def test_type_filter_fills_limit(self, db: SQLiteAdapter):
        facts, prefs = self._seed(db)
        pref_ids = {b.id for b in prefs}

        results = db.query_blocks(
            {"text_search": self.QUERY, "type": BlockType.PREFERENCE, "limit": 5}
        )
        assert len(results) == 5
        assert all(b.type == BlockType.PREFERENCE for b in results)
        assert {b.id for b in results} <= pref_ids

    def test_session_id_metadata_filter_fills_limit(self, db: SQLiteAdapter):
        facts, prefs = self._seed(db)
        pref_ids = {b.id for b in prefs}

        results = db.query_blocks(
            {"text_search": self.QUERY, "session_id": "sess-prefs", "limit": 5}
        )
        assert len(results) == 5
        assert all(b.metadata.session_id == "sess-prefs" for b in results)
        assert {b.id for b in results} <= pref_ids

    def test_min_confidence_filter_fills_limit(self, db: SQLiteAdapter):
        facts, prefs = self._seed(db)
        pref_ids = {b.id for b in prefs}

        results = db.query_blocks(
            {"text_search": self.QUERY, "min_confidence": 0.8, "limit": 5}
        )
        assert len(results) == 5
        assert all(b.metadata.confidence >= 0.8 for b in results)
        assert {b.id for b in results} <= pref_ids

    def test_combined_filters_fill_limit(self, db: SQLiteAdapter):
        facts, prefs = self._seed(db)
        results = db.query_blocks(
            {
                "text_search": self.QUERY,
                "type": BlockType.PREFERENCE,
                "session_id": "sess-prefs",
                "min_confidence": 0.8,
                "sort_by": "relevance",
                "limit": 7,
            }
        )
        assert len(results) == 7
        assert all(b.type == BlockType.PREFERENCE for b in results)

    def test_unfiltered_text_search_ranks_facts_first(self, db: SQLiteAdapter):
        facts, prefs = self._seed(db)
        fact_ids = {b.id for b in facts}
        results = db.query_blocks({"text_search": self.QUERY, "limit": 5})
        assert len(results) == 5
        assert {b.id for b in results} <= fact_ids


# ─── (c) limit bounds hydration ──────────────────────────────────────────────


class TestLimitBoundsHydration:
    def test_row_to_block_called_at_most_limit_times(self, db: SQLiteAdapter, monkeypatch):
        for i in range(300):
            _save(db, f"notebook entry {i} about the notebook workflow", minutes=i)

        # Sanity: all 300 match the query.
        assert len(db.query_blocks({"text_search": "notebook"})) == 300

        calls = {"n": 0}
        original = SQLiteAdapter._row_to_block

        def counting(self, row):
            calls["n"] += 1
            return original(self, row)

        monkeypatch.setattr(SQLiteAdapter, "_row_to_block", counting)

        results = db.query_blocks({"text_search": "notebook", "limit": 10})
        assert len(results) == 10
        assert calls["n"] <= 10

        calls["n"] = 0
        results = db.query_blocks(
            {"text_search": "notebook", "sort_by": "created_at", "limit": 7}
        )
        assert len(results) == 7
        assert calls["n"] <= 7


# ─── (d) plan guard ──────────────────────────────────────────────────────────


FILTER_COMBOS = {
    "none": {},
    "type": {"type": BlockType.FACT},
    "session_id": {"session_id": "s1"},
    "min_confidence": {"min_confidence": 0.5},
    "tags": {"tags": ["alpha"]},
    "type+session+conf+tags": {
        "type": BlockType.FACT,
        "session_id": "s1",
        "min_confidence": 0.5,
        "tags": ["alpha"],
    },
}


def _assert_fts_first(details: list[str], sql: str) -> None:
    assert details, f"empty plan for:\n{sql}"
    first = details[0]
    # The derived table is either flattened into a direct FTS scan or kept as a
    # co-routine; either way it must be the outermost loop.
    assert "blocks_fts" in first or first.upper().startswith("CO-ROUTINE"), (
        f"FTS scan is not the outer loop. Plan:\n" + "\n".join(details) + f"\nSQL:\n{sql}"
    )
    # Nothing may drive from blocks via the deleted index.
    assert not any("idx_blocks_deleted" in d for d in details), (
        "plan drives from idx_blocks_deleted (per-row MATCH re-evaluation). Plan:\n"
        + "\n".join(details)
    )
    # The blocks table (alias `b`) must be looked up by primary key, not scanned.
    # Note: `SCAN blocks_fts` also starts with "SCAN b", hence the word boundary.
    assert not any(re.match(r"^SCAN b(\s|$)", d) for d in details), (
        "plan scans blocks instead of probing by id. Plan:\n" + "\n".join(details)
    )


class TestQueryPlanGuard:
    @pytest.fixture(autouse=True)
    def _seed(self, db: SQLiteAdapter):
        # Enough rows that the planner has real stats to reason about,
        # well under the 5k experiment ceiling.
        for i in range(200):
            _save(
                db,
                f"alpha beta gamma row {i}" if i % 2 == 0 else f"delta epsilon row {i}",
                minutes=i,
                confidence=(i % 10) / 10,
                session_id=f"s{i % 3}",
                tags=["alpha"] if i % 4 == 0 else [],
            )
        # Deliberately NO `ANALYZE`: with planner statistics even the old
        # v0.13.1 shape plans FTS-first, so the guard would never trip. The
        # hazard (driving from idx_blocks_deleted and re-running MATCH per
        # row) shows up on a fresh, un-analysed database — the production
        # case — which is what this fixture reproduces.

    @pytest.mark.parametrize("combo", list(FILTER_COMBOS.keys()))
    @pytest.mark.parametrize("sort_by", [None, "relevance", "created_at", "confidence"])
    def test_fts_is_outer_loop(self, db: SQLiteAdapter, combo: str, sort_by):
        filters = {"text_search": "alpha beta", "limit": 10, **FILTER_COMBOS[combo]}
        if sort_by is not None:
            filters["sort_by"] = sort_by

        details, sql, params = _plan(db, filters)
        _assert_fts_first(details, sql)

        # MATCH param is first; LIMIT param is last.
        assert params[0] == '"alpha" OR "beta"'
        assert params[-1] == 10

        # And the query actually runs and honors the filters.
        results = db.query_blocks(filters)
        assert len(results) <= 10
        for b in results:
            if "type" in filters:
                assert b.type == filters["type"]
            if "session_id" in filters:
                assert b.metadata.session_id == filters["session_id"]
            if "min_confidence" in filters:
                assert b.metadata.confidence >= filters["min_confidence"]
            if "tags" in filters:
                assert any(t in b.tags for t in filters["tags"])

    def test_relevance_sorts_by_rank_then_newest(self, db: SQLiteAdapter):
        # ORDER BY fts.rank, b.created_at DESC: best bm25 first, newest first
        # on ties (matches Postgres). The tiebreak needs a temp b-tree over
        # the matched rows, but the FTS scan must remain the outer loop.
        details, sql, _ = _plan(db, {"text_search": "alpha", "limit": 10})
        _assert_fts_first(details, sql)
        assert "ORDER BY fts.rank, b.created_at DESC" in sql

    def test_equal_rank_ties_come_back_newest_first(self, db: SQLiteAdapter):
        old = _save(db, "favourite color is blue", minutes=0)
        new = _save(db, "favourite color is red", minutes=10_000)  # _at(): larger = newer
        ids = [b.id for b in db.query_blocks({"text_search": "favourite color", "limit": 5})]
        assert ids.index(new.id) < ids.index(old.id)

    def test_plan_without_limit_is_still_fts_first(self, db: SQLiteAdapter):
        for combo in FILTER_COMBOS.values():
            details, sql, params = _plan(db, {"text_search": "alpha", **combo})
            _assert_fts_first(details, sql)
            assert "LIMIT" not in sql

    def test_non_text_query_sql_unchanged(self, db: SQLiteAdapter):
        sql, params = db._build_query_sql({"type": BlockType.FACT, "limit": 3})
        assert "blocks_fts" not in sql
        assert "MATCH" not in sql
        assert "ORDER BY b.created_at DESC" in sql
        assert params == ["fact", 3]


# ─── (e) timing guard ────────────────────────────────────────────────────────


class TestTimingGuard:
    def test_hot_query_with_limit_is_fast(self, db: SQLiteAdapter):
        # Plain wall-clock asserts: a regression makes this test fail slowly
        # (~14 s on v0.13.1) instead of hard-killing the pytest process.
        if True:
            n = 4000
            for i in range(n):
                content = (
                    f"python snippet number {i} for the shared python toolbox"
                    if i % 5  # 80% of rows match "python"
                    else f"rust snippet {i}"
                )
                _save(db, content, minutes=i, session_id=f"s{i % 4}")

            # Warm-up run so we measure the query, not first-touch page loads.
            db.query_blocks({"text_search": "python", "limit": 10})

            t0 = time.perf_counter()
            results = db.query_blocks({"text_search": "python", "limit": 10})
            elapsed_ms = (time.perf_counter() - t0) * 1000
            assert len(results) == 10
            assert elapsed_ms < 300, f"hot text query took {elapsed_ms:.1f} ms"

            # Same with a metadata filter and an explicit sort.
            t0 = time.perf_counter()
            results = db.query_blocks(
                {"text_search": "python", "session_id": "s2", "sort_by": "created_at", "limit": 10}
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            assert len(results) == 10
            assert all(b.metadata.session_id == "s2" for b in results)
            assert elapsed_ms < 300, f"filtered hot text query took {elapsed_ms:.1f} ms"


# ─── (f) deleted blocks excluded ─────────────────────────────────────────────


class TestDeletedExcluded:
    def test_soft_deleted_block_not_returned(self, db: SQLiteAdapter):
        keep = _save(db, "graphql schema stitching", minutes=0)
        gone = _save(db, "graphql schema federation", minutes=1)

        db.update_block(gone.id, {"deleted": True})

        for filters in (
            {"text_search": "graphql"},
            {"text_search": "graphql", "limit": 10},
            {"text_search": "graphql", "sort_by": "relevance", "limit": 10},
            {"text_search": "graphql", "sort_by": "created_at", "limit": 10},
        ):
            ids = [b.id for b in db.query_blocks(filters)]
            assert ids == [keep.id], filters

    def test_deleted_flag_guard_even_if_fts_row_lingers(self, db: SQLiteAdapter):
        # Simulate an FTS row that was not cleaned up: b.deleted = 0 must still
        # exclude it from text search results.
        keep = _save(db, "terraform module registry", minutes=0)
        stale = _save(db, "terraform module versioning", minutes=1)
        db.conn.execute("UPDATE blocks SET deleted = 1 WHERE id = ?", (stale.id,))
        db.conn.commit()

        ids = [b.id for b in db.query_blocks({"text_search": "terraform", "limit": 10})]
        assert ids == [keep.id]

        # Explicit deleted=True lifts the guard.
        ids = {b.id for b in db.query_blocks({"text_search": "terraform", "deleted": True})}
        assert ids == {keep.id, stale.id}


# ─── (g) no word characters ──────────────────────────────────────────────────


class TestNoWordCharacters:
    @pytest.mark.parametrize("text", ["!!! ???", "   ", "-- ** //", "\"'\"", ""])
    def test_returns_empty_without_raising(self, db: SQLiteAdapter, text: str):
        _save(db, "some content that exists", minutes=0)

        assert db.query_blocks({"text_search": text}) == []
        assert db.query_blocks({"text_search": text, "limit": 5}) == []
        assert db.query_blocks({"text_search": text, "sort_by": "relevance", "limit": 5}) == []
        assert db.query_blocks({"text_search": text, "sort_by": "created_at"}) == []

        sql, params = db._build_query_sql({"text_search": text, "limit": 5})
        assert params[0] == '""'

    def test_fts_special_characters_are_stripped(self, db: SQLiteAdapter):
        block = _save(db, "user likes c++ and rust", minutes=0)
        # Would be an FTS5 syntax error if passed through unquoted.
        results = db.query_blocks({"text_search": "rust* AND (c++) NOT \"x\"", "limit": 5})
        assert block.id in {b.id for b in results}
