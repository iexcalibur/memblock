"""Full-text relevance ranking in the Postgres adapters' ``query_blocks``.

Covers the sync ``PostgreSQLAdapter`` (psycopg) and the native
``AsyncPostgreSQLAdapter`` (asyncpg) through one parametrized harness so
the two stay behaviourally identical. The storage contract under test:

- ``text_search`` with ``sort_by`` absent or ``"relevance"`` orders by
  ``ts_rank`` DESC (tiebreak ``created_at`` DESC), so an old block that
  matches most of the query beats a newer block sharing a single token.
- explicit ``sort_by="created_at"`` / ``"confidence"`` keep their order.
- ``limit`` is applied by the same statement, after every WHERE
  condition, so a filtered text query never under-fills.
- soft-deleted rows never surface.
- over a 10k-row table the ``content_tsv @@ to_tsquery(...)`` predicate
  is served by the GIN index ``idx_mb_blocks_tsv``.

The ``TestQueryShape`` unit tests run anywhere. Everything else needs a
live database and skips cleanly when ``MEMBLOCK_TEST_DB_URL`` is unset
(see ``tests/conftest.py``).
"""

from __future__ import annotations

import importlib
import time
import types
from datetime import timedelta
from typing import Any

import pytest

from memblock.block import Block
from memblock.storage.async_postgresql import HAS_ASYNCPG, AsyncPostgreSQLAdapter
from memblock.storage.postgresql import HAS_PSYCOPG, PostgreSQLAdapter
from memblock.types import BlockMetadata, BlockType, now_utc


# ─── Helpers ─────────────────────────────────────────────────────────


def _real_postgresql_module():
    """Return ``memblock.storage.postgresql`` with a real ``psycopg``.

    ``tests/test_connection_pool.py`` and ``tests/test_pgvector.py`` reload
    that module with a ``MagicMock`` psycopg in ``sys.modules`` and never
    reload it back, so in a full-suite run its ``psycopg`` global stays a
    mock and ``adapter.conn`` returns a MagicMock. Reloading with the real
    driver restores the module globals in place (methods of the already
    imported ``PostgreSQLAdapter`` resolve through the same dict).
    """
    import memblock.storage.postgresql as pg_mod

    if not isinstance(getattr(pg_mod, "psycopg", None), types.ModuleType):
        importlib.reload(pg_mod)
    return pg_mod


QUERY = "postgres connection pool timeout"
QUERY_WORDS = QUERY.split()


def _block(
    content: str,
    *,
    created_at,
    type: BlockType = BlockType.FACT,
    session_id: str | None = None,
    confidence: float = 1.0,
    tags: list[str] | None = None,
) -> Block:
    return Block(
        type=type,
        content=content,
        tags=tags or [],
        metadata=BlockMetadata(
            created_at=created_at,
            session_id=session_id,
            confidence=confidence,
        ),
    )


class _SyncHarness:
    """Drives the psycopg adapter behind an async facade so the same
    test bodies run against both adapters."""

    kind = "psycopg"

    def __init__(self, url: str, schema: str) -> None:
        import psycopg

        # The sync adapter does not create its schema (the async one does).
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        self.schema = schema
        pg_mod = _real_postgresql_module()
        self.adapter = pg_mod.PostgreSQLAdapter(dsn=url, user_id="u_fts", schema=schema)
        self.adapter.initialize()

    async def save(self, block: Block) -> None:
        self.adapter.save_block(block)

    async def query(self, filters: dict[str, Any]) -> list[Block]:
        return self.adapter.query_blocks(filters)

    async def soft_delete(self, block_id: str) -> None:
        # Same path `BlockStore.delete` uses.
        self.adapter.update_block(block_id, {"deleted": True})

    async def bulk_fill(self, n: int) -> None:
        with self.adapter.conn.cursor() as cur:
            cur.execute(_FILL_BLOCKS.format(schema=self.schema, n=n))
            cur.execute(_FILL_META.format(schema=self.schema, n=n))
        self.adapter.conn.commit()

    async def maintain(self) -> None:
        """What autovacuum would eventually do: flush the GIN pending
        list and refresh planner stats."""
        with self.adapter.conn.cursor() as cur:
            cur.execute(
                f"SELECT gin_clean_pending_list('{self.schema}.idx_mb_blocks_tsv'::regclass)"
            )
            cur.execute(f"ANALYZE {self.schema}.memblock_blocks")
            cur.execute(f"ANALYZE {self.schema}.memblock_metadata")
        self.adapter.conn.commit()

    async def explain(self, filters: dict[str, Any]) -> str:
        sql, params = self.adapter._build_query_blocks_sql(filters)
        conn = self.adapter.conn
        with conn.cursor() as cur:
            cur.execute("EXPLAIN " + sql, params)
            rows = cur.fetchall()
        conn.commit()
        return "\n".join(next(iter(row.values())) for row in rows)

    async def close(self) -> None:
        # Release the connection (and its open transaction) so the
        # fixture's DROP SCHEMA ... CASCADE is not blocked by our locks.
        self.adapter.close()


class _AsyncHarness:
    kind = "asyncpg"

    def __init__(self, url: str, schema: str) -> None:
        self.schema = schema
        self.adapter = AsyncPostgreSQLAdapter(
            dsn=url, user_id="u_fts", schema=schema,
            pool_min_size=1, pool_max_size=2,
        )

    async def setup(self) -> None:
        await self.adapter.initialize()

    async def save(self, block: Block) -> None:
        await self.adapter.save_block(block)

    async def query(self, filters: dict[str, Any]) -> list[Block]:
        return await self.adapter.query_blocks(filters)

    async def soft_delete(self, block_id: str) -> None:
        await self.adapter.update_block(block_id, {"deleted": True})

    async def bulk_fill(self, n: int) -> None:
        pool = await self.adapter._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(_FILL_BLOCKS.format(schema=self.schema, n=n))
            await conn.execute(_FILL_META.format(schema=self.schema, n=n))

    async def maintain(self) -> None:
        pool = await self.adapter._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"SELECT gin_clean_pending_list('{self.schema}.idx_mb_blocks_tsv'::regclass)"
            )
            await conn.execute(f"ANALYZE {self.schema}.memblock_blocks")
            await conn.execute(f"ANALYZE {self.schema}.memblock_metadata")

    async def explain(self, filters: dict[str, Any]) -> str:
        sql, params = self.adapter._build_query_blocks_sql(filters)
        pool = await self.adapter._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("EXPLAIN " + sql, *params)
        return "\n".join(row[0] for row in rows)

    async def close(self) -> None:
        await self.adapter.close()


@pytest.fixture(params=["psycopg", "asyncpg"])
async def harness(request, postgres_sync_url: str, postgres_async_url: str, fresh_schema: str):
    """One adapter per param, each in its own pristine schema."""
    if request.param == "psycopg":
        if not HAS_PSYCOPG:
            pytest.skip("psycopg not installed")
        h: _SyncHarness | _AsyncHarness = _SyncHarness(postgres_sync_url, fresh_schema)
    else:
        if not HAS_ASYNCPG:
            pytest.skip("asyncpg not installed")
        h = _AsyncHarness(postgres_async_url, fresh_schema)
        await h.setup()
    yield h
    await h.close()


async def _seed_relevance(h) -> tuple[Block, list[Block]]:
    """One 10-day-old block matching all four query terms, then 30 newer
    blocks that each share exactly one token with the query."""
    base = now_utc()
    strong = _block(
        "The postgres connection pool timeout was raised to thirty seconds",
        created_at=base - timedelta(days=10),
    )
    await h.save(strong)
    weak: list[Block] = []
    for i in range(30):
        word = QUERY_WORDS[i % len(QUERY_WORDS)]
        b = _block(
            f"Note {i}: remember to check the {word} later",
            created_at=base + timedelta(seconds=i),
        )
        await h.save(b)
        weak.append(b)
    return strong, weak


# ─── (a) Relevance ordering ──────────────────────────────────────────


class TestRelevanceOrdering:

    async def test_default_sort_is_relevance(self, harness):
        strong, weak = await _seed_relevance(harness)
        results = await harness.query({"text_search": QUERY})
        assert len(results) == 31
        assert results[0].id == strong.id, (
            f"[{harness.kind}] the old four-term match must rank first, got "
            f"{results[0].content!r}"
        )

    async def test_explicit_relevance_sort(self, harness):
        strong, weak = await _seed_relevance(harness)
        results = await harness.query({"text_search": QUERY, "sort_by": "relevance"})
        assert results[0].id == strong.id
        # Ties among the one-token matches break newest-first.
        assert [b.id for b in results[1:]] == [b.id for b in reversed(weak)]

    async def test_relevance_with_limit_keeps_best_first(self, harness):
        strong, _ = await _seed_relevance(harness)
        results = await harness.query({"text_search": QUERY, "limit": 5})
        assert len(results) == 5
        assert results[0].id == strong.id

    async def test_explicit_created_at_is_newest_first(self, harness):
        strong, weak = await _seed_relevance(harness)
        results = await harness.query({"text_search": QUERY, "sort_by": "created_at"})
        assert len(results) == 31
        assert results[0].id == weak[-1].id
        assert results[-1].id == strong.id
        stamps = [b.metadata.created_at for b in results]
        assert stamps == sorted(stamps, reverse=True)

    async def test_explicit_confidence_sort_keeps_its_order(self, harness):
        base = now_utc()
        # Rank and confidence pull in opposite directions.
        blocks = []
        for i in range(4):
            content = "signal " + " ".join(QUERY_WORDS[: i + 1])
            b = _block(content, created_at=base + timedelta(seconds=i), confidence=1.0 - i * 0.2)
            await harness.save(b)
            blocks.append(b)
        results = await harness.query({"text_search": QUERY, "sort_by": "confidence"})
        assert [b.id for b in results] == [b.id for b in blocks]

    async def test_non_text_query_ignores_relevance(self, harness):
        base = now_utc()
        blocks = [
            _block(f"plain block {i}", created_at=base + timedelta(seconds=i))
            for i in range(5)
        ]
        for b in blocks:
            await harness.save(b)
        # No text_search: "relevance" falls back to the created_at default.
        results = await harness.query({"sort_by": "relevance"})
        assert [b.id for b in results] == [b.id for b in reversed(blocks)]

    async def test_text_search_without_word_chars_matches_nothing(self, harness):
        base = now_utc()
        for i in range(3):
            await harness.save(_block(f"anything {i}", created_at=base + timedelta(seconds=i)))
        # No lexemes → nothing can match (parity with SQLite's MATCH '""');
        # must not error and must not fall back to "every row".
        results = await harness.query({"text_search": "!!! ???"})
        assert results == []


# ─── (b) No under-fill with filters + limit ──────────────────────────


_FILTER_CASES = {
    # name: (target kwargs, decoy kwargs, filter dict)
    "type": (
        {"type": BlockType.FACT},
        {"type": BlockType.EVENT},
        {"type": BlockType.FACT},
    ),
    "session_id": (
        {"session_id": "sess_target"},
        {"session_id": "sess_other"},
        {"session_id": "sess_target"},
    ),
    "tags": (
        {"tags": ["keep"]},
        {"tags": ["drop"]},
        {"tags": ["keep"]},
    ),
    "min_confidence": (
        {"confidence": 0.9},
        {"confidence": 0.3},
        {"min_confidence": 0.5},
    ),
}


class TestNoUnderfill:

    @pytest.mark.parametrize("case", sorted(_FILTER_CASES))
    async def test_limit_applies_after_filters(self, harness, case: str):
        target_kw, decoy_kw, extra_filters = _FILTER_CASES[case]
        base = now_utc()
        targets: list[Block] = []
        # 40 target rows matching 1..4 query terms (10 of each).
        for i in range(40):
            k = (i % 4) + 1
            b = _block(
                f"target {i} " + " ".join(QUERY_WORDS[:k]),
                created_at=base + timedelta(seconds=i),
                **target_kw,
            )
            await harness.save(b)
            targets.append(b)
        # 40 newer decoys that match all four terms — they outrank every
        # target, so a LIMIT applied before the filter would starve it.
        for i in range(40):
            await harness.save(_block(
                f"decoy {i} " + " ".join(QUERY_WORDS),
                created_at=base + timedelta(minutes=5, seconds=i),
                **decoy_kw,
            ))

        filters = {"text_search": QUERY, "limit": 10, **extra_filters}
        results = await harness.query(filters)

        assert len(results) == 10, f"[{harness.kind}] under-filled with {case} filter"
        target_ids = {b.id for b in targets}
        assert all(b.id in target_ids for b in results), "decoy leaked through filter"
        # The 10 returned must be the ten four-term targets, newest first.
        four_term = [b for b in targets if b.content.count(" ") >= 5]
        assert len(four_term) == 10
        expected = [b.id for b in reversed(four_term)]
        assert [b.id for b in results] == expected


# ─── (c) Limit respected ─────────────────────────────────────────────


class TestLimit:

    async def test_limit_caps_and_omitting_it_returns_all(self, harness):
        base = now_utc()
        for i in range(25):
            await harness.save(_block(
                f"entry {i} about the connection pool",
                created_at=base + timedelta(seconds=i),
            ))
        assert len(await harness.query({"text_search": QUERY, "limit": 7})) == 7
        assert len(await harness.query({"text_search": QUERY, "limit": 100})) == 25
        assert len(await harness.query({"text_search": QUERY})) == 25


# ─── (d) Deleted excluded ────────────────────────────────────────────


class TestDeleted:

    async def test_soft_deleted_best_match_never_surfaces(self, harness):
        base = now_utc()
        best = _block(
            "postgres connection pool timeout tuning notes",
            created_at=base - timedelta(days=1),
        )
        others = [
            _block(f"reminder {i} about the pool", created_at=base + timedelta(seconds=i))
            for i in range(2)
        ]
        for b in [best, *others]:
            await harness.save(b)

        before = await harness.query({"text_search": QUERY})
        assert before[0].id == best.id

        await harness.soft_delete(best.id)

        after = await harness.query({"text_search": QUERY})
        assert {b.id for b in after} == {b.id for b in others}
        after_limited = await harness.query({"text_search": QUERY, "limit": 1})
        assert after_limited[0].id != best.id
        # Opting in still sees it (existing semantics of deleted=True).
        with_deleted = await harness.query({"text_search": QUERY, "deleted": True})
        assert with_deleted[0].id == best.id


# ─── (e) GIN index usage ─────────────────────────────────────────────


HAYSTACK_ROWS = 10_000

# Filler rows go straight into the tables (the BEFORE INSERT trigger fills
# content_tsv exactly as it does for adapter writes); only the rows the
# assertions care about go through `save_block`. Every second filler row
# is in session "s_a" so a session filter is NOT the selective predicate.
_FILL_BLOCKS = """
    INSERT INTO {schema}.memblock_blocks
        (id, user_id, type, content, children_ids, tags, created_at, content_hash)
    SELECT 'fill_' || i, 'u_fts', 'fact',
           'row ' || i || ' unrelated filler content number ' || i,
           '[]', '[]', now() + (i || ' seconds')::interval, md5(i::text)
    FROM generate_series(1, {n}) AS i
"""
_FILL_META = """
    INSERT INTO {schema}.memblock_metadata
        (block_id, user_id, confidence, source, created_at, session_id)
    SELECT 'fill_' || i, 'u_fts', 1.0, 'explicit', now(),
           CASE WHEN MOD(i, 2) = 0 THEN 's_a' ELSE 's_b' END
    FROM generate_series(1, {n}) AS i
"""


class TestIndexUsage:
    """EXPLAINs the exact statement ``query_blocks`` runs over a ~10k-row
    table and asserts the planner serves the ``@@`` predicate from the GIN
    index ``idx_mb_blocks_tsv`` (a Bitmap Index Scan), then checks the
    query is fast and correctly ordered.

    Two planner facts shaped this test (measured on Postgres 15):

    - With the GIN pending list (fastupdate) unflushed after a bulk load,
      ``gincostestimate`` prices the index far above a seq scan even for a
      16-row estimate (cost 483 seq vs GIN not chosen at 10k rows). In
      production autovacuum's VACUUM flushes it; here
      ``gin_clean_pending_list`` + ANALYZE is the transaction-safe
      equivalent. After the flush the GIN plan costs 90 vs 483 for the
      seq scan, so the choice is stable.
    - At ~2k rows the natural choice is data-dependent (the same 40
      matches drew a 79-row estimate in one stats sample and 1 row in
      another), which is why the haystack is 10k rows, not 2k.
    """

    async def _seed_haystack(self, harness) -> Block:
        await harness.bulk_fill(HAYSTACK_ROWS)
        base = now_utc()
        needle = _block(
            "postgres connection pool timeout is the needle in this haystack",
            created_at=base - timedelta(days=30),
            session_id="s_a",
        )
        await harness.save(needle)
        for i in range(8):
            await harness.save(_block(
                f"hay {i} mentions the {QUERY_WORDS[i % 4]} once",
                created_at=base + timedelta(seconds=i),
                session_id="s_a" if i % 2 == 0 else "s_b",
            ))
        await harness.maintain()
        return needle

    async def test_gin_index_serves_the_fts_predicate(self, harness):
        needle = await self._seed_haystack(harness)
        filters = {"text_search": QUERY, "limit": 10}

        plan = await harness.explain(filters)
        assert "Bitmap Index Scan on idx_mb_blocks_tsv" in plan, (
            f"[{harness.kind}] FTS predicate not served by the GIN index:\n{plan}"
        )
        assert "ts_rank" in plan  # relevance sort is in the plan

        t0 = time.perf_counter()
        results = await harness.query(filters)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 500, f"[{harness.kind}] {elapsed_ms:.1f} ms:\n{plan}"
        assert len(results) == 9  # needle + 8 hay; limit 10 not reached
        assert results[0].id == needle.id
        assert all(b.id != needle.id for b in results[1:])

    async def test_filtered_query_with_limit_starts_at_the_gin_index(self, harness):
        """The metadata join (session_id filter on half the rows) must not
        stop the planner from starting at the GIN index."""
        needle = await self._seed_haystack(harness)
        filters = {"text_search": QUERY, "session_id": "s_a", "limit": 3}

        plan = await harness.explain(filters)
        assert "Bitmap Index Scan on idx_mb_blocks_tsv" in plan, (
            f"[{harness.kind}] filtered query does not use the GIN index:\n{plan}"
        )
        results = await harness.query(filters)
        assert len(results) == 3  # needle + 4 hay in s_a; limit applies after filter
        assert results[0].id == needle.id
        assert all(b.metadata.session_id == "s_a" for b in results)


# ─── SQL shape (no database needed) ──────────────────────────────────


class TestQueryShape:
    """Pure string checks on the built statement; run without a DB."""

    @pytest.mark.skipif(not HAS_PSYCOPG, reason="psycopg not installed")
    def test_sync_relevance_shape(self):
        adapter = PostgreSQLAdapter(dsn="postgresql://x/y", user_id="u", schema="s")
        sql, params = adapter._build_query_blocks_sql(
            {"text_search": "alpha beta", "type": BlockType.FACT, "limit": 50}
        )
        assert "b.content_tsv @@ to_tsquery('english', %s)" in sql
        order = sql.split("ORDER BY", 1)[1]
        assert order.lstrip().startswith("ts_rank(b.content_tsv, to_tsquery('english', %s)) DESC")
        assert "b.created_at DESC" in order
        assert sql.rstrip().endswith("LIMIT %s")
        # user_id, type, tsquery (WHERE), tsquery (ORDER BY), limit
        assert params == ["u", "fact", "alpha | beta", "alpha | beta", 50]

    @pytest.mark.skipif(not HAS_PSYCOPG, reason="psycopg not installed")
    @pytest.mark.parametrize("sort_by", ["created_at", "access_count", "confidence"])
    def test_sync_explicit_sort_has_no_ts_rank(self, sort_by: str):
        adapter = PostgreSQLAdapter(dsn="postgresql://x/y", user_id="u", schema="s")
        sql, params = adapter._build_query_blocks_sql(
            {"text_search": "alpha", "sort_by": sort_by}
        )
        assert "ts_rank" not in sql
        assert params == ["u", "alpha"]

    @pytest.mark.skipif(not HAS_PSYCOPG, reason="psycopg not installed")
    def test_sync_non_text_query_unchanged(self):
        adapter = PostgreSQLAdapter(dsn="postgresql://x/y", user_id="u", schema="s")
        sql, params = adapter._build_query_blocks_sql({"session_id": "s1"})
        assert "ts_rank" not in sql and "to_tsquery" not in sql
        assert "ORDER BY b.created_at DESC" in sql
        assert "LIMIT" not in sql
        assert params == ["u", "s1"]

    @pytest.mark.skipif(not HAS_ASYNCPG, reason="asyncpg not installed")
    def test_async_relevance_shape_reuses_placeholder(self):
        adapter = AsyncPostgreSQLAdapter(dsn="postgresql://x/y", user_id="u", schema="s")
        sql, params = adapter._build_query_blocks_sql(
            {"text_search": "alpha beta", "type": BlockType.FACT, "limit": 50}
        )
        assert "b.content_tsv @@ to_tsquery('english', $3)" in sql
        order = sql.split("ORDER BY", 1)[1]
        assert order.lstrip().startswith("ts_rank(b.content_tsv, to_tsquery('english', $3)) DESC")
        assert "b.created_at DESC" in order
        assert sql.rstrip().endswith("LIMIT $4")
        assert params == ["u", "fact", "alpha | beta", 50]

    @pytest.mark.skipif(not HAS_ASYNCPG, reason="asyncpg not installed")
    @pytest.mark.parametrize("sort_by", ["created_at", "access_count", "confidence"])
    def test_async_explicit_sort_has_no_ts_rank(self, sort_by: str):
        adapter = AsyncPostgreSQLAdapter(dsn="postgresql://x/y", user_id="u", schema="s")
        sql, params = adapter._build_query_blocks_sql(
            {"text_search": "alpha", "sort_by": sort_by}
        )
        assert "ts_rank" not in sql
        assert params == ["u", "alpha"]


# ─── Review regressions (v0.13.2 review pass) ───────────────────────


class TestReviewRegressions:
    async def test_no_word_text_search_matches_nothing(self, harness):
        # Parity with SQLite (MATCH '""'): a query with no searchable words
        # returns no rows instead of the newest `limit` rows.
        await harness.save(_block("some ordinary content", created_at=now_utc()))
        assert await harness.query({"text_search": "!!!", "limit": 5}) == []
        assert await harness.query({"text_search": "...", "sort_by": "created_at", "limit": 5}) == []

    async def test_tsvector_trigger_is_created_per_schema(
        self, harness, postgres_sync_url: str, fresh_schema: str
    ):
        # Regression for the unscoped `pg_trigger` name check: a second
        # schema in the same database got no trg_memblock_tsv, so its
        # content_tsv stayed NULL and text search returned nothing.
        if harness.kind != "psycopg":
            pytest.skip("sync adapter only")
        import psycopg

        second = _SyncHarness(postgres_sync_url, fresh_schema + "_b")
        try:
            await second.save(_block("the postgres trigger must exist in every schema", created_at=now_utc()))
            rows = await second.query({"text_search": "postgres trigger"})
            assert len(rows) == 1
            with psycopg.connect(postgres_sync_url) as conn:
                n = conn.execute(
                    "SELECT count(*) FROM pg_trigger WHERE tgname = 'trg_memblock_tsv' "
                    "AND tgrelid = %s::regclass",
                    (f"{fresh_schema}_b.memblock_blocks",),
                ).fetchone()[0]
            assert n == 1
        finally:
            await second.close()
            with psycopg.connect(postgres_sync_url, autocommit=True) as conn:
                conn.execute(f'DROP SCHEMA IF EXISTS "{fresh_schema}_b" CASCADE')
