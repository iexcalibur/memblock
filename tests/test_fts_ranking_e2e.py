"""End-to-end retrieval ranking through the public API (v0.13.2).

The adapter-level tests (tests/test_fts_ranking_sqlite.py,
tests/test_fts_ranking_postgres.py) and the engine-level tests
(tests/test_query_pool.py) each pin one layer with mocks or raw filters.
These tests go through ``MemBlock`` / ``AsyncMemBlock`` on
``sqlite:///:memory:`` -- and through the sync/async query engines on a
live Postgres schema when ``MEMBLOCK_TEST_DB_URL`` is set -- and check the
user-visible outcome:

  (i)   FTS-only: a 30-day-old exact-phrase block outranks 40 newer
        one-token blocks via ``mem.query(text_search=...)``;
  (ii)  the same through ``AsyncMemBlock`` (legacy SQLite mode);
  (iii) hybrid: the keyword-best and the semantically-closest block are
        different rows and both land in the top 3;
  (iv)  ``build_context(query=...)`` reflects the new ordering;
  (v)   a type-filtered text search with a limit still fills;
  (vi)  an explicit ``sort_by="recency"`` text search is still newest-first.
"""

from __future__ import annotations

import asyncio
import importlib
import types
from datetime import timedelta
from unittest.mock import MagicMock

import pytest

from memblock import AsyncMemBlock, BlockType, MemBlock
from memblock.async_query import AsyncQueryEngine
from memblock.block import Block
from memblock.decay import DecayEngine
from memblock.embeddings import CallableEmbeddingProvider
from memblock.query import QueryEngine
from memblock.types import BlockMetadata, now_utc

# ─── Corpus ────────────────────────────────────────────────────────────

# No temporal words (the engine runs TemporalQueryParser over the query).
QUERY = "kubernetes cluster deploy"
BEST = "kubernetes cluster deploy checklist for the staging cluster rollout"
ONE_TOKEN = [
    "kubernetes upgrade notes for node pool {i}",
    "cluster maintenance reminder number {i}",
    "deploy freeze announcement for release {i}",
]

# Hybrid corpus: KEYWORD_BEST holds every query term but a weak vector;
# SEMANTIC_BEST shares no query term but has the query's vector.
QUERY_H = "cat food brand"
KEYWORD_BEST = "cat food brand comparison for a picky cat"
SEMANTIC_BEST = "kitten kibble recommendations"
HYBRID_DECOYS = [
    "brand new bicycle for commuting {i}",
    "food truck festival lineup {i}",
    "cat photo album from the trip {i}",
]


def _hybrid_embed(texts: list[str]) -> list[list[float]]:
    out = []
    for t in texts:
        if t in (QUERY_H, SEMANTIC_BEST):
            out.append([1.0, 0.0, 0.0, 0.0])
        elif t == KEYWORD_BEST:
            out.append([0.5, 1.0, 0.0, 0.0])  # cosine ~0.45 to the query
        else:
            out.append([0.0, 0.0, 1.0, 0.0])  # orthogonal
    return out


def _backdate(storage, block: Block, days: int) -> None:
    """Re-save ``block`` with an older ``created_at`` (bypasses the op log,
    which is fine for ranking tests)."""
    block.metadata.created_at = now_utc() - timedelta(days=days)
    # save_block() replaces the row and drops its embedding; keep it so a
    # backdated block stays a hybrid candidate (production edits go through
    # MemBlock.update(), which re-embeds).
    embedding = storage.get_embedding(block.id)
    storage.save_block(block)
    if embedding is not None:
        storage.save_embedding(block.id, embedding)


def _decoy(i: int) -> str:
    return ONE_TOKEN[i % len(ONE_TOKEN)].format(i=i)


def _seed_fts(mem: MemBlock, decoys: int = 40) -> tuple[Block, list[Block]]:
    best = mem.store(BEST, type=BlockType.FACT)
    _backdate(mem._storage, best, days=30)
    stored = [mem.store(_decoy(i), type=BlockType.FACT) for i in range(decoys)]
    return best, stored


# ─── (i), (iv), (v), (vi): MemBlock on SQLite ──────────────────────────


class TestMemBlockSqlite:
    def test_old_exact_match_outranks_forty_newer_one_token_blocks(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        try:
            best, decoys = _seed_fts(mem)
            results = mem.query(text_search=QUERY, limit=10)

            assert len(results) == 10
            assert results[0].id == best.id
            decoy_ids = {d.id for d in decoys}
            assert all(r.id in decoy_ids for r in results[1:])
        finally:
            mem.close()

    def test_semantic_false_takes_the_same_fts_only_path(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        try:
            best, _ = _seed_fts(mem)
            results = mem.query(text_search=QUERY, limit=10, semantic=False)
            assert results[0].id == best.id
        finally:
            mem.close()

    def test_explicit_recency_sort_is_still_newest_first(self):
        """(vi) ``sort_by="recency"`` asks storage for the newest matches,
        so the 30-day-old best match does not appear and the page is in
        ``created_at`` order (v0.13.1 semantics)."""
        mem = MemBlock(storage="sqlite:///:memory:")
        try:
            best, _ = _seed_fts(mem)
            results = mem.query(text_search=QUERY, sort_by="recency", limit=10)

            assert len(results) == 10
            assert best.id not in {r.id for r in results}
            created = [r.metadata.created_at for r in results]
            assert created == sorted(created, reverse=True)
        finally:
            mem.close()

    def test_build_context_reflects_the_ranking(self):
        """(iv) The relevance strategy lists the best keyword match first."""
        mem = MemBlock(storage="sqlite:///:memory:")
        try:
            best, decoys = _seed_fts(mem)
            ctx = mem.build_context(query=QUERY)

            assert BEST in ctx
            decoy_positions = [
                ctx.index(d.content) for d in decoys if d.content in ctx
            ]
            assert decoy_positions, "context should include some decoys too"
            assert ctx.index(BEST) < min(decoy_positions)
        finally:
            mem.close()

    def test_type_filtered_text_search_with_limit_still_fills(self):
        """(v) 60 newer, better-matching PREFERENCE blocks must not crowd
        the 15 weaker FACT matches out of the candidate pool."""
        mem = MemBlock(storage="sqlite:///:memory:")
        try:
            facts = [mem.store(_decoy(i), type=BlockType.FACT) for i in range(15)]
            for i in range(60):
                mem.store(
                    f"kubernetes cluster deploy preference number {i}",
                    type=BlockType.PREFERENCE,
                )

            results = mem.query(text_search=QUERY, type=BlockType.FACT, limit=10)
            assert len(results) == 10
            assert all(r.type is BlockType.FACT for r in results)
            assert {r.id for r in results} <= {f.id for f in facts}

            # Sanity: unfiltered, the better PREFERENCE matches win.
            unfiltered = mem.query(text_search=QUERY, limit=10)
            assert all(r.type is BlockType.PREFERENCE for r in unfiltered)
        finally:
            mem.close()


# ─── (ii): AsyncMemBlock (legacy SQLite mode) ──────────────────────────


class TestAsyncMemBlockSqlite:
    def test_old_exact_match_outranks_forty_newer_one_token_blocks(self):
        async def _test():
            async with AsyncMemBlock(storage="sqlite:///:memory:") as mem:
                best = await mem.store(BEST, type=BlockType.FACT)
                _backdate(mem._mem._storage, best, days=30)
                for i in range(40):
                    await mem.store(_decoy(i), type=BlockType.FACT)

                results = await mem.query(text_search=QUERY, limit=10)
                assert len(results) == 10
                assert results[0].id == best.id

                newest = await mem.query(text_search=QUERY, sort_by="recency", limit=10)
                assert best.id not in {r.id for r in newest}

        asyncio.run(_test())


# ─── (iii): hybrid with deterministic vectors ──────────────────────────


class TestHybridSqlite:
    def _mem(self) -> MemBlock:
        mem = MemBlock(storage="sqlite:///:memory:")
        provider = CallableEmbeddingProvider(_hybrid_embed, dims=4)
        mem._embedding_provider = provider
        mem._query._embedding_provider = provider
        return mem

    def test_keyword_best_and_semantic_best_both_in_top_three(self):
        mem = self._mem()
        try:
            # KEYWORD_BEST is stored first and backdated so recency order
            # (v0.13.1) would NOT put it first: the FTS-only control below
            # only passes when storage ranks by bm25.
            keyword_best = mem.store(KEYWORD_BEST, type=BlockType.FACT)
            _backdate(mem._storage, keyword_best, days=30)
            for i in range(30):
                mem.store(HYBRID_DECOYS[i % 3].format(i=i), type=BlockType.FACT)
            semantic_best = mem.store(SEMANTIC_BEST, type=BlockType.FACT)
            assert keyword_best.id != semantic_best.id
            assert mem._storage.get_embedding(semantic_best.id) is not None

            top = mem.query(text_search=QUERY_H, limit=10)
            top3 = {b.id for b in top[:3]}
            assert keyword_best.id in top3
            assert semantic_best.id in top3

            # Control: FTS-only cannot see SEMANTIC_BEST (no shared token)
            # but still puts the keyword-best row first.
            fts_only = mem.query(text_search=QUERY_H, limit=10, semantic=False)
            assert fts_only[0].id == keyword_best.id
            assert semantic_best.id not in {b.id for b in fts_only}
        finally:
            mem.close()


# ─── Live Postgres: sync + async query engines on the real adapters ────


def _real_pg_module():
    """``memblock.storage.postgresql`` with a real psycopg (see the autouse
    fixture in tests/test_connection_pool.py / tests/test_pgvector.py)."""
    import memblock.storage.postgresql as pg_mod

    if not isinstance(getattr(pg_mod, "psycopg", None), types.ModuleType):
        importlib.reload(pg_mod)
    return pg_mod


def _pg_blocks() -> tuple[Block, list[Block]]:
    best = Block(
        content=BEST,
        metadata=BlockMetadata(created_at=now_utc() - timedelta(days=30)),
    )
    decoys = [Block(content=_decoy(i)) for i in range(40)]
    prefs = [
        Block(
            content=f"kubernetes cluster deploy preference number {i}",
            type=BlockType.PREFERENCE,
        )
        for i in range(60)
    ]
    return best, decoys + prefs


class TestLivePostgresEngines:
    async def test_sync_engine_on_psycopg_adapter(self, postgres_sync_url, fresh_schema):
        import psycopg

        with psycopg.connect(postgres_sync_url, autocommit=True) as conn:
            conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{fresh_schema}"')
        pg_mod = _real_pg_module()
        adapter = pg_mod.PostgreSQLAdapter(
            dsn=postgres_sync_url, user_id="u_e2e", schema=fresh_schema,
        )
        adapter.initialize()
        try:
            best, others = _pg_blocks()
            adapter.save_block(best)
            for b in others:
                adapter.save_block(b)
            engine = QueryEngine(adapter, graph=MagicMock(), decay=DecayEngine(adapter))

            results = engine.query(text_search=QUERY, type=BlockType.FACT, limit=10)
            assert len(results) == 10
            assert results[0].id == best.id
            assert all(r.type is BlockType.FACT for r in results)

            newest = engine.query(text_search=QUERY, sort_by="recency", limit=10)
            assert best.id not in {r.id for r in newest}
        finally:
            adapter.close()

    async def test_async_engine_on_asyncpg_adapter(self, postgres_async_url, fresh_schema):
        from memblock.storage.async_postgresql import AsyncPostgreSQLAdapter

        adapter = AsyncPostgreSQLAdapter(
            dsn=postgres_async_url, user_id="u_e2e", schema=fresh_schema,
            pool_min_size=1, pool_max_size=2,
        )
        await adapter.initialize()
        try:
            best, others = _pg_blocks()
            await adapter.save_block(best)
            for b in others:
                await adapter.save_block(b)
            engine = AsyncQueryEngine(adapter, decay=DecayEngine(adapter))

            results = await engine.query(text_search=QUERY, type=BlockType.FACT, limit=10)
            assert len(results) == 10
            assert results[0].id == best.id
            assert all(r.type is BlockType.FACT for r in results)

            newest = await engine.query(text_search=QUERY, sort_by="recency", limit=10)
            assert best.id not in {r.id for r in newest}
        finally:
            await adapter.close()


# ─── Review regressions (v0.13.2 review pass) ──────────────────────────


class TestReviewRegressions:
    """Behaviours the v0.13.2 review found missing or regressed."""

    def test_min_strength_refills_from_the_full_match_set(self):
        # The SQL pool (50) is cut before the Python-side min_strength filter.
        # 60 decayed exact matches fill the pool; 30 live weaker matches sit
        # beyond it. v0.13.1 returned the live ones; the pool cut alone
        # returns nothing — the refill must restore the v0.13.1 result.
        mem = MemBlock(storage="sqlite:///:memory:")
        try:
            for i in range(60):
                b = mem.store(f"kotak xirr report {i} kotak xirr", type=BlockType.FACT, decay_rate=1.0)
                _backdate(mem._storage, b, days=30)
            fresh = {mem.store(f"xirr note {i}", type=BlockType.FACT).id for i in range(30)}
            results = mem.query(text_search="kotak xirr", min_strength=0.5, limit=10)
            assert len(results) == 10
            assert {b.id for b in results} <= fresh
        finally:
            mem.close()

    def test_strength_sort_with_text_search_sees_every_match(self):
        # sort_by="strength" has no SQL equivalent, so the engine must not
        # cap the candidate set: a weak-keyword but strong block beyond the
        # top-50 bm25 pool still wins (v0.13.1 semantics).
        mem = MemBlock(storage="sqlite:///:memory:")
        try:
            for i in range(60):
                b = mem.store(f"alpha beta gamma {i}", type=BlockType.FACT)
                _backdate(mem._storage, b, days=20)
            strong = mem.store("gamma only", type=BlockType.FACT)
            results = mem.query(text_search="alpha beta gamma", sort_by="strength", limit=5)
            assert results[0].id == strong.id
        finally:
            mem.close()

    def test_vector_only_hit_respects_session_filter(self):
        # A hybrid query scoped to one session must not surface a block from
        # another session just because its vector is close.
        def embed(texts):
            return [[1.0, 0.0, 0.0] if "needle" in t else [0.0, 1.0, 0.0] for t in texts]
        mem = MemBlock(storage="sqlite:///:memory:")
        provider = CallableEmbeddingProvider(embed, dims=3)
        mem._embedding_provider = provider
        mem._query._embedding_provider = provider
        try:
            other = mem.store("needle in the haystack", type=BlockType.FACT, session_id="other")
            mine = mem.store("haystack maintenance", type=BlockType.FACT, session_id="mine")
            results = mem.query(text_search="needle", session_id="mine", limit=10)
            ids = {b.id for b in results}
            assert other.id not in ids
            assert ids <= {mine.id}
        finally:
            mem.close()

    def test_reranker_receives_the_top_of_the_relevance_pool(self):
        from memblock.rerankers import CallableReranker

        seen: list[list[Block]] = []

        def fn(query, blocks, top_k):
            seen.append(list(blocks))
            return blocks[:top_k]

        mem = MemBlock(storage="sqlite:///:memory:", reranker=CallableReranker(fn))
        try:
            for i in range(200):
                mem.store(f"python snippet number {i}", type=BlockType.FACT)
            results = mem.query(text_search="python snippet", limit=10)
            assert len(results) == 10
            assert len(seen) == 1
            assert len(seen[0]) == 30  # limit * 3, drawn from the 50-row pool
        finally:
            mem.close()


class TestInferredTemporalWindowFallback:
    """A temporal phrase in the query ("last week") becomes a happened_at
    window relative to now. When nothing falls inside it — the store holds
    only older memories, or blocks carry no happened_at — the engine must
    fall back to the unconstrained search instead of returning nothing
    (v0.13.1 only avoided this through an unfiltered vector-hydration
    leak that v0.13.2 closes)."""

    def test_window_with_no_matches_falls_back_to_full_search(self):
        from datetime import datetime, timezone

        mem = MemBlock(storage="sqlite:///:memory:")
        try:
            old = datetime(2023, 5, 20, tzinfo=timezone.utc)
            best = mem.store("I bought a new air fryer for the kitchen", type=BlockType.FACT, happened_at=old)
            mem.store("The weather was nice in the park", type=BlockType.FACT, happened_at=old)
            mem.store("Kitchen renovation quotes were expensive", type=BlockType.FACT)  # no happened_at
            results = mem.query(text_search="what kitchen appliance did I buy last week", limit=5)
            assert results, "inferred temporal window must not empty the result set"
            assert results[0].id == best.id
        finally:
            mem.close()

    def test_window_with_matches_is_still_honoured(self):
        from datetime import timedelta

        mem = MemBlock(storage="sqlite:///:memory:")
        try:
            recent = mem.store("I bought a kettle for the kitchen", type=BlockType.FACT, happened_at=now_utc() - timedelta(days=2))
            stale = mem.store("I bought a toaster for the kitchen", type=BlockType.FACT, happened_at=now_utc() - timedelta(days=400))
            results = mem.query(text_search="what did I buy for the kitchen last week", limit=5)
            ids = [b.id for b in results]
            assert ids == [recent.id], ids  # the stale block is outside the window
        finally:
            mem.close()

    def test_async_engine_falls_back_too(self):
        from datetime import datetime, timezone

        async def _run():
            mem = AsyncMemBlock(storage="sqlite:///:memory:")
            try:
                old = datetime(2023, 5, 20, tzinfo=timezone.utc)
                best = await mem.store("I bought a new air fryer for the kitchen", type=BlockType.FACT, happened_at=old)
                await mem.store("The weather was nice in the park", type=BlockType.FACT, happened_at=old)
                results = await mem.query(text_search="what kitchen appliance did I buy last week", limit=5)
                assert results and results[0].id == best.id
            finally:
                await mem.close()

        asyncio.run(_run())
