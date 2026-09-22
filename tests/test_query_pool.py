"""Query-engine candidate pool and FTS-only ranking.

Covers the engine half of the retrieval fix:

  * text searches push ``sort_by="relevance"`` (``"created_at"`` /
    ``"access_count"`` for an explicit recency / access_count sort) and a
    bounded ``limit`` (the hybrid pool, ``max(limit * 5, 50)``) down to
    ``storage.query_blocks``; non-text queries push neither;
  * the FTS-only relevance curve is steep enough that the best keyword
    match beats recency/strength at any age;
  * the hybrid "add vector-only blocks" step hydrates at most ``pool``
    blocks instead of every embedded block;
  * when no vector signal is available (no provider, ``semantic=False``,
    or no embeddings) the keyword rank order returned by storage still
    drives ``relevance`` scoring instead of being discarded;
  * the hybrid (FTS + vector) path is unchanged;
  * the sync and async engines behave identically.

Storage is a ``MagicMock(spec=StorageAdapter)`` / ``MagicMock(spec=
AsyncStorageAdapter)`` so nothing here depends on the SQLite adapter's
SQL shape.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from unittest.mock import MagicMock

from memblock.async_query import AsyncQueryEngine
from memblock.block import Block
from memblock.decay import DecayEngine
from memblock.embeddings import CallableEmbeddingProvider, pack_embedding
from memblock.query import QueryEngine
from memblock.storage.async_base import AsyncStorageAdapter
from memblock.storage.base import StorageAdapter
from memblock.types import BlockMetadata, BlockType, now_utc


# ─── Helpers ───────────────────────────────────────────────────────────


def _block(
    block_id: str, minutes_ago: int, confidence: float = 0.8,
) -> Block:
    meta = BlockMetadata(
        confidence=confidence,
        created_at=now_utc() - timedelta(minutes=minutes_ago),
    )
    return Block(id=block_id, content="python", metadata=meta)


def _abc() -> list[Block]:
    """Three blocks in keyword-rank order A, B, C — same confidence,
    A oldest and C newest, so recency/strength alone would order them
    C, B, A."""
    return [_block("A", 3), _block("B", 2), _block("C", 1)]


def _provider() -> CallableEmbeddingProvider:
    return CallableEmbeddingProvider(
        lambda texts: [[1.0, 0.0, 0.0] for _ in texts], dims=3,
    )


def _sync_storage(candidates: list[Block]) -> MagicMock:
    storage = MagicMock(spec=StorageAdapter)
    storage.query_blocks.return_value = list(candidates)
    storage.search_similar_embeddings.return_value = []
    storage.get_all_embeddings.return_value = []
    storage.get_block.return_value = None
    return storage


def _async_storage(candidates: list[Block]) -> MagicMock:
    # spec'd coroutine methods become AsyncMocks automatically.
    storage = MagicMock(spec=AsyncStorageAdapter)
    storage.query_blocks.return_value = list(candidates)
    storage.search_similar_embeddings.return_value = []
    storage.get_all_embeddings.return_value = []
    storage.get_block.return_value = None
    return storage


def _sync_engine(storage: MagicMock, provider=None) -> QueryEngine:
    return QueryEngine(
        storage,
        graph=MagicMock(),
        decay=DecayEngine(storage=None),  # type: ignore[arg-type]
        embedding_provider=provider,
    )


def _async_engine(storage: MagicMock, provider=None) -> AsyncQueryEngine:
    return AsyncQueryEngine(
        storage,
        decay=DecayEngine(storage=None),  # type: ignore[arg-type]
        embedding_provider=provider,
    )


def _ids(blocks: list[Block]) -> list[str]:
    return [b.id for b in blocks]


def _filters(storage: MagicMock) -> dict:
    storage.query_blocks.assert_called_once()
    return storage.query_blocks.call_args.args[0]


# engine sort_by -> storage sort_by pushed for text searches
_SORT_MAP = [
    ("relevance", "relevance"),
    ("strength", "relevance"),
    ("recency", "created_at"),
    ("access_count", "access_count"),
]


def _embedded(n: int) -> tuple[dict[str, Block], list[tuple[str, bytes]]]:
    """n embedded blocks V0..V{n-1}; cosine to the query [1, 0, 0]
    strictly decreases with i, and V0 is the newest."""
    blocks: dict[str, Block] = {}
    rows: list[tuple[str, bytes]] = []
    for i in range(n):
        bid = f"V{i}"
        blocks[bid] = _block(bid, minutes_ago=i)
        rows.append((bid, pack_embedding([1.0, 0.01 * i, 0.0])))
    return blocks, rows


# ─── (a) Filters pushed to storage ─────────────────────────────────────


class TestCandidatePool:
    def test_text_search_pushes_relevance_sort_and_pool_limit(self):
        storage = _sync_storage(_abc())
        _sync_engine(storage).query(text_search="python", limit=10)

        filters = _filters(storage)
        assert filters["text_search"] == "python"
        assert filters["sort_by"] == "relevance"
        assert filters["limit"] == 50  # max(10 * 5, 50)

    def test_pool_scales_with_limit(self):
        storage = _sync_storage(_abc())
        _sync_engine(storage).query(text_search="python", limit=20)
        assert _filters(storage)["limit"] == 100  # max(20 * 5, 50)

    def test_pool_floor_is_fifty(self):
        storage = _sync_storage(_abc())
        _sync_engine(storage).query(text_search="python", limit=1)
        assert _filters(storage)["limit"] == 50

    def test_non_text_query_pushes_neither(self):
        storage = _sync_storage(_abc())
        _sync_engine(storage).query(type=BlockType.FACT, limit=10)

        filters = _filters(storage)
        assert filters == {"type": BlockType.FACT}
        assert "sort_by" not in filters
        assert "limit" not in filters

    def test_tags_only_query_pushes_neither(self):
        storage = _sync_storage(_abc())
        _sync_engine(storage).query(tags=["x"], limit=2)
        assert _filters(storage) == {"tags": ["x"]}

    def test_other_filters_still_forwarded_alongside_pool(self):
        storage = _sync_storage(_abc())
        _sync_engine(storage).query(
            text_search="python",
            type=BlockType.FACT,
            tags=["x"],
            min_confidence=0.5,
            session_id="s1",
            org_id="o1",
            project_id="p1",
            agent_id="a1",
            metadata_filters={"env": "test"},
            limit=3,
        )
        filters = _filters(storage)
        assert filters["type"] == BlockType.FACT
        assert filters["tags"] == ["x"]
        assert filters["min_confidence"] == 0.5
        assert filters["session_id"] == "s1"
        assert filters["org_id"] == "o1"
        assert filters["project_id"] == "p1"
        assert filters["agent_id"] == "a1"
        assert filters["metadata_filters"] == {"env": "test"}
        assert filters["sort_by"] == "relevance"
        assert filters["limit"] == 50

    def test_explicit_sorts_pick_the_matching_storage_sort(self):
        """An explicit recency / access_count sort is pushed down so the
        pool holds the newest / most-accessed matches (v0.13.1 semantics,
        bounded to the pool); relevance and strength take the best keyword
        matches. The pool limit is pushed for every sort except strength."""
        for engine_sort, storage_sort in _SORT_MAP:
            storage = _sync_storage(_abc())
            _sync_engine(storage).query(text_search="python", sort_by=engine_sort)
            assert _filters(storage)["sort_by"] == storage_sort, engine_sort
            if engine_sort == "strength":
                # No SQL equivalent: strength must see every match (v0.13.1).
                assert "limit" not in _filters(storage), engine_sort
            else:
                assert _filters(storage)["limit"] == 50, engine_sort

    def test_explicit_sort_on_non_text_query_pushes_nothing(self):
        storage = _sync_storage(_abc())
        _sync_engine(storage).query(type=BlockType.FACT, sort_by="recency")
        assert _filters(storage) == {"type": BlockType.FACT}


# ─── (b) FTS-only ordering ─────────────────────────────────────────────


class TestFtsOnlyOrdering:
    def test_keyword_rank_beats_recency_without_provider(self):
        storage = _sync_storage(_abc())
        results = _sync_engine(storage).query(text_search="python", limit=10)
        assert _ids(results) == ["A", "B", "C"]

    def test_keyword_rank_beats_recency_with_semantic_false(self):
        storage = _sync_storage(_abc())
        engine = _sync_engine(storage, provider=_provider())
        results = engine.query(text_search="python", semantic=False, limit=10)

        assert _ids(results) == ["A", "B", "C"]
        storage.search_similar_embeddings.assert_not_called()
        storage.get_all_embeddings.assert_not_called()

    def test_keyword_rank_used_when_provider_has_no_embeddings_yet(self):
        """Provider configured, semantic=True, but the store holds no
        embeddings: hybrid yields nothing, so the FTS order must still
        win over recency."""
        storage = _sync_storage(_abc())
        engine = _sync_engine(storage, provider=_provider())
        results = engine.query(text_search="python", limit=10)

        assert _ids(results) == ["A", "B", "C"]
        storage.get_all_embeddings.assert_called_once()

    def test_control_non_text_query_orders_by_recency(self):
        """Control: the same blocks through a non-text query score on
        confidence/strength/recency only, giving C, B, A. This proves
        the A, B, C ordering above comes from the keyword rank, not
        from the input order happening to survive the sort."""
        storage = _sync_storage(_abc())
        results = _sync_engine(storage).query(type=BlockType.FACT, limit=10)
        assert _ids(results) == ["C", "B", "A"]

    def test_explicit_recency_sort_still_honoured(self):
        storage = _sync_storage(_abc())
        results = _sync_engine(storage).query(
            text_search="python", sort_by="recency", limit=10,
        )
        assert _ids(results) == ["C", "B", "A"]

    def test_limit_still_applied_after_ranking(self):
        storage = _sync_storage(_abc())
        results = _sync_engine(storage).query(text_search="python", limit=2)
        assert _ids(results) == ["A", "B"]

    def test_min_strength_filter_still_applies(self):
        # confidence 0 → strength 0 → dropped by min_strength.
        weak = _block("W", 0, confidence=0.0)
        storage = _sync_storage([weak, *_abc()])
        results = _sync_engine(storage).query(
            text_search="python", min_strength=0.5, limit=10,
        )
        assert _ids(results) == ["A", "B", "C"]
        # FTS-only scoring must not trigger re-fetches of dropped blocks.
        storage.get_block.assert_not_called()

    def test_top_ranked_block_is_not_refetched(self):
        storage = _sync_storage(_abc())
        _sync_engine(storage).query(text_search="python", limit=10)
        storage.get_block.assert_not_called()

    def test_old_exact_match_beats_forty_fresh_one_token_matches(self):
        """The FTS-only curve must be steep enough that the best keyword
        match wins on rank alone: a 30-day-old rank-0 block (recency 0,
        strength ~0) against 40 fresh lower-ranked blocks of equal
        confidence. With the flat default RRF k=60 the rank-1 block won."""
        old = _block("OLD", minutes_ago=30 * 24 * 60)
        fresh = [_block(f"F{i}", minutes_ago=0) for i in range(40)]
        storage = _sync_storage([old, *fresh])
        ids = _ids(_sync_engine(storage).query(text_search="python", limit=10))
        assert ids[0] == "OLD"
        assert ids[1:] == [f"F{i}" for i in range(9)]  # keyword order kept


# ─── (b2) Hybrid fan-out bounded ───────────────────────────────────────


class TestHybridFanOutBounded:
    """The Python (non-pgvector) vector fallback scores EVERY embedded
    block and ``weighted_rrf_merge`` gives each one a cosine bonus, so
    ``hybrid_boost`` carries one entry per embedded block. The engine must
    hydrate at most the top ``pool`` of them; before this cap every hybrid
    query fetched every embedded block from storage (O(N) get_block)."""

    N = 400

    def test_only_top_pool_vec_only_blocks_are_hydrated(self):
        blocks, rows = _embedded(self.N)
        storage = _sync_storage(_abc())  # FTS pool: A, B, C
        storage.get_all_embeddings.return_value = rows
        storage.get_block.side_effect = lambda bid: blocks.get(bid)

        engine = _sync_engine(storage, provider=_provider())
        results = engine.query(text_search="python", limit=10)

        fetched = [c.args[0] for c in storage.get_block.call_args_list]
        assert len(fetched) == 50  # == pool, not N
        assert set(fetched) == {f"V{i}" for i in range(50)}
        assert _ids(results) == [f"V{i}" for i in range(10)]

    def test_semantic_only_query_hydrates_at_most_pool(self):
        """No keyword match at all: the vector pool alone feeds results,
        still bounded to ``pool`` fetches."""
        blocks, rows = _embedded(self.N)
        storage = _sync_storage([])
        storage.get_all_embeddings.return_value = rows
        storage.get_block.side_effect = lambda bid: blocks.get(bid)

        engine = _sync_engine(storage, provider=_provider())
        results = engine.query(text_search="python", limit=10)

        assert storage.get_block.call_count == 50
        assert _ids(results) == [f"V{i}" for i in range(10)]

    def test_pool_scales_the_cap(self):
        blocks, rows = _embedded(self.N)
        storage = _sync_storage([])
        storage.get_all_embeddings.return_value = rows
        storage.get_block.side_effect = lambda bid: blocks.get(bid)

        engine = _sync_engine(storage, provider=_provider())
        results = engine.query(text_search="python", limit=30)

        assert storage.get_block.call_count == 150
        assert _ids(results) == [f"V{i}" for i in range(30)]


# ─── (c) Hybrid path unchanged ─────────────────────────────────────────


class TestHybridPathUnchanged:
    def test_vec_only_block_added_and_overlap_favoured(self):
        a, b, v = _block("A", 3), _block("B", 2), _block("V", 1)
        storage = _sync_storage([a, b])  # FTS pool: A, B
        # pgvector-style server-side results: B (also in FTS) and V (vec only)
        storage.search_similar_embeddings.return_value = [("B", 0.9), ("V", 0.8)]
        storage.get_block.side_effect = lambda bid: {"V": v}.get(bid)

        engine = _sync_engine(storage, provider=_provider())
        results = engine.query(text_search="python", limit=10)
        ids = _ids(results)

        assert "V" in ids  # vec-only block fetched and added
        assert ids[0] == "B"  # present in both lists → RRF favours it
        assert set(ids) == {"A", "B", "V"}
        storage.get_block.assert_called_once_with("V")
        # Server-side path taken; brute force skipped.
        storage.get_all_embeddings.assert_not_called()
        # Vector search is asked for the same pool size storage was.
        assert storage.search_similar_embeddings.call_args.args[1] == 50

    def test_hybrid_still_ranks_over_fts_only_curve(self):
        """With vector signal present the FTS-only curve is not used:
        a block that is only vector-matched can outrank an FTS-only one."""
        a, b, v = _block("A", 3), _block("B", 2), _block("V", 1)
        storage = _sync_storage([a, b])
        storage.search_similar_embeddings.return_value = [("V", 0.99)]
        storage.get_block.side_effect = lambda bid: {"V": v}.get(bid)

        engine = _sync_engine(storage, provider=_provider())
        ids = _ids(engine.query(text_search="python", limit=10))
        assert ids.index("V") < ids.index("B")


# ─── (d) Async engine parity ───────────────────────────────────────────


class TestAsyncParity:
    def test_text_search_pushes_relevance_sort_and_pool_limit(self):
        async def _test():
            storage = _async_storage(_abc())
            await _async_engine(storage).query(text_search="python", limit=10)
            filters = _filters(storage)
            assert filters["text_search"] == "python"
            assert filters["sort_by"] == "relevance"
            assert filters["limit"] == 50

        asyncio.run(_test())

    def test_pool_scales_with_limit(self):
        async def _test():
            storage = _async_storage(_abc())
            await _async_engine(storage).query(text_search="python", limit=20)
            assert _filters(storage)["limit"] == 100

        asyncio.run(_test())

    def test_non_text_query_pushes_neither(self):
        async def _test():
            storage = _async_storage(_abc())
            await _async_engine(storage).query(type=BlockType.FACT, limit=10)
            assert _filters(storage) == {"type": BlockType.FACT}

        asyncio.run(_test())

    def test_keyword_rank_beats_recency_without_provider(self):
        async def _test():
            storage = _async_storage(_abc())
            results = await _async_engine(storage).query(
                text_search="python", limit=10,
            )
            assert _ids(results) == ["A", "B", "C"]
            storage.get_block.assert_not_called()

        asyncio.run(_test())

    def test_keyword_rank_beats_recency_with_semantic_false(self):
        async def _test():
            storage = _async_storage(_abc())
            engine = _async_engine(storage, provider=_provider())
            results = await engine.query(
                text_search="python", semantic=False, limit=10,
            )
            assert _ids(results) == ["A", "B", "C"]
            storage.search_similar_embeddings.assert_not_called()
            storage.get_all_embeddings.assert_not_called()

        asyncio.run(_test())

    def test_control_non_text_query_orders_by_recency(self):
        async def _test():
            storage = _async_storage(_abc())
            results = await _async_engine(storage).query(
                type=BlockType.FACT, limit=10,
            )
            assert _ids(results) == ["C", "B", "A"]

        asyncio.run(_test())

    def test_hybrid_vec_only_block_added_and_overlap_favoured(self):
        async def _test():
            a, b, v = _block("A", 3), _block("B", 2), _block("V", 1)
            storage = _async_storage([a, b])
            storage.search_similar_embeddings.return_value = [
                ("B", 0.9), ("V", 0.8),
            ]
            storage.get_block.side_effect = lambda bid: {"V": v}.get(bid)

            engine = _async_engine(storage, provider=_provider())
            ids = _ids(await engine.query(text_search="python", limit=10))
            assert ids[0] == "B"
            assert set(ids) == {"A", "B", "V"}
            storage.get_block.assert_called_once_with("V")
            storage.get_all_embeddings.assert_not_called()

        asyncio.run(_test())

    def test_sync_and_async_send_identical_filters(self):
        async def _test():
            kwargs = dict(
                text_search="python",
                type=BlockType.FACT,
                tags=["x"],
                min_confidence=0.5,
                session_id="s1",
                limit=7,
            )
            sync_storage = _sync_storage(_abc())
            _sync_engine(sync_storage).query(**kwargs)

            async_storage = _async_storage(_abc())
            await _async_engine(async_storage).query(**kwargs)

            assert _filters(sync_storage) == _filters(async_storage)

        asyncio.run(_test())

    def test_explicit_sorts_pick_the_matching_storage_sort(self):
        async def _test():
            for engine_sort, storage_sort in _SORT_MAP:
                storage = _async_storage(_abc())
                await _async_engine(storage).query(
                    text_search="python", sort_by=engine_sort,
                )
                assert _filters(storage)["sort_by"] == storage_sort, engine_sort
                if engine_sort == "strength":
                    assert "limit" not in _filters(storage), engine_sort
                else:
                    assert _filters(storage)["limit"] == 50, engine_sort

        asyncio.run(_test())

    def test_old_exact_match_beats_forty_fresh_one_token_matches(self):
        async def _test():
            old = _block("OLD", minutes_ago=30 * 24 * 60)
            fresh = [_block(f"F{i}", minutes_ago=0) for i in range(40)]
            storage = _async_storage([old, *fresh])
            ids = _ids(await _async_engine(storage).query(
                text_search="python", limit=10,
            ))
            assert ids[0] == "OLD"
            assert ids[1:] == [f"F{i}" for i in range(9)]

        asyncio.run(_test())

    def test_hybrid_fan_out_bounded_to_pool(self):
        async def _test():
            blocks, rows = _embedded(400)
            storage = _async_storage(_abc())
            storage.get_all_embeddings.return_value = rows
            storage.get_block.side_effect = lambda bid: blocks.get(bid)

            engine = _async_engine(storage, provider=_provider())
            results = await engine.query(text_search="python", limit=10)

            fetched = [c.args[0] for c in storage.get_block.call_args_list]
            assert len(fetched) == 50
            assert set(fetched) == {f"V{i}" for i in range(50)}
            assert _ids(results) == [f"V{i}" for i in range(10)]

        asyncio.run(_test())

    def test_sync_and_async_agree_on_hybrid_fan_out_results(self):
        async def _test():
            blocks, rows = _embedded(400)
            sync_storage = _sync_storage(_abc())
            sync_storage.get_all_embeddings.return_value = rows
            sync_storage.get_block.side_effect = lambda bid: blocks.get(bid)
            sync_ids = _ids(_sync_engine(sync_storage, provider=_provider()).query(
                text_search="python", limit=10,
            ))

            async_storage = _async_storage(_abc())
            async_storage.get_all_embeddings.return_value = rows
            async_storage.get_block.side_effect = lambda bid: blocks.get(bid)
            async_ids = _ids(await _async_engine(async_storage, provider=_provider()).query(
                text_search="python", limit=10,
            ))

            assert sync_ids == async_ids
            assert sync_storage.get_block.call_count == async_storage.get_block.call_count == 50

        asyncio.run(_test())
