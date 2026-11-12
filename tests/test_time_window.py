"""A time phrase in a query ("last week", "recent") ranks results; it does not
filter them (`time_window="rank"`, the default).

The phrase becomes a `happened_at` window. Blocks dated inside it are ranked
higher, and a strong match dated outside it is still returned. Before this,
the window was a filter whenever anything at all fell inside it, which hid
answers that were a little older than the phrase suggested. `time_window=
"filter"` keeps that behaviour.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from unittest.mock import MagicMock

import pytest

from memblock import AsyncMemBlock, BlockType, MemBlock, MemorySession
from memblock.async_query import AsyncQueryEngine
from memblock.block import Block
from memblock.decay import DecayEngine
from memblock.embeddings import CallableEmbeddingProvider, pack_embedding
from memblock.query import QueryEngine, infer_time_window, merge_time_window
from memblock.storage.async_base import AsyncStorageAdapter
from memblock.storage.base import StorageAdapter
from memblock.types import BlockMetadata, now_utc


def days_ago(days: int):
    return now_utc() - timedelta(days=days)


def ids(blocks) -> list[str]:
    return [b.id for b in blocks]


@pytest.fixture
def mem():
    m = MemBlock(storage="sqlite:///:memory:")
    yield m
    m.close()


# ─── Through MemBlock, on SQLite ─────────────────────────────────────────────


class TestRankInsteadOfFilter:
    QUESTION = "Where did Rachel move to after her recent relocation?"  # "recent" = the last 7 days

    def _rachel(self, mem):
        answer = mem.store("Rachel moved to Chicago for her new job", type=BlockType.FACT, happened_at=days_ago(60))
        other = mem.store(
            "Lunch with Rachel to plan the office relocation budget", type=BlockType.FACT, happened_at=days_ago(2),
        )
        return answer, other

    def test_answer_dated_outside_the_window_is_still_returned(self, mem):
        answer, other = self._rachel(mem)
        found = ids(mem.query(text_search=self.QUESTION, limit=5))
        assert answer.id in found and other.id in found

    def test_filter_mode_keeps_the_earlier_behaviour(self, mem):
        answer, other = self._rachel(mem)
        found = ids(mem.query(text_search=self.QUESTION, limit=5, time_window="filter"))
        assert found == [other.id]  # the answer is two months old: hidden

    def test_block_inside_the_window_is_lifted_above_a_better_keyword_match(self, mem):
        stale = mem.store(
            "I bought a kitchen appliance, a toaster, as a kitchen gift", type=BlockType.FACT,
            happened_at=days_ago(400),
        )
        recent = mem.store("I bought a kettle", type=BlockType.FACT, happened_at=days_ago(2))
        # Without a time phrase the better keyword match comes first.
        assert ids(mem.query(text_search="what kitchen appliance did I buy", limit=5)) == [stale.id, recent.id]
        # With one, the block from that period comes first and the other stays.
        assert ids(mem.query(text_search="what kitchen appliance did I buy last week", limit=5)) == \
            [recent.id, stale.id]

    def test_empty_window_changes_nothing(self, mem):
        mem.store("I bought a new air fryer for the kitchen", type=BlockType.FACT, happened_at=days_ago(900))
        mem.store("Kitchen renovation quotes were expensive", type=BlockType.FACT, happened_at=days_ago(800))
        mem.store("The kitchen tap is leaking again", type=BlockType.FACT)  # no date at all
        question = "what kitchen appliance did I buy last week"
        ranked = ids(mem.query(text_search=question, limit=5))
        assert len(ranked) == 3
        assert ranked == ids(mem.query(text_search=question, limit=5, time_window="filter"))
        assert ranked == ids(mem.query(text_search="what kitchen appliance did I buy", limit=5))

    def test_query_without_a_time_phrase_is_unaffected(self, mem):
        self._rachel(mem)
        question = "Where did Rachel move to"
        assert ids(mem.query(text_search=question)) == ids(mem.query(text_search=question, time_window="filter"))

    def test_an_explicit_sort_order_still_filters(self, mem):
        answer, other = self._rachel(mem)
        for order in ("recency", "access_count", "strength"):
            assert ids(mem.query(text_search=self.QUESTION, sort_by=order, limit=5)) == [other.id], order

    def test_explicit_filters_are_kept(self, mem):
        mem.store("Rachel moved to Chicago", type=BlockType.FACT, happened_at=days_ago(60), session_id="a")
        b = mem.store("Rachel moved to Denver", type=BlockType.FACT, happened_at=days_ago(60), session_id="b")
        mem.store("Rachel relocation lunch", type=BlockType.EVENT, happened_at=days_ago(2), session_id="b")
        found = mem.query(text_search=self.QUESTION, session_id="b", type=BlockType.FACT, limit=5)
        assert ids(found) == [b.id]

    def test_unknown_mode_is_refused(self, mem):
        with pytest.raises(ValueError):
            mem.query(text_search="anything", time_window="soft")

    def test_session_recall_shows_the_older_answer(self, mem):
        mem.store_episode("Rachel moved to Chicago for her new job", role="user", happened_at=days_ago(60))
        mem.store_episode("Lunch with Rachel to plan the office relocation budget", role="user", happened_at=days_ago(2))
        text, _ = MemorySession(mem, lambda messages: "ok").recall(self.QUESTION)
        assert "Chicago" in text and "relocation budget" in text

    def test_async_memblock_takes_the_same_option(self):
        async def run():
            mem = AsyncMemBlock(storage="sqlite:///:memory:")
            try:
                answer = await mem.store(
                    "Rachel moved to Chicago for her new job", type=BlockType.FACT, happened_at=days_ago(60),
                )
                other = await mem.store(
                    "Lunch with Rachel to plan the office relocation budget", type=BlockType.FACT,
                    happened_at=days_ago(2),
                )
                ranked = ids(await mem.query(text_search=self.QUESTION, limit=5))
                filtered = ids(await mem.query(text_search=self.QUESTION, limit=5, time_window="filter"))
                return answer.id in ranked and other.id in ranked, filtered == [other.id]
            finally:
                await mem.close()

        assert asyncio.run(run()) == (True, True)


# ─── The merge itself ────────────────────────────────────────────────────────


def _block(block_id: str, minutes_ago: int = 1) -> Block:
    return Block(
        id=block_id, content="python",
        metadata=BlockMetadata(confidence=0.8, created_at=now_utc() - timedelta(minutes=minutes_ago)),
    )


class TestMerge:
    def test_lists_alternate_window_first(self):
        a, b, c, d, e = ((_block(x), 1.0) for x in "ABCDE")
        merged = merge_time_window(inside=[c, d], overall=[a, b, c, e])
        assert [item[0].id for item in merged] == ["C", "A", "D", "B", "E"]

    def test_best_match_outside_the_window_is_never_below_second(self):
        # 30 blocks in the window, all of them also in the overall list just
        # below one block from outside it. That block must stay near the top:
        # the r-th block overall is never placed later than position 2r.
        window = [(_block(f"W{i}"), 1.0) for i in range(30)]
        outside = [(_block(f"O{i}"), 1.0) for i in range(3)]
        overall = [outside[0], *window[:10], outside[1], *window[10:], outside[2]]
        merged = [item[0].id for item in merge_time_window(window, overall)]
        assert merged[:2] == ["W0", "O0"]
        assert len(merged) == 33 and len(set(merged)) == 33
        for rank, item in enumerate(overall, start=1):
            assert merged.index(item[0].id) + 1 <= 2 * rank

    def test_empty_sides(self):
        a, b = (_block("A"), 1.0), (_block("B"), 1.0)
        assert merge_time_window([], [a, b]) == [a, b]
        assert merge_time_window([a, b], []) == [a, b]
        assert merge_time_window([], []) == []

    def test_time_phrase_is_read_and_removed(self):
        window, cleaned = infer_time_window("what did I buy last week")
        assert set(window) == {"happened_after", "happened_before"} and cleaned == "what did I buy"
        assert infer_time_window("what did I buy") == ({}, "what did I buy")
        assert infer_time_window(None) == ({}, None)


# ─── Both engines, on a mocked store ─────────────────────────────────────────


INSIDE = ["W1", "W2"]            # blocks dated inside the window, keyword order
OVERALL = ["O1", "W1", "O2", "W2"]  # every block, keyword order
QUESTION = "what did I cook last week"


def _by_window(filters: dict) -> list[Block]:
    names = INSIDE if "happened_after" in filters else OVERALL
    return [_block(name, minutes_ago=i + 1) for i, name in enumerate(names)]


def _sync_engine(provider=None, rows=None):
    storage = MagicMock(spec=StorageAdapter)
    storage.query_blocks.side_effect = _by_window
    storage.search_similar_embeddings.return_value = []
    storage.get_all_embeddings.return_value = rows or []
    storage.get_block.return_value = None
    engine = QueryEngine(
        storage, graph=MagicMock(), decay=DecayEngine(storage=None),  # type: ignore[arg-type]
        embedding_provider=provider,
    )
    return engine, storage


def _async_engine(provider=None, rows=None):
    storage = MagicMock(spec=AsyncStorageAdapter)  # coroutine methods become AsyncMocks
    storage.query_blocks.side_effect = _by_window
    storage.search_similar_embeddings.return_value = []
    storage.get_all_embeddings.return_value = rows or []
    storage.get_block.return_value = None
    engine = AsyncQueryEngine(
        storage, decay=DecayEngine(storage=None),  # type: ignore[arg-type]
        embedding_provider=provider,
    )
    return engine, storage


def _run(engine, **kwargs) -> list[str]:
    result = engine.query(**kwargs)
    if asyncio.iscoroutine(result):
        result = asyncio.run(result)
    return ids(result)


def _windows_asked(storage) -> list[bool]:
    return ["happened_after" in call.args[0] for call in storage.query_blocks.call_args_list]


@pytest.mark.parametrize("make", [_sync_engine, _async_engine], ids=["sync", "async"])
class TestBothEngines:
    def test_rank_merges_the_two_lists(self, make):
        engine, storage = make()
        assert _run(engine, text_search=QUESTION, limit=10) == ["W1", "O1", "W2", "O2"]
        assert _windows_asked(storage) == [True, False]  # one search inside the window, one overall

    def test_filter_searches_inside_the_window_only(self, make):
        engine, storage = make()
        assert _run(engine, text_search=QUESTION, limit=10, time_window="filter") == ["W1", "W2"]
        assert _windows_asked(storage) == [True]

    def test_no_time_phrase_means_one_search(self, make):
        engine, storage = make()
        assert _run(engine, text_search="what did I cook", limit=10) == OVERALL
        assert _windows_asked(storage) == [False]

    def test_explicit_sort_order_filters(self, make):
        engine, storage = make()
        assert set(_run(engine, text_search=QUESTION, sort_by="recency", limit=10)) == set(INSIDE)
        assert _windows_asked(storage) == [True]

    def test_limit_is_applied_after_the_merge(self, make):
        engine, _ = make()
        assert _run(engine, text_search=QUESTION, limit=3) == ["W1", "O1", "W2"]

    def test_unknown_mode_is_refused(self, make):
        engine, _ = make()
        with pytest.raises(ValueError):
            _run(engine, text_search=QUESTION, time_window="soft")

    def test_query_is_embedded_and_vectors_searched_once(self, make):
        calls = []

        def embed(texts):
            calls.append(list(texts))
            return [[1.0, 0.0, 0.0] for _ in texts]

        rows = [(name, pack_embedding([1.0, 0.01 * i, 0.0])) for i, name in enumerate(OVERALL)]
        engine, storage = make(CallableEmbeddingProvider(embed, dims=3), rows)
        found = _run(engine, text_search=QUESTION, limit=10)
        assert set(found) == set(OVERALL) and found[0] == "W1"
        assert calls == [[QUESTION]]
        assert storage.get_all_embeddings.call_count == 1
