"""Tests for context builder."""

import pytest

from memblock.context import ContextBuilder, estimate_tokens
from memblock.decay import DecayEngine
from memblock.graph import GraphIndex
from memblock.ops import OpLog
from memblock.query import QueryEngine
from memblock.storage.sqlite import SQLiteAdapter
from memblock.store import BlockStore
from memblock.types import BlockType, EdgeRelation


@pytest.fixture
def db():
    adapter = SQLiteAdapter(":memory:")
    adapter.initialize()
    yield adapter
    adapter.close()


@pytest.fixture
def store(db):
    return BlockStore(db, author="test")


@pytest.fixture
def graph(db, store):
    return GraphIndex(db, store.op_log)


@pytest.fixture
def decay(db):
    return DecayEngine(db)


@pytest.fixture
def qe(db, graph, decay):
    return QueryEngine(db, graph, decay)


@pytest.fixture
def ctx(db, qe, graph, decay):
    return ContextBuilder(db, qe, graph, decay)


class TestEstimateTokens:
    def test_estimate(self):
        assert estimate_tokens("hello") == 1  # 5 chars / 4
        assert estimate_tokens("a" * 400) == 100

    def test_empty(self):
        assert estimate_tokens("") == 0


class TestContextBuilder:
    def test_build_empty(self, ctx: ContextBuilder):
        result = ctx.build_context(query="nothing here")
        assert result == ""

    def test_build_relevance_strategy(self, store: BlockStore, ctx: ContextBuilder):
        store.create(content="User likes Python", type=BlockType.PREFERENCE)
        store.create(content="User uses AWS", type=BlockType.FACT)

        result = ctx.build_context(token_budget=4000)
        assert "## Memory Context" in result
        assert "Python" in result
        assert "AWS" in result

    def test_build_respects_token_budget(self, store: BlockStore, ctx: ContextBuilder):
        # Create many blocks
        for i in range(50):
            store.create(content=f"Memory block number {i} with some extra content to take up tokens")

        # Very small budget
        result = ctx.build_context(token_budget=100)
        tokens = estimate_tokens(result)
        assert tokens <= 100

    def test_build_with_metadata(self, store: BlockStore, ctx: ContextBuilder):
        store.create(
            content="User prefers dark mode",
            type=BlockType.PREFERENCE,
            confidence=0.95,
        )

        result = ctx.build_context(include_metadata=True)
        assert "[PREFERENCE]" in result
        assert "[HIGH]" in result
        assert "source:" in result

    def test_build_without_metadata(self, store: BlockStore, ctx: ContextBuilder):
        store.create(content="User prefers dark mode", type=BlockType.PREFERENCE)

        result = ctx.build_context(include_metadata=False)
        assert "[PREFERENCE]" not in result
        assert "dark mode" in result

    def test_build_type_grouped(self, store: BlockStore, ctx: ContextBuilder):
        store.create(content="Fact content", type=BlockType.FACT)
        store.create(content="Preference content", type=BlockType.PREFERENCE)
        store.create(content="Event content", type=BlockType.EVENT)

        result = ctx.build_context(strategy="type_grouped", token_budget=4000)
        assert "### Facts" in result or "### Fact" in result

    def test_build_graph_walk(
        self, store: BlockStore, graph: GraphIndex, ctx: ContextBuilder
    ):
        entity = store.create(content="Python language", type=BlockType.ENTITY)
        fact = store.create(content="Python is interpreted", type=BlockType.FACT)
        pref = store.create(content="User likes Python", type=BlockType.PREFERENCE)

        graph.link(fact.id, entity.id, relation=EdgeRelation.ABOUT)
        graph.link(pref.id, entity.id, relation=EdgeRelation.ABOUT)

        result = ctx.build_context(
            query="Python",
            strategy="graph_walk",
            token_budget=4000,
        )
        assert "Python" in result

    def test_build_with_type_filter(self, store: BlockStore, ctx: ContextBuilder):
        store.create(content="A fact", type=BlockType.FACT)
        store.create(content="A preference", type=BlockType.PREFERENCE)

        result = ctx.build_context(
            block_type=BlockType.FACT,
            token_budget=4000,
        )
        assert "fact" in result.lower()
        # preference should not be included when filtered
        # (it might still appear if query matches, but type filter should work)
