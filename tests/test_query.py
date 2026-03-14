"""Tests for query engine."""

import pytest

from memblock.decay import DecayEngine
from memblock.graph import GraphIndex
from memblock.ops import OpLog
from memblock.query import QueryEngine
from memblock.storage.sqlite import SQLiteAdapter
from memblock.store import BlockStore
from memblock.types import BlockType, EdgeRelation, SourceType


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


class TestQueryEngine:
    def test_query_by_type(self, store: BlockStore, qe: QueryEngine):
        store.create(content="Fact one", type=BlockType.FACT)
        store.create(content="Fact two", type=BlockType.FACT)
        store.create(content="A preference", type=BlockType.PREFERENCE)

        results = qe.query(type=BlockType.FACT)
        assert len(results) == 2
        assert all(b.type == BlockType.FACT for b in results)

    def test_query_by_text(self, store: BlockStore, qe: QueryEngine):
        store.create(content="User likes Python programming")
        store.create(content="User deploys to AWS")
        store.create(content="Python is a language")

        results = qe.query(text_search="Python")
        assert len(results) == 2

    def test_query_by_min_confidence(self, store: BlockStore, qe: QueryEngine):
        store.create(content="High conf", confidence=0.95)
        store.create(content="Low conf", confidence=0.2)

        results = qe.query(min_confidence=0.5)
        assert len(results) == 1
        assert results[0].content == "High conf"

    def test_query_by_tags(self, store: BlockStore, qe: QueryEngine):
        store.create(content="Python stuff", tags=["python"])
        store.create(content="JS stuff", tags=["javascript"])

        results = qe.query(tags=["python"])
        assert len(results) == 1

    def test_query_related_to(self, store: BlockStore, graph: GraphIndex, qe: QueryEngine):
        entity = store.create(content="Python", type=BlockType.ENTITY)
        fact1 = store.create(content="Python is great", type=BlockType.FACT)
        fact2 = store.create(content="Java is verbose", type=BlockType.FACT)

        graph.link(fact1.id, entity.id, relation=EdgeRelation.ABOUT)
        # fact2 is NOT linked

        results = qe.query(related_to=entity.id)
        result_ids = {b.id for b in results}
        assert fact1.id in result_ids
        assert fact2.id not in result_ids

    def test_query_sort_by_recency(self, store: BlockStore, qe: QueryEngine):
        store.create(content="First")
        store.create(content="Second")
        store.create(content="Third")

        results = qe.query(sort_by="recency")
        assert results[0].content == "Third"  # most recent first

    def test_query_with_limit(self, store: BlockStore, qe: QueryEngine):
        for i in range(10):
            store.create(content=f"Block {i}")

        results = qe.query(limit=3)
        assert len(results) == 3

    def test_query_combined(self, store: BlockStore, qe: QueryEngine):
        store.create(content="Python preference", type=BlockType.PREFERENCE, confidence=0.9)
        store.create(content="Python fact", type=BlockType.FACT, confidence=0.9)
        store.create(content="JS preference", type=BlockType.PREFERENCE, confidence=0.9)
        store.create(content="Python low conf", type=BlockType.PREFERENCE, confidence=0.2)

        results = qe.query(
            type=BlockType.PREFERENCE,
            text_search="Python",
            min_confidence=0.5,
        )
        assert len(results) == 1
        assert results[0].content == "Python preference"
