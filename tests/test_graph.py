"""Tests for graph index — edges, traversal, and contradictions."""

import pytest

from memblock.graph import GraphIndex
from memblock.ops import OpLog
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


class TestGraphLink:
    def test_link_blocks(self, store: BlockStore, graph: GraphIndex):
        b1 = store.create(content="Python", type=BlockType.ENTITY)
        b2 = store.create(content="User likes Python", type=BlockType.PREFERENCE)

        edge = graph.link(b2.id, b1.id, relation=EdgeRelation.ABOUT)
        assert edge.source_id == b2.id
        assert edge.target_id == b1.id
        assert edge.relation == EdgeRelation.ABOUT

    def test_link_nonexistent_source(self, store: BlockStore, graph: GraphIndex):
        b1 = store.create(content="Target", type=BlockType.ENTITY)
        with pytest.raises(ValueError, match="not found"):
            graph.link("blk_fake", b1.id)

    def test_link_validates_edge_relation(self, store: BlockStore, graph: GraphIndex):
        b1 = store.create(content="Fact 1", type=BlockType.FACT)
        b2 = store.create(content="Fact 2", type=BlockType.FACT)
        # ABOUT requires target to be ENTITY
        with pytest.raises(Exception):
            graph.link(b1.id, b2.id, relation=EdgeRelation.ABOUT)

    def test_unlink(self, store: BlockStore, graph: GraphIndex):
        b1 = store.create(content="A", type=BlockType.FACT)
        b2 = store.create(content="B", type=BlockType.FACT)

        graph.link(b1.id, b2.id, relation=EdgeRelation.SUPPORTS)
        removed = graph.unlink(b1.id, b2.id)
        assert removed == 1

    def test_unlink_specific_relation(self, store: BlockStore, graph: GraphIndex):
        b1 = store.create(content="A", type=BlockType.FACT)
        b2 = store.create(content="B", type=BlockType.FACT)

        graph.link(b1.id, b2.id, relation=EdgeRelation.SUPPORTS)
        graph.link(b1.id, b2.id, relation=EdgeRelation.RELATED_TO)

        removed = graph.unlink(b1.id, b2.id, relation=EdgeRelation.SUPPORTS)
        assert removed == 1

        # RELATED_TO edge should still exist
        edges = graph.get_edges_between(b1.id, b2.id)
        assert len(edges) == 1
        assert edges[0].relation == EdgeRelation.RELATED_TO


class TestGraphNeighbors:
    def test_neighbors(self, store: BlockStore, graph: GraphIndex):
        center = store.create(content="Center", type=BlockType.ENTITY)
        n1 = store.create(content="N1", type=BlockType.FACT)
        n2 = store.create(content="N2", type=BlockType.FACT)
        n3 = store.create(content="Unconnected", type=BlockType.FACT)

        graph.link(n1.id, center.id, relation=EdgeRelation.ABOUT)
        graph.link(n2.id, center.id, relation=EdgeRelation.ABOUT)

        neighbors = graph.neighbors(center.id)
        assert len(neighbors) == 2
        neighbor_ids = {b.id for b in neighbors}
        assert n1.id in neighbor_ids
        assert n2.id in neighbor_ids
        assert n3.id not in neighbor_ids

    def test_neighbors_with_relation_filter(self, store: BlockStore, graph: GraphIndex):
        b1 = store.create(content="A", type=BlockType.FACT)
        b2 = store.create(content="B", type=BlockType.FACT)
        b3 = store.create(content="C", type=BlockType.FACT)

        graph.link(b1.id, b2.id, relation=EdgeRelation.SUPPORTS)
        graph.link(b1.id, b3.id, relation=EdgeRelation.CONTRADICTS)

        supports = graph.neighbors(b1.id, relation=EdgeRelation.SUPPORTS)
        assert len(supports) == 1
        assert supports[0].id == b2.id


class TestGraphTraversal:
    def test_traverse(self, store: BlockStore, graph: GraphIndex):
        # Build: A -> B -> C -> D
        a = store.create(content="A", type=BlockType.FACT)
        b = store.create(content="B", type=BlockType.FACT)
        c = store.create(content="C", type=BlockType.FACT)
        d = store.create(content="D", type=BlockType.FACT)

        graph.link(a.id, b.id, relation=EdgeRelation.RELATED_TO)
        graph.link(b.id, c.id, relation=EdgeRelation.RELATED_TO)
        graph.link(c.id, d.id, relation=EdgeRelation.RELATED_TO)

        # From A, depth 2 should reach B and C but not D
        results = graph.traverse(a.id, max_depth=2)
        result_ids = {r.id for r in results}
        assert b.id in result_ids
        assert c.id in result_ids
        assert d.id not in result_ids

    def test_traverse_full_depth(self, store: BlockStore, graph: GraphIndex):
        a = store.create(content="A", type=BlockType.FACT)
        b = store.create(content="B", type=BlockType.FACT)
        c = store.create(content="C", type=BlockType.FACT)

        graph.link(a.id, b.id, relation=EdgeRelation.RELATED_TO)
        graph.link(b.id, c.id, relation=EdgeRelation.RELATED_TO)

        results = graph.traverse(a.id, max_depth=10)
        assert len(results) == 2

    def test_traverse_excludes_start(self, store: BlockStore, graph: GraphIndex):
        a = store.create(content="A", type=BlockType.FACT)
        b = store.create(content="B", type=BlockType.FACT)

        graph.link(a.id, b.id, relation=EdgeRelation.RELATED_TO)

        results = graph.traverse(a.id)
        result_ids = {r.id for r in results}
        assert a.id not in result_ids


class TestGraphShortestPath:
    def test_shortest_path(self, store: BlockStore, graph: GraphIndex):
        a = store.create(content="A", type=BlockType.FACT)
        b = store.create(content="B", type=BlockType.FACT)
        c = store.create(content="C", type=BlockType.FACT)

        graph.link(a.id, b.id, relation=EdgeRelation.RELATED_TO)
        graph.link(b.id, c.id, relation=EdgeRelation.RELATED_TO)

        path = graph.shortest_path(a.id, c.id)
        assert len(path) == 3
        assert path[0].id == a.id
        assert path[-1].id == c.id

    def test_no_path(self, store: BlockStore, graph: GraphIndex):
        a = store.create(content="A", type=BlockType.FACT)
        b = store.create(content="B", type=BlockType.FACT)
        # No edges

        path = graph.shortest_path(a.id, b.id)
        assert path == []

    def test_same_node_path(self, store: BlockStore, graph: GraphIndex):
        a = store.create(content="A", type=BlockType.FACT)
        path = graph.shortest_path(a.id, a.id)
        assert len(path) == 1


class TestGraphContradictions:
    def test_find_contradictions(self, store: BlockStore, graph: GraphIndex):
        b1 = store.create(content="User likes React", type=BlockType.PREFERENCE)
        b2 = store.create(content="User switched to Vue", type=BlockType.PREFERENCE)

        graph.link(b1.id, b2.id, relation=EdgeRelation.CONTRADICTS)

        contradictions = graph.find_contradictions(b1.id)
        assert len(contradictions) == 1
        assert contradictions[0].id == b2.id

    def test_get_cluster(self, store: BlockStore, graph: GraphIndex):
        a = store.create(content="A", type=BlockType.FACT)
        b = store.create(content="B", type=BlockType.FACT)
        c = store.create(content="C", type=BlockType.FACT)
        isolated = store.create(content="Isolated", type=BlockType.FACT)

        graph.link(a.id, b.id, relation=EdgeRelation.RELATED_TO)
        graph.link(b.id, c.id, relation=EdgeRelation.RELATED_TO)

        cluster = graph.get_cluster(a.id)
        cluster_ids = {bl.id for bl in cluster}
        assert b.id in cluster_ids
        assert c.id in cluster_ids
        assert isolated.id not in cluster_ids
