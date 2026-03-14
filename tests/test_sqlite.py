"""Tests for SQLite storage adapter."""

import pytest

from memblock.block import Block
from memblock.storage.sqlite import SQLiteAdapter
from memblock.types import (
    BlockMetadata,
    BlockType,
    Edge,
    EdgeRelation,
    EncryptionLevel,
    Operation,
    OpAction,
    SourceType,
)


@pytest.fixture
def db():
    adapter = SQLiteAdapter(":memory:")
    adapter.initialize()
    yield adapter
    adapter.close()


# ─── Block CRUD ───────────────────────────────────────────────────────────────


class TestBlockCRUD:
    def test_save_and_get_block(self, db: SQLiteAdapter):
        block = Block(
            content="User likes Python",
            type=BlockType.PREFERENCE,
            tags=["programming", "language"],
            metadata=BlockMetadata(confidence=0.9, source=SourceType.EXPLICIT),
        )
        db.save_block(block)

        retrieved = db.get_block(block.id)
        assert retrieved is not None
        assert retrieved.id == block.id
        assert retrieved.content == "User likes Python"
        assert retrieved.type == BlockType.PREFERENCE
        assert retrieved.metadata.confidence == 0.9
        assert retrieved.tags == ["programming", "language"]

    def test_get_nonexistent_block(self, db: SQLiteAdapter):
        assert db.get_block("blk_nonexistent") is None

    def test_delete_block(self, db: SQLiteAdapter):
        block = Block(content="To be deleted", type=BlockType.FACT)
        db.save_block(block)
        assert db.get_block(block.id) is not None

        db.delete_block(block.id)
        assert db.get_block(block.id) is None

    def test_update_block(self, db: SQLiteAdapter):
        block = Block(content="Original", type=BlockType.FACT)
        db.save_block(block)

        db.update_block(block.id, {"content": "Updated", "confidence": 0.5})

        updated = db.get_block(block.id)
        assert updated.content == "Updated"
        assert updated.metadata.confidence == 0.5

    def test_save_block_with_encryption(self, db: SQLiteAdapter):
        block = Block(
            content="Sensitive data",
            type=BlockType.FACT,
            encryption_level=EncryptionLevel.SENSITIVE,
            encrypted=True,
        )
        db.save_block(block)

        retrieved = db.get_block(block.id)
        assert retrieved.encryption_level == EncryptionLevel.SENSITIVE
        assert retrieved.encrypted is True

    def test_get_all_blocks(self, db: SQLiteAdapter):
        for i in range(5):
            db.save_block(Block(content=f"Block {i}", type=BlockType.FACT))

        blocks = db.get_all_blocks()
        assert len(blocks) == 5

    def test_get_all_blocks_excludes_deleted(self, db: SQLiteAdapter):
        b1 = Block(content="Active", type=BlockType.FACT)
        b2 = Block(content="Deleted", type=BlockType.FACT, deleted=True)
        db.save_block(b1)
        db.save_block(b2)

        assert len(db.get_all_blocks(include_deleted=False)) == 1
        assert len(db.get_all_blocks(include_deleted=True)) == 2


# ─── Query ────────────────────────────────────────────────────────────────────


class TestBlockQuery:
    def test_query_by_type(self, db: SQLiteAdapter):
        db.save_block(Block(content="Fact 1", type=BlockType.FACT))
        db.save_block(Block(content="Fact 2", type=BlockType.FACT))
        db.save_block(Block(content="Pref 1", type=BlockType.PREFERENCE))

        results = db.query_blocks({"type": BlockType.FACT})
        assert len(results) == 2
        assert all(b.type == BlockType.FACT for b in results)

    def test_query_by_min_confidence(self, db: SQLiteAdapter):
        db.save_block(Block(content="High", metadata=BlockMetadata(confidence=0.9)))
        db.save_block(Block(content="Low", metadata=BlockMetadata(confidence=0.3)))

        results = db.query_blocks({"min_confidence": 0.7})
        assert len(results) == 1
        assert results[0].content == "High"

    def test_query_by_tags(self, db: SQLiteAdapter):
        db.save_block(Block(content="Python stuff", tags=["python", "code"]))
        db.save_block(Block(content="JS stuff", tags=["javascript", "code"]))
        db.save_block(Block(content="Food", tags=["cooking"]))

        results = db.query_blocks({"tags": ["python"]})
        assert len(results) == 1
        assert results[0].content == "Python stuff"

    def test_query_text_search(self, db: SQLiteAdapter):
        db.save_block(Block(content="User prefers TypeScript for frontend"))
        db.save_block(Block(content="User deploys to AWS"))
        db.save_block(Block(content="TypeScript is a superset of JavaScript"))

        results = db.query_blocks({"text_search": "TypeScript"})
        assert len(results) == 2

    def test_query_with_limit(self, db: SQLiteAdapter):
        for i in range(10):
            db.save_block(Block(content=f"Block {i}"))

        results = db.query_blocks({"limit": 3})
        assert len(results) == 3

    def test_query_combined_filters(self, db: SQLiteAdapter):
        db.save_block(Block(
            content="Python is great",
            type=BlockType.PREFERENCE,
            metadata=BlockMetadata(confidence=0.9),
        ))
        db.save_block(Block(
            content="Python is okay",
            type=BlockType.PREFERENCE,
            metadata=BlockMetadata(confidence=0.4),
        ))
        db.save_block(Block(
            content="Python event",
            type=BlockType.EVENT,
            metadata=BlockMetadata(confidence=0.9),
        ))

        results = db.query_blocks({
            "type": BlockType.PREFERENCE,
            "min_confidence": 0.7,
        })
        assert len(results) == 1
        assert results[0].content == "Python is great"


# ─── Edge Operations ─────────────────────────────────────────────────────────


class TestEdgeOperations:
    def test_save_and_get_edges(self, db: SQLiteAdapter):
        b1 = Block(content="Block A", type=BlockType.FACT)
        b2 = Block(content="Block B", type=BlockType.FACT)
        db.save_block(b1)
        db.save_block(b2)

        edge = Edge(
            source_id=b1.id,
            target_id=b2.id,
            relation=EdgeRelation.SUPPORTS,
            weight=0.9,
        )
        db.save_edge(edge)

        edges = db.get_edges(b1.id, direction="outgoing")
        assert len(edges) == 1
        assert edges[0].relation == EdgeRelation.SUPPORTS
        assert edges[0].weight == 0.9

    def test_get_edges_incoming(self, db: SQLiteAdapter):
        b1 = Block(content="A")
        b2 = Block(content="B")
        db.save_block(b1)
        db.save_block(b2)

        edge = Edge(source_id=b1.id, target_id=b2.id, relation=EdgeRelation.ABOUT)
        db.save_edge(edge)

        incoming = db.get_edges(b2.id, direction="incoming")
        assert len(incoming) == 1

        outgoing = db.get_edges(b2.id, direction="outgoing")
        assert len(outgoing) == 0

    def test_delete_edge(self, db: SQLiteAdapter):
        b1 = Block(content="A")
        b2 = Block(content="B")
        db.save_block(b1)
        db.save_block(b2)

        edge = Edge(source_id=b1.id, target_id=b2.id, relation=EdgeRelation.RELATED_TO)
        db.save_edge(edge)

        db.delete_edge(edge.id)
        assert len(db.get_edges(b1.id)) == 0

    def test_delete_edges_for_block(self, db: SQLiteAdapter):
        b1 = Block(content="A")
        b2 = Block(content="B")
        b3 = Block(content="C")
        db.save_block(b1)
        db.save_block(b2)
        db.save_block(b3)

        db.save_edge(Edge(source_id=b1.id, target_id=b2.id, relation=EdgeRelation.SUPPORTS))
        db.save_edge(Edge(source_id=b3.id, target_id=b1.id, relation=EdgeRelation.CONTRADICTS))

        db.delete_edges_for_block(b1.id)
        assert len(db.get_edges(b1.id)) == 0
        assert len(db.get_edges(b2.id)) == 0  # edge was from b1->b2, deleted


# ─── Operation Log ────────────────────────────────────────────────────────────


class TestOperationLog:
    def test_save_and_get_operations(self, db: SQLiteAdapter):
        op = Operation(
            block_id="blk_123",
            action=OpAction.CREATE,
            data={"content": "test"},
            hash="abc123",
            clock=1,
        )
        db.save_operation(op)

        ops = db.get_operations(block_id="blk_123")
        assert len(ops) == 1
        assert ops[0].action == OpAction.CREATE
        assert ops[0].data == {"content": "test"}

    def test_get_last_operation(self, db: SQLiteAdapter):
        for i in range(5):
            op = Operation(block_id="blk_x", action=OpAction.UPDATE, clock=i, hash=f"h{i}")
            db.save_operation(op)

        last = db.get_last_operation()
        assert last is not None
        assert last.clock == 4

    def test_get_operations_ordered_by_clock(self, db: SQLiteAdapter):
        for i in [3, 1, 4, 1, 5]:
            op = Operation(block_id="blk_x", action=OpAction.UPDATE, clock=i)
            db.save_operation(op)

        ops = db.get_operations()
        clocks = [o.clock for o in ops]
        assert clocks == sorted(clocks)

    def test_get_last_operation_empty(self, db: SQLiteAdapter):
        assert db.get_last_operation() is None
