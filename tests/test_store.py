"""Tests for BlockStore CRUD layer."""

import pytest

from memblock.block import Block
from memblock.schema import SchemaValidationError
from memblock.storage.sqlite import SQLiteAdapter
from memblock.store import BlockStore
from memblock.types import BlockType, SourceType


@pytest.fixture
def db():
    adapter = SQLiteAdapter(":memory:")
    adapter.initialize()
    yield adapter
    adapter.close()


@pytest.fixture
def store(db):
    return BlockStore(db, author="test_agent")


class TestBlockStoreCreate:
    def test_create_block(self, store: BlockStore):
        block = store.create(content="User likes Python", type=BlockType.PREFERENCE)
        assert block.id.startswith("blk_")
        assert block.content == "User likes Python"
        assert block.type == BlockType.PREFERENCE
        assert block.version == 1
        assert block.op_hash.startswith("sha256:")

    def test_create_with_all_options(self, store: BlockStore):
        block = store.create(
            content="Deployed v2.0",
            type=BlockType.EVENT,
            confidence=0.95,
            source=SourceType.OBSERVED,
            tags=["deployment", "v2"],
            decay_rate=0.05,
            ttl=86400,
        )
        assert block.metadata.confidence == 0.95
        assert block.metadata.source == SourceType.OBSERVED
        assert block.tags == ["deployment", "v2"]
        assert block.metadata.ttl == 86400

    def test_create_validates_content(self, store: BlockStore):
        with pytest.raises(SchemaValidationError, match="content must not be empty"):
            store.create(content="", type=BlockType.FACT)

    def test_create_validates_confidence(self, store: BlockStore):
        with pytest.raises(SchemaValidationError, match="Confidence"):
            store.create(content="Test", confidence=1.5)

    def test_create_with_parent(self, store: BlockStore):
        parent = store.create(content="Programming", type=BlockType.ENTITY)
        child = store.create(
            content="Python is great",
            type=BlockType.FACT,
            parent_id=parent.id,
        )
        assert child.parent_id == parent.id

        # Verify parent has child in children_ids
        refreshed_parent = store.get(parent.id)
        assert child.id in refreshed_parent.children_ids

    def test_create_with_invalid_parent(self, store: BlockStore):
        with pytest.raises(SchemaValidationError, match="not found"):
            store.create(content="Test", parent_id="blk_nonexistent")


class TestBlockStoreGet:
    def test_get_block(self, store: BlockStore):
        block = store.create(content="Test fact")
        retrieved = store.get(block.id)
        assert retrieved is not None
        assert retrieved.content == "Test fact"

    def test_get_increments_access_count(self, store: BlockStore):
        block = store.create(content="Test")
        store.get(block.id)
        store.get(block.id)
        final = store.get(block.id)
        assert final.metadata.access_count == 3

    def test_get_nonexistent(self, store: BlockStore):
        assert store.get("blk_fake") is None

    def test_get_deleted_returns_none(self, store: BlockStore):
        block = store.create(content="To delete")
        store.delete(block.id)
        assert store.get(block.id) is None

    def test_get_without_touch(self, store: BlockStore):
        block = store.create(content="Test")
        store.get_without_touch(block.id)
        store.get_without_touch(block.id)
        retrieved = store.get_without_touch(block.id)
        assert retrieved.metadata.access_count == 0


class TestBlockStoreUpdate:
    def test_update_content(self, store: BlockStore):
        block = store.create(content="Original")
        updated = store.update(block.id, content="Modified")
        assert updated.content == "Modified"
        assert updated.version == 2

    def test_update_confidence(self, store: BlockStore):
        block = store.create(content="Test", confidence=0.5)
        updated = store.update(block.id, confidence=0.9)
        assert updated.metadata.confidence == 0.9

    def test_update_tags(self, store: BlockStore):
        block = store.create(content="Test", tags=["old"])
        updated = store.update(block.id, tags=["new", "tags"])
        assert updated.tags == ["new", "tags"]

    def test_update_nonexistent(self, store: BlockStore):
        result = store.update("blk_fake", content="nope")
        assert result is None

    def test_update_validates(self, store: BlockStore):
        block = store.create(content="Test")
        with pytest.raises(SchemaValidationError):
            store.update(block.id, content="")


class TestBlockStoreDelete:
    def test_soft_delete(self, store: BlockStore):
        block = store.create(content="Delete me")
        assert store.delete(block.id) is True
        assert store.get(block.id) is None  # soft deleted

    def test_cascade_delete(self, store: BlockStore):
        parent = store.create(content="Parent", type=BlockType.ENTITY)
        child = store.create(content="Child", type=BlockType.ENTITY, parent_id=parent.id)

        store.delete(parent.id, cascade=True)
        assert store.get(parent.id) is None
        assert store.get(child.id) is None

    def test_delete_reparents_children(self, store: BlockStore):
        grandparent = store.create(content="GP", type=BlockType.ENTITY)
        parent = store.create(content="P", type=BlockType.ENTITY, parent_id=grandparent.id)
        child = store.create(content="C", type=BlockType.ENTITY, parent_id=parent.id)

        store.delete(parent.id, cascade=False)
        child_after = store.get_without_touch(child.id)
        assert child_after.parent_id == grandparent.id

    def test_hard_delete(self, store: BlockStore):
        block = store.create(content="Permanent delete")
        assert store.hard_delete(block.id) is True
        # Even get_without_touch returns None
        assert store.get_without_touch(block.id) is None


class TestBlockStoreTree:
    def test_get_children(self, store: BlockStore):
        parent = store.create(content="Parent", type=BlockType.ENTITY)
        c1 = store.create(content="Child 1", type=BlockType.ENTITY, parent_id=parent.id)
        c2 = store.create(content="Child 2", type=BlockType.ENTITY, parent_id=parent.id)

        children = store.get_children(parent.id)
        assert len(children) == 2

    def test_get_parent(self, store: BlockStore):
        parent = store.create(content="Parent", type=BlockType.ENTITY)
        child = store.create(content="Child", type=BlockType.ENTITY, parent_id=parent.id)

        found_parent = store.get_parent(child.id)
        assert found_parent is not None
        assert found_parent.id == parent.id

    def test_move_block(self, store: BlockStore):
        p1 = store.create(content="Parent 1", type=BlockType.ENTITY)
        p2 = store.create(content="Parent 2", type=BlockType.ENTITY)
        child = store.create(content="Child", type=BlockType.ENTITY, parent_id=p1.id)

        store.move(child.id, p2.id)

        assert store.get_parent(child.id).id == p2.id


class TestBlockStoreBulk:
    def test_get_by_type(self, store: BlockStore):
        store.create(content="Fact 1", type=BlockType.FACT)
        store.create(content="Fact 2", type=BlockType.FACT)
        store.create(content="Pref", type=BlockType.PREFERENCE)

        facts = store.get_by_type(BlockType.FACT)
        assert len(facts) == 2

    def test_get_by_tags(self, store: BlockStore):
        store.create(content="Python stuff", tags=["python"])
        store.create(content="JS stuff", tags=["javascript"])

        results = store.get_by_tags(["python"])
        assert len(results) == 1
        assert results[0].content == "Python stuff"
