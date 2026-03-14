"""Tests for Block model and types."""

import pytest

from memblock.block import Block
from memblock.types import (
    BlockMetadata,
    BlockType,
    Edge,
    EdgeRelation,
    EncryptionLevel,
    SourceType,
    generate_block_id,
)
from memblock.schema import BlockSchema, SchemaValidationError


# ─── Block ID Generation ─────────────────────────────────────────────────────


class TestBlockId:
    def test_generates_unique_ids(self):
        ids = {generate_block_id() for _ in range(100)}
        assert len(ids) == 100

    def test_has_prefix(self):
        bid = generate_block_id()
        assert bid.startswith("blk_")

    def test_reasonable_length(self):
        bid = generate_block_id()
        assert 10 < len(bid) < 30


# ─── Block Model ─────────────────────────────────────────────────────────────


class TestBlock:
    def test_create_default_block(self):
        block = Block(content="User likes Python")
        assert block.content == "User likes Python"
        assert block.type == BlockType.FACT
        assert block.version == 1
        assert block.deleted is False
        assert block.id.startswith("blk_")

    def test_create_preference_block(self):
        block = Block(
            content="Prefers dark mode",
            type=BlockType.PREFERENCE,
            tags=["ui", "theme"],
            metadata=BlockMetadata(confidence=0.85, source=SourceType.EXPLICIT),
        )
        assert block.type == BlockType.PREFERENCE
        assert block.metadata.confidence == 0.85
        assert block.tags == ["ui", "theme"]

    def test_to_dict_roundtrip(self):
        block = Block(
            content="Test content",
            type=BlockType.EVENT,
            tags=["test"],
            metadata=BlockMetadata(confidence=0.9, source=SourceType.INFERRED),
        )
        data = block.to_dict()
        restored = Block.from_dict(data)

        assert restored.id == block.id
        assert restored.type == block.type
        assert restored.content == block.content
        assert restored.metadata.confidence == block.metadata.confidence
        assert restored.metadata.source == block.metadata.source
        assert restored.tags == block.tags

    def test_to_snapshot(self):
        block = Block(
            content="User uses AWS",
            type=BlockType.FACT,
            tags=["cloud"],
            metadata=BlockMetadata(confidence=0.95),
        )
        snapshot = block.to_snapshot()
        assert snapshot["type"] == "fact"
        assert snapshot["confidence"] == 0.95
        assert "metadata" not in snapshot  # snapshot is lightweight

    def test_repr(self):
        block = Block(content="Short content", type=BlockType.FACT)
        r = repr(block)
        assert "Block(" in r
        assert "fact" in r


# ─── Edge ─────────────────────────────────────────────────────────────────────


class TestEdge:
    def test_create_edge(self):
        edge = Edge(source_id="blk_a", target_id="blk_b", relation=EdgeRelation.SUPPORTS)
        assert edge.source_id == "blk_a"
        assert edge.relation == EdgeRelation.SUPPORTS
        assert edge.weight == 1.0

    def test_edge_roundtrip(self):
        edge = Edge(
            source_id="blk_a",
            target_id="blk_b",
            relation=EdgeRelation.CONTRADICTS,
            weight=0.8,
        )
        data = edge.to_dict()
        restored = Edge.from_dict(data)
        assert restored.source_id == edge.source_id
        assert restored.relation == edge.relation
        assert restored.weight == edge.weight


# ─── BlockMetadata ────────────────────────────────────────────────────────────


class TestBlockMetadata:
    def test_default_metadata(self):
        meta = BlockMetadata()
        assert meta.confidence == 1.0
        assert meta.source == SourceType.EXPLICIT
        assert meta.access_count == 0
        assert meta.decay_rate == 0.01

    def test_metadata_roundtrip(self):
        meta = BlockMetadata(
            confidence=0.75,
            source=SourceType.OBSERVED,
            created_by="test_agent",
            decay_rate=0.05,
            ttl=3600,
        )
        data = meta.to_dict()
        restored = BlockMetadata.from_dict(data)
        assert restored.confidence == 0.75
        assert restored.source == SourceType.OBSERVED
        assert restored.ttl == 3600


# ─── Schema Validation ───────────────────────────────────────────────────────


class TestSchemaValidation:
    def test_valid_block_passes(self):
        block = Block(content="Valid content", type=BlockType.FACT)
        BlockSchema.validate_block(block)  # should not raise

    def test_empty_content_fails(self):
        block = Block(content="", type=BlockType.FACT)
        with pytest.raises(SchemaValidationError, match="content must not be empty"):
            BlockSchema.validate_block(block)

    def test_whitespace_content_fails(self):
        block = Block(content="   ", type=BlockType.FACT)
        with pytest.raises(SchemaValidationError, match="content must not be empty"):
            BlockSchema.validate_block(block)

    def test_confidence_out_of_range_fails(self):
        block = Block(
            content="Test",
            metadata=BlockMetadata(confidence=1.5),
        )
        with pytest.raises(SchemaValidationError, match="Confidence must be between"):
            BlockSchema.validate_block(block)

    def test_negative_confidence_fails(self):
        block = Block(
            content="Test",
            metadata=BlockMetadata(confidence=-0.1),
        )
        with pytest.raises(SchemaValidationError, match="Confidence must be between"):
            BlockSchema.validate_block(block)

    def test_negative_decay_rate_fails(self):
        block = Block(
            content="Test",
            metadata=BlockMetadata(decay_rate=-0.01),
        )
        with pytest.raises(SchemaValidationError, match="Decay rate must be"):
            BlockSchema.validate_block(block)

    def test_zero_ttl_fails(self):
        block = Block(
            content="Test",
            metadata=BlockMetadata(ttl=0),
        )
        with pytest.raises(SchemaValidationError, match="TTL must be positive"):
            BlockSchema.validate_block(block)

    def test_empty_tag_fails(self):
        block = Block(content="Test", tags=["valid", ""])
        with pytest.raises(SchemaValidationError, match="Tags must be non-empty"):
            BlockSchema.validate_block(block)

    def test_valid_parent_child(self):
        BlockSchema.validate_parent_child(BlockType.FACT, None)  # root parent
        BlockSchema.validate_parent_child(BlockType.FACT, BlockType.ENTITY)

    def test_invalid_parent_child(self):
        with pytest.raises(SchemaValidationError, match="cannot have parent"):
            BlockSchema.validate_parent_child(BlockType.FACT, BlockType.PREFERENCE)

    def test_valid_edge(self):
        BlockSchema.validate_edge(EdgeRelation.SUPPORTS, BlockType.FACT, BlockType.FACT)

    def test_invalid_edge_about_non_entity(self):
        with pytest.raises(SchemaValidationError, match="not allowed to block type"):
            BlockSchema.validate_edge(EdgeRelation.ABOUT, BlockType.FACT, BlockType.FACT)

    def test_caused_by_requires_event_source(self):
        with pytest.raises(SchemaValidationError, match="not allowed from block type"):
            BlockSchema.validate_edge(EdgeRelation.CAUSED_BY, BlockType.FACT, BlockType.EVENT)
