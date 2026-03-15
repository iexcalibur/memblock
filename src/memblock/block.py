"""Block model — the core unit of memory in MemBlock."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memblock.types import (
    BlockMetadata,
    BlockType,
    Edge,
    EncryptionLevel,
    generate_block_id,
)


@dataclass
class Block:
    """
    A single memory block — the fundamental unit of the MemBlock system.

    Each block represents a piece of knowledge (fact, preference, event, entity,
    or relation) with associated metadata, graph edges, and version tracking.
    """

    # Identity
    id: str = field(default_factory=generate_block_id)
    type: BlockType = BlockType.FACT
    content: str = ""

    # Metadata
    metadata: BlockMetadata = field(default_factory=BlockMetadata)

    # Encryption
    encryption_level: EncryptionLevel = EncryptionLevel.NONE
    encrypted: bool = False

    # Graph edges (relationships to other blocks)
    edges: list[Edge] = field(default_factory=list)

    # Tree structure
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)

    # Versioning & integrity
    version: int = 1
    op_hash: str = ""  # hash from operation log for tamper detection

    # Tags for categorization
    tags: list[str] = field(default_factory=list)

    # Content hash for deduplication
    content_hash: str = ""

    # Soft delete
    deleted: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize block to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "encryption_level": self.encryption_level.value,
            "encrypted": self.encrypted,
            "edges": [e.to_dict() for e in self.edges],
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "version": self.version,
            "op_hash": self.op_hash,
            "tags": self.tags,
            "content_hash": self.content_hash,
            "deleted": self.deleted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Block:
        """Deserialize block from dictionary."""
        return cls(
            id=data["id"],
            type=BlockType(data["type"]),
            content=data.get("content", ""),
            metadata=BlockMetadata.from_dict(data["metadata"]) if "metadata" in data else BlockMetadata(),
            encryption_level=EncryptionLevel(data.get("encryption_level", "none")),
            encrypted=data.get("encrypted", False),
            edges=[Edge.from_dict(e) for e in data.get("edges", [])],
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", []),
            version=data.get("version", 1),
            op_hash=data.get("op_hash", ""),
            tags=data.get("tags", []),
            content_hash=data.get("content_hash", ""),
            deleted=data.get("deleted", False),
        )

    def to_snapshot(self) -> dict[str, Any]:
        """Lightweight export for context building — no internal state."""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "confidence": self.metadata.confidence,
            "source": self.metadata.source.value,
            "tags": self.tags,
            "edge_count": len(self.edges),
        }

    def __repr__(self) -> str:
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Block(id={self.id!r}, type={self.type.value}, content={content_preview!r})"
