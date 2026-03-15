"""Core types, enums, and data structures for MemBlock SDK."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ─── Enums ────────────────────────────────────────────────────────────────────


class BlockType(str, Enum):
    """Types of memory blocks."""

    FACT = "fact"  # Objective information: "User works at Google"
    PREFERENCE = "preference"  # User preferences: "Prefers dark mode"
    EVENT = "event"  # Something that happened: "Deployed v2.0 on March 10"
    ENTITY = "entity"  # A named thing: "Project Alpha", "React", "AWS"
    RELATION = "relation"  # A relationship description between entities


class SourceType(str, Enum):
    """How the memory was acquired."""

    EXPLICIT = "explicit"  # User directly stated it
    INFERRED = "inferred"  # Derived from context by the agent
    OBSERVED = "observed"  # Noticed from user behavior/patterns


class EncryptionLevel(str, Enum):
    """Per-block encryption levels."""

    NONE = "none"  # Content stored as plaintext
    STANDARD = "standard"  # Content encrypted, metadata visible
    SENSITIVE = "sensitive"  # Content + tags encrypted


class EdgeRelation(str, Enum):
    """Types of relationships between blocks."""

    ABOUT = "about"  # Block is about another block/entity
    SUPPORTS = "supports"  # Evidence supporting another block
    CONTRADICTS = "contradicts"  # Conflicts with another block
    CAUSED_BY = "caused_by"  # Causal relationship
    RELATED_TO = "related_to"  # General association
    PART_OF = "part_of"  # Hierarchical membership
    DERIVED_FROM = "derived_from"  # One block derived from another
    SUPERSEDES = "supersedes"  # Newer version of information


class OpAction(str, Enum):
    """Types of operations in the operation log."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    LINK = "link"
    UNLINK = "unlink"


# ─── Data Structures ─────────────────────────────────────────────────────────


def generate_block_id() -> str:
    """Generate a unique block ID with blk_ prefix."""
    return f"blk_{uuid.uuid4().hex[:12]}"


def generate_edge_id() -> str:
    """Generate a unique edge ID with edg_ prefix."""
    return f"edg_{uuid.uuid4().hex[:12]}"


def generate_op_id() -> str:
    """Generate a unique operation ID with op_ prefix."""
    return f"op_{uuid.uuid4().hex[:12]}"


def now_utc() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass
class Edge:
    """A directed relationship between two blocks."""

    id: str = field(default_factory=generate_edge_id)
    source_id: str = ""
    target_id: str = ""
    relation: EdgeRelation = EdgeRelation.RELATED_TO
    weight: float = 1.0
    created_at: datetime = field(default_factory=now_utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation.value,
            "weight": self.weight,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Edge:
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation=EdgeRelation(data["relation"]),
            weight=data.get("weight", 1.0),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass
class BlockMetadata:
    """Metadata attached to every memory block."""

    confidence: float = 1.0  # 0.0 to 1.0
    source: SourceType = SourceType.EXPLICIT
    created_at: datetime = field(default_factory=now_utc)
    created_by: str = "agent"  # who/what created this block
    access_count: int = 0
    last_accessed: datetime | None = None
    decay_rate: float = 0.01  # how fast memory fades (0 = never)
    ttl: int | None = None  # seconds until expiry, None = permanent
    session_id: str | None = None  # optional session scope

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "source": self.source.value,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "decay_rate": self.decay_rate,
            "ttl": self.ttl,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BlockMetadata:
        return cls(
            confidence=data.get("confidence", 1.0),
            source=SourceType(data.get("source", "explicit")),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else now_utc(),
            created_by=data.get("created_by", "agent"),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
            decay_rate=data.get("decay_rate", 0.01),
            ttl=data.get("ttl"),
            session_id=data.get("session_id"),
        )


@dataclass
class Operation:
    """A single operation in the append-only log."""

    id: str = field(default_factory=generate_op_id)
    block_id: str = ""
    author: str = "agent"
    clock: int = 0  # incrementing version counter
    action: OpAction = OpAction.CREATE
    data: dict[str, Any] = field(default_factory=dict)
    hash: str = ""  # SHA-256 of (prev_hash + action + data + timestamp)
    prev_hash: str = ""  # hash of previous operation
    timestamp: datetime = field(default_factory=now_utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "block_id": self.block_id,
            "author": self.author,
            "clock": self.clock,
            "action": self.action.value,
            "data": self.data,
            "hash": self.hash,
            "prev_hash": self.prev_hash,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Operation:
        return cls(
            id=data["id"],
            block_id=data.get("block_id", ""),
            author=data.get("author", "agent"),
            clock=data.get("clock", 0),
            action=OpAction(data["action"]),
            data=data.get("data", {}),
            hash=data.get("hash", ""),
            prev_hash=data.get("prev_hash", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else now_utc(),
        )
