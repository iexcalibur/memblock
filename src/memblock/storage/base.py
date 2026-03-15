"""Abstract storage adapter interface for MemBlock."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from memblock.block import Block
from memblock.types import Edge, Operation


class StorageAdapter(ABC):
    """
    Abstract base class for all MemBlock storage backends.

    Implementations: SQLiteAdapter (local), PostgreSQLAdapter (cloud/multi-user).
    """

    # ─── Adapter Identity ─────────────────────────────────────────────────

    @property
    def adapter_type(self) -> str:
        """Return the adapter type identifier ('sqlite' or 'postgresql')."""
        return "unknown"

    def run_migration_sql(self, sql: str, params: tuple | None = None) -> None:
        """Execute a migration SQL statement. Override in subclass."""
        pass

    # ─── Block Operations ─────────────────────────────────────────────────

    @abstractmethod
    def save_block(self, block: Block) -> None:
        """Persist a block (insert or replace)."""
        ...

    @abstractmethod
    def get_block(self, block_id: str) -> Block | None:
        """Retrieve a block by ID. Returns None if not found."""
        ...

    @abstractmethod
    def delete_block(self, block_id: str) -> None:
        """Delete a block by ID."""
        ...

    @abstractmethod
    def update_block(self, block_id: str, updates: dict[str, Any]) -> None:
        """Update specific fields of a block."""
        ...

    @abstractmethod
    def query_blocks(self, filters: dict[str, Any]) -> list[Block]:
        """
        Query blocks with filters.

        Supported filters:
        - type: BlockType
        - tags: list[str] (match any)
        - min_confidence: float
        - text_search: str (FTS)
        - parent_id: str
        - deleted: bool (default False)
        - limit: int
        - sort_by: str (created_at, access_count, confidence)
        """
        ...

    @abstractmethod
    def get_all_blocks(self, include_deleted: bool = False) -> list[Block]:
        """Retrieve all blocks."""
        ...

    # ─── Edge Operations ──────────────────────────────────────────────────

    @abstractmethod
    def save_edge(self, edge: Edge) -> None:
        """Persist an edge."""
        ...

    @abstractmethod
    def get_edges(self, block_id: str, direction: str = "both") -> list[Edge]:
        """
        Get edges for a block.

        direction: 'outgoing' (source=block_id), 'incoming' (target=block_id), 'both'
        """
        ...

    @abstractmethod
    def delete_edge(self, edge_id: str) -> None:
        """Delete a specific edge."""
        ...

    @abstractmethod
    def delete_edges_for_block(self, block_id: str) -> None:
        """Delete all edges involving a block (source or target)."""
        ...

    # ─── Operation Log ────────────────────────────────────────────────────

    @abstractmethod
    def save_operation(self, op: Operation) -> None:
        """Append an operation to the log."""
        ...

    @abstractmethod
    def get_operations(self, block_id: str | None = None) -> list[Operation]:
        """
        Get operations, optionally filtered by block_id.

        Returns operations ordered by clock (ascending).
        """
        ...

    @abstractmethod
    def get_last_operation(self) -> Operation | None:
        """Get the most recent operation (for hash chaining)."""
        ...

    # ─── Embedding Operations ────────────────────────────────────────────

    def save_embedding(self, block_id: str, embedding: bytes) -> None:
        """Save an embedding vector for a block. Override in subclass."""
        pass

    def get_embedding(self, block_id: str) -> bytes | None:
        """Get the embedding for a block. Override in subclass."""
        return None

    def get_all_embeddings(self) -> list[tuple[str, bytes]]:
        """Get all (block_id, embedding) pairs. Override in subclass."""
        return []

    def delete_embedding(self, block_id: str) -> None:
        """Delete the embedding for a block. Override in subclass."""
        pass

    # ─── Deduplication ────────────────────────────────────────────────────

    def get_block_by_content_hash(self, content_hash: str) -> Block | None:
        """Get a block by content hash. Override in subclass for dedup support."""
        return None

    # ─── Lifecycle ────────────────────────────────────────────────────────

    @abstractmethod
    def close(self) -> None:
        """Close the storage connection."""
        ...

    @abstractmethod
    def initialize(self) -> None:
        """Initialize storage (create tables, indexes, etc.)."""
        ...
