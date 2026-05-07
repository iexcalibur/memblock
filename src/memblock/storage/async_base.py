"""Async-native storage adapter interface for MemBlock.

Parallel to `memblock.storage.base.StorageAdapter` but every I/O
method is `async def` so consumers running under asyncio (FastAPI,
aiohttp, asyncio chat handlers) can do native non-blocking I/O.

The first implementation is `AsyncPostgreSQLAdapter` (asyncpg-backed).
A SQLite async adapter could land later if there's demand — most
async consumers will be on Postgres.

Why a separate ABC instead of subclassing `StorageAdapter`:
  - LSP would be violated: `def save_block` → `async def save_block`
    isn't substitutable on the type system level.
  - Consumers know in advance whether they want sync or async; routing
    through `AsyncMemBlock` (which detects URL form) keeps the API
    discoverable without forcing every storage implementation to be
    dual-mode.

Method surface mirrors `StorageAdapter` 1:1 with `async` prefixes.
Analytics methods are deliberately omitted from v0 — most async
consumers don't enable that subsystem; they can be added when there's
a concrete consumer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from memblock.block import Block
from memblock.types import Edge, Operation


class AsyncStorageAdapter(ABC):
    """Abstract async storage backend for MemBlock.

    Every I/O method returns a coroutine. `initialize` and `close`
    are also async because they may issue queries (CREATE TABLE,
    connection close, etc.).
    """

    # ─── Adapter Identity ─────────────────────────────────────────────

    @property
    def adapter_type(self) -> str:
        """Identifier string ('postgresql_async', 'sqlite_async')."""
        return "unknown"

    async def run_migration_sql(
        self, sql: str, params: tuple | None = None,
    ) -> None:
        """Execute a migration SQL statement. Override in subclass."""
        return None

    # ─── Block Operations ─────────────────────────────────────────────

    @abstractmethod
    async def save_block(self, block: Block) -> None:
        """Persist a block (insert or replace)."""
        ...

    @abstractmethod
    async def get_block(self, block_id: str) -> Block | None:
        """Retrieve a block by ID. Returns None when not found."""
        ...

    @abstractmethod
    async def delete_block(self, block_id: str) -> None:
        """Hard-delete a block by ID."""
        ...

    @abstractmethod
    async def update_block(
        self, block_id: str, updates: dict[str, Any],
    ) -> None:
        """Update specific fields of a block."""
        ...

    @abstractmethod
    async def query_blocks(
        self, filters: dict[str, Any],
    ) -> list[Block]:
        """Query blocks with the same filter shape as
        `StorageAdapter.query_blocks`."""
        ...

    @abstractmethod
    async def get_all_blocks(
        self, include_deleted: bool = False,
    ) -> list[Block]:
        """Return every block for the adapter's user_id scope."""
        ...

    async def get_block_by_content_hash(
        self, content_hash: str,
    ) -> Block | None:
        """Find a non-deleted block by content hash. Override for
        dedup support; default returns None."""
        return None

    # ─── Edge Operations ──────────────────────────────────────────────

    @abstractmethod
    async def save_edge(self, edge: Edge) -> None:
        """Persist an edge."""
        ...

    @abstractmethod
    async def get_edges(
        self, block_id: str, direction: str = "both",
    ) -> list[Edge]:
        """Get edges for a block. `direction`: 'outgoing' /
        'incoming' / 'both'."""
        ...

    @abstractmethod
    async def delete_edge(self, edge_id: str) -> None:
        """Delete a specific edge."""
        ...

    @abstractmethod
    async def delete_edges_for_block(self, block_id: str) -> None:
        """Delete every edge involving a block (source or target)."""
        ...

    # ─── Operation Log ────────────────────────────────────────────────

    @abstractmethod
    async def save_operation(self, op: Operation) -> None:
        """Append an operation to the audit log."""
        ...

    @abstractmethod
    async def get_operations(
        self, block_id: str | None = None,
    ) -> list[Operation]:
        """Get operations, ordered by clock ascending. Optional
        filter by block_id."""
        ...

    @abstractmethod
    async def get_last_operation(self) -> Operation | None:
        """Return the most recent operation (for hash chaining)."""
        ...

    # ─── Embedding Operations ────────────────────────────────────────

    async def save_embedding(
        self, block_id: str, embedding: bytes,
    ) -> None:
        """Save an embedding vector. Override in subclass."""
        return None

    async def get_embedding(self, block_id: str) -> bytes | None:
        """Get the embedding for a block. Override in subclass."""
        return None

    async def get_all_embeddings(self) -> list[tuple[str, bytes]]:
        """Get all (block_id, embedding) pairs. Override in
        subclass."""
        return []

    async def delete_embedding(self, block_id: str) -> None:
        """Delete the embedding for a block. Override in subclass."""
        return None

    async def search_similar_embeddings(
        self, query_embedding: bytes, limit: int = 20,
    ) -> list[tuple[str, float]]:
        """Server-side similarity search. Override for native
        (e.g. pgvector). Default returns empty so callers fall back
        to Python cosine over `get_all_embeddings`."""
        return []

    # ─── Lifecycle ────────────────────────────────────────────────────

    @abstractmethod
    async def initialize(self) -> None:
        """Create tables, indexes, triggers, extensions."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the storage connection / pool."""
        ...
