"""AsyncMemBlock — async wrapper around MemBlock for use in async frameworks."""

from __future__ import annotations

import asyncio
from typing import Any

from memblock.block import Block
from memblock.memblock import MemBlock
from memblock.ops import TamperReport
from memblock.types import (
    BlockType,
    EdgeRelation,
    EncryptionLevel,
    SourceType,
)


class AsyncMemBlock:
    """
    Async-compatible wrapper around MemBlock.

    All I/O methods are wrapped with asyncio.to_thread() so they don't
    block the event loop. Uses the same MemBlock instance internally.

    Usage:
        async with AsyncMemBlock(storage="sqlite:///./memory.db") as mem:
            block = await mem.store("User prefers Python", type=BlockType.PREFERENCE)
            results = await mem.query(type=BlockType.PREFERENCE)
            context = await mem.build_context(query="user preferences")
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize AsyncMemBlock with the same parameters as MemBlock.

        All constructor args are passed directly to MemBlock.
        """
        self._mem = MemBlock(**kwargs)

    @property
    def sync(self) -> MemBlock:
        """Access the underlying sync MemBlock instance."""
        return self._mem

    # ─── Store Operations ─────────────────────────────────────────────────

    async def store(
        self,
        content: str,
        type: BlockType = BlockType.FACT,
        confidence: float = 1.0,
        source: SourceType = SourceType.EXPLICIT,
        tags: list[str] | None = None,
        parent_id: str | None = None,
        encryption_level: EncryptionLevel = EncryptionLevel.NONE,
        decay_rate: float = 0.01,
        ttl: int | None = None,
        session_id: str | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Block:
        """Store a new memory block (async)."""
        return await asyncio.to_thread(
            self._mem.store,
            content=content,
            type=type,
            confidence=confidence,
            source=source,
            tags=tags,
            parent_id=parent_id,
            encryption_level=encryption_level,
            decay_rate=decay_rate,
            ttl=ttl,
            session_id=session_id,
            org_id=org_id,
            project_id=project_id,
            agent_id=agent_id,
            metadata=metadata,
        )

    async def get(self, block_id: str, decrypt: bool = True) -> Block | None:
        """Retrieve a block by ID (async)."""
        return await asyncio.to_thread(self._mem.get, block_id, decrypt)

    async def update(self, block_id: str, **updates: Any) -> Block | None:
        """Update a block's fields (async)."""
        return await asyncio.to_thread(self._mem.update, block_id, **updates)

    async def delete(self, block_id: str, cascade: bool = False) -> bool:
        """Soft-delete a block (async)."""
        return await asyncio.to_thread(self._mem.delete, block_id, cascade)

    # ─── Graph Operations ─────────────────────────────────────────────────

    async def link(
        self,
        source_id: str,
        target_id: str,
        relation: EdgeRelation | str = EdgeRelation.RELATED_TO,
        weight: float = 1.0,
    ) -> None:
        """Create a relationship between two blocks (async)."""
        await asyncio.to_thread(self._mem.link, source_id, target_id, relation, weight)

    async def unlink(
        self,
        source_id: str,
        target_id: str,
        relation: EdgeRelation | str | None = None,
    ) -> int:
        """Remove relationship(s) between two blocks (async)."""
        return await asyncio.to_thread(self._mem.unlink, source_id, target_id, relation)

    async def neighbors(self, block_id: str, relation: EdgeRelation | None = None) -> list[Block]:
        """Get blocks directly connected to a block (async)."""
        return await asyncio.to_thread(self._mem.neighbors, block_id, relation)

    async def traverse(self, block_id: str, max_depth: int = 3) -> list[Block]:
        """Walk the graph from a block (async)."""
        return await asyncio.to_thread(self._mem.traverse, block_id, max_depth)

    # ─── Query ────────────────────────────────────────────────────────────

    async def query(
        self,
        type: BlockType | None = None,
        tags: list[str] | None = None,
        text_search: str | None = None,
        related_to: str | None = None,
        min_confidence: float = 0.0,
        sort_by: str = "relevance",
        limit: int = 10,
        semantic: bool = True,
        session_id: str | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        agent_id: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[Block]:
        """Query memory blocks with structured filters (async)."""
        return await asyncio.to_thread(
            self._mem.query,
            type=type,
            tags=tags,
            text_search=text_search,
            related_to=related_to,
            min_confidence=min_confidence,
            sort_by=sort_by,
            limit=limit,
            semantic=semantic,
            session_id=session_id,
            org_id=org_id,
            project_id=project_id,
            agent_id=agent_id,
            metadata_filters=metadata_filters,
        )

    # ─── Context Builder ──────────────────────────────────────────────────

    async def build_context(
        self,
        query: str | None = None,
        token_budget: int = 4000,
        strategy: str = "relevance",
        include_metadata: bool = True,
        session_id: str | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        agent_id: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> str:
        """Build LLM-ready context from relevant memory blocks (async)."""
        return await asyncio.to_thread(
            self._mem.build_context,
            query=query,
            token_budget=token_budget,
            strategy=strategy,
            include_metadata=include_metadata,
            session_id=session_id,
            org_id=org_id,
            project_id=project_id,
            agent_id=agent_id,
            metadata_filters=metadata_filters,
        )

    # ─── Extraction ───────────────────────────────────────────────────────

    async def extract(self, conversation: str, **kwargs: Any) -> Any:
        """Auto-extract memory blocks from a conversation (async)."""
        return await asyncio.to_thread(self._mem.extract, conversation, **kwargs)

    async def extract_messages(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        """Auto-extract from a list of message dicts (async)."""
        return await asyncio.to_thread(self._mem.extract_messages, messages, **kwargs)

    # ─── Session Operations ────────────────────────────────────────────────

    async def get_sessions(self) -> list[str]:
        """Get all distinct session IDs (async)."""
        return await asyncio.to_thread(self._mem.get_sessions)

    async def get_session_history(self, session_id: str, limit: int = 100) -> list[Block]:
        """Get blocks for a specific session (async)."""
        return await asyncio.to_thread(self._mem.get_session_history, session_id, limit)

    # ─── Integrity ────────────────────────────────────────────────────────

    async def verify(self) -> TamperReport:
        """Verify the integrity of the operation log (async)."""
        return await asyncio.to_thread(self._mem.verify)

    # ─── Decay & Maintenance ──────────────────────────────────────────────

    async def prune(self, min_strength: float = 0.1) -> list[Block]:
        """Remove decayed memories below the strength threshold (async)."""
        return await asyncio.to_thread(self._mem.prune, min_strength)

    async def strongest(self, limit: int = 10) -> list[tuple[Block, float]]:
        """Get the strongest memories (async)."""
        return await asyncio.to_thread(self._mem.strongest, limit)

    async def weakest(self, limit: int = 10) -> list[tuple[Block, float]]:
        """Get the weakest memories (async)."""
        return await asyncio.to_thread(self._mem.weakest, limit)

    # ─── Hooks ────────────────────────────────────────────────────────────

    def on(self, event: str, callback: Any) -> None:
        """Register a callback for a memory lifecycle event."""
        self._mem.on(event, callback)

    # ─── Properties ───────────────────────────────────────────────────────

    @property
    def has_embeddings(self) -> bool:
        """Whether embedding-based semantic search is enabled."""
        return self._mem.has_embeddings

    async def stats(self) -> dict[str, Any]:
        """Get statistics about the memory store (async)."""
        return await asyncio.to_thread(self._mem.stats)

    async def export_markdown(self) -> str:
        """Export all memories as human-readable markdown (async)."""
        return await asyncio.to_thread(self._mem.export_markdown)

    # ─── Lifecycle ────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the storage connection (async)."""
        await asyncio.to_thread(self._mem.close)

    async def __aenter__(self) -> AsyncMemBlock:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __repr__(self) -> str:
        return f"Async{repr(self._mem)}"
