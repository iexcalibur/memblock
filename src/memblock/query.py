"""Query engine — structured retrieval combining filters, graph, and decay."""

from __future__ import annotations

from memblock.block import Block
from memblock.decay import DecayEngine
from memblock.graph import GraphIndex
from memblock.storage.base import StorageAdapter
from memblock.types import BlockType


class QueryEngine:
    """
    Structured query engine for memory retrieval.

    Combines: type filtering, tag matching, text search (FTS5),
    graph proximity, confidence thresholds, and decay-adjusted scoring.
    """

    def __init__(
        self,
        storage: StorageAdapter,
        graph: GraphIndex,
        decay: DecayEngine,
    ) -> None:
        self.storage = storage
        self.graph = graph
        self.decay = decay

    def query(
        self,
        type: BlockType | None = None,
        tags: list[str] | None = None,
        text_search: str | None = None,
        related_to: str | None = None,
        min_confidence: float = 0.0,
        sort_by: str = "relevance",
        limit: int = 10,
        include_decayed: bool = False,
        min_strength: float = 0.0,
    ) -> list[Block]:
        """
        Query memory blocks with multiple filter dimensions.

        Args:
            type: Filter by block type
            tags: Filter by tags (match any)
            text_search: Full-text search query (FTS5)
            related_to: Block ID — return blocks connected via graph
            min_confidence: Minimum confidence threshold
            sort_by: 'relevance', 'recency', 'access_count', 'strength'
            limit: Maximum results
            include_decayed: Include blocks with low strength
            min_strength: Minimum decay-adjusted strength (0.0-1.0)

        Returns:
            List of blocks, sorted by the specified criteria.
        """
        # Step 1: Get candidate blocks from storage
        filters: dict = {}
        if type is not None:
            filters["type"] = type
        if tags:
            filters["tags"] = tags
        if text_search:
            filters["text_search"] = text_search
        if min_confidence > 0:
            filters["min_confidence"] = min_confidence

        candidates = self.storage.query_blocks(filters)

        # Step 2: If related_to is specified, intersect with graph neighbors
        if related_to:
            graph_blocks = self.graph.traverse(related_to, max_depth=3)
            graph_ids = {b.id for b in graph_blocks}

            if candidates:
                # Intersect: keep only candidates that are in the graph neighborhood
                candidates = [b for b in candidates if b.id in graph_ids]
            else:
                # No other filters — use graph results directly
                candidates = graph_blocks

        # Step 3: Calculate strength and filter by min_strength
        scored: list[tuple[Block, float]] = []
        for block in candidates:
            if block.deleted and not include_decayed:
                continue

            strength = self.decay.calculate_strength(block)

            if strength < min_strength and not include_decayed:
                continue

            scored.append((block, strength))

        # Step 4: Sort
        if sort_by == "strength":
            scored.sort(key=lambda x: x[1], reverse=True)
        elif sort_by == "recency":
            scored.sort(
                key=lambda x: x[0].metadata.created_at.isoformat(),
                reverse=True,
            )
        elif sort_by == "access_count":
            scored.sort(key=lambda x: x[0].metadata.access_count, reverse=True)
        else:  # relevance — combine confidence, strength, access
            def relevance_score(item: tuple[Block, float]) -> float:
                block, strength = item
                conf = block.metadata.confidence
                access = block.metadata.access_count
                # Weighted combination
                return (strength * 0.4) + (conf * 0.3) + (min(access / 100, 1.0) * 0.3)

            scored.sort(key=relevance_score, reverse=True)

        # Step 5: Apply limit and return blocks
        return [block for block, _ in scored[:limit]]
