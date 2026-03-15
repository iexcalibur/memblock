"""Query engine — structured retrieval combining filters, graph, decay, and hybrid search."""

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
    vector similarity search, RRF hybrid merge,
    graph proximity, confidence thresholds, and decay-adjusted scoring.
    """

    def __init__(
        self,
        storage: StorageAdapter,
        graph: GraphIndex,
        decay: DecayEngine,
        embedding_provider: object | None = None,
    ) -> None:
        self.storage = storage
        self.graph = graph
        self.decay = decay
        self._embedding_provider = embedding_provider

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
        semantic: bool = True,
        session_id: str | None = None,
    ) -> list[Block]:
        """
        Query memory blocks with multiple filter dimensions.

        Args:
            type: Filter by block type
            tags: Filter by tags (match any)
            text_search: Full-text search query (FTS5 + optional vector)
            related_to: Block ID — return blocks connected via graph
            min_confidence: Minimum confidence threshold
            sort_by: 'relevance', 'recency', 'access_count', 'strength'
            limit: Maximum results
            include_decayed: Include blocks with low strength
            min_strength: Minimum decay-adjusted strength (0.0-1.0)
            semantic: Enable hybrid search (FTS + vector) when embeddings are available

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
        if session_id is not None:
            filters["session_id"] = session_id

        candidates = self.storage.query_blocks(filters)

        # Step 1b: If embeddings available and text_search given, do hybrid search
        hybrid_boost: dict[str, float] = {}
        if (
            text_search
            and semantic
            and self._embedding_provider is not None
        ):
            hybrid_boost = self._hybrid_search(text_search, candidates, limit * 2)

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

        # If hybrid search found blocks not in FTS results, add them
        if hybrid_boost:
            candidate_ids = {b.id for b, _ in scored}
            for block_id in hybrid_boost:
                if block_id not in candidate_ids:
                    block = self.storage.get_block(block_id)
                    if block and not block.deleted:
                        strength = self.decay.calculate_strength(block)
                        if strength >= min_strength or include_decayed:
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
        else:  # relevance — combine confidence, strength, access, + hybrid boost
            def relevance_score(item: tuple[Block, float]) -> float:
                block, strength = item
                conf = block.metadata.confidence
                access = block.metadata.access_count
                base = (strength * 0.4) + (conf * 0.3) + (min(access / 100, 1.0) * 0.3)
                # Add hybrid boost if available (0-0.03 range from RRF)
                boost = hybrid_boost.get(block.id, 0.0)
                return base + boost
            scored.sort(key=relevance_score, reverse=True)

        # Step 5: Apply limit and return blocks
        return [block for block, _ in scored[:limit]]

    def _hybrid_search(
        self,
        query: str,
        fts_candidates: list[Block],
        limit: int,
    ) -> dict[str, float]:
        """
        Run vector search and merge with FTS results using Reciprocal Rank Fusion.

        Returns a dict of block_id -> RRF boost score.
        """
        from memblock.embeddings import (
            EmbeddingProvider,
            cosine_similarity,
            rrf_merge,
            unpack_embedding,
        )

        provider = self._embedding_provider
        if not isinstance(provider, EmbeddingProvider):
            return {}

        try:
            # Embed the query
            query_embeddings = provider.embed([query])
            if not query_embeddings:
                return {}
            query_vec = query_embeddings[0]

            # Get all stored embeddings
            all_embeddings = self.storage.get_all_embeddings()
            if not all_embeddings:
                return {}

            # Score each block by cosine similarity
            vec_scored: list[tuple[str, float]] = []
            for block_id, emb_bytes in all_embeddings:
                emb = unpack_embedding(emb_bytes)
                score = cosine_similarity(query_vec, emb)
                vec_scored.append((block_id, score))

            # Sort by similarity (descending)
            vec_scored.sort(key=lambda x: x[1], reverse=True)
            vec_ids = [bid for bid, _ in vec_scored[:limit]]

            # FTS ranked list
            fts_ids = [b.id for b in fts_candidates[:limit]]

            # RRF merge
            merged = rrf_merge(fts_ids, vec_ids)
            return dict(merged)

        except Exception:
            return {}
