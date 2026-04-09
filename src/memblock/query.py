"""Query engine — structured retrieval combining filters, graph, decay, and hybrid search."""

from __future__ import annotations

from datetime import timezone

from memblock.block import Block
from memblock.decay import DecayEngine
from memblock.graph import GraphIndex
from memblock.storage.base import StorageAdapter
from typing import Any

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
        reranker: object | None = None,
    ) -> None:
        self.storage = storage
        self.graph = graph
        self.decay = decay
        self._embedding_provider = embedding_provider
        self._reranker = reranker

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
        org_id: str | None = None,
        project_id: str | None = None,
        agent_id: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
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
        # Temporal query parsing: extract time constraints before search
        temporal_filters: dict[str, Any] = {}
        cleaned_text_search = text_search
        if text_search:
            try:
                from memblock.temporal import TemporalQueryParser
                parser = TemporalQueryParser()
                constraint = parser.parse(text_search)
                if constraint is not None:
                    if constraint.after is not None:
                        temporal_filters["happened_after"] = constraint.after
                    if constraint.before is not None:
                        temporal_filters["happened_before"] = constraint.before
                    cleaned_text_search = constraint.cleaned_query or text_search
            except ImportError:
                pass

        # Step 1: Get candidate blocks from storage
        filters: dict = {}
        if type is not None:
            filters["type"] = type
        if tags:
            filters["tags"] = tags
        if cleaned_text_search:
            filters["text_search"] = cleaned_text_search
        if min_confidence > 0:
            filters["min_confidence"] = min_confidence
        if session_id is not None:
            filters["session_id"] = session_id
        if org_id is not None:
            filters["org_id"] = org_id
        if project_id is not None:
            filters["project_id"] = project_id
        if agent_id is not None:
            filters["agent_id"] = agent_id
        if metadata_filters:
            filters["metadata_filters"] = metadata_filters
        # Add temporal filters
        filters.update(temporal_filters)

        candidates = self.storage.query_blocks(filters)

        # Step 1b: If embeddings available and text_search given, do hybrid search
        hybrid_boost: dict[str, float] = {}
        if (
            text_search
            and semantic
            and self._embedding_provider is not None
        ):
            hybrid_boost = self._hybrid_search(text_search, candidates, max(limit * 5, 50))

        # Normalize hybrid scores to 0-1 range for meaningful relevance contribution
        normalized_hybrid = self._normalize_hybrid_scores(hybrid_boost)

        # Step 2: If related_to is specified, compute graph proximity boost (union, not intersection)
        graph_boost: dict[str, float] = {}
        if related_to:
            depth_map = self.graph.traverse_with_depth(related_to, max_depth=3)
            for block_id, depth in depth_map.items():
                graph_boost[block_id] = 0.3 / depth  # closer = higher boost

            # If no other filters, also add graph neighbor blocks as candidates
            if not candidates:
                for block_id in depth_map:
                    block = self.storage.get_block(block_id)
                    if block and not block.deleted:
                        candidates.append(block)

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
        else:  # relevance — semantic-first scoring with graph proximity
            def relevance_score(item: tuple[Block, float]) -> float:
                block, strength = item
                conf = block.metadata.confidence
                sem = normalized_hybrid.get(block.id, 0.0)
                # Recency bonus: linear decay over 30 days
                hours_age = (
                    block.metadata.created_at.astimezone(timezone.utc)
                    .timestamp()
                )
                import time
                hours_since = max(0, (time.time() - hours_age) / 3600)
                recency = max(0.0, 1.0 - (hours_since / (30 * 24)))
                # Graph proximity boost
                g_boost = graph_boost.get(block.id, 0.0)
                return (sem * 0.70) + (conf * 0.10) + (strength * 0.05) + (recency * 0.05) + (g_boost * 0.10)
            scored.sort(key=relevance_score, reverse=True)

        # Step 5: Rerank first (on wider pool), THEN apply limit
        if text_search and self._reranker is not None:
            try:
                from memblock.rerankers import Reranker
                if isinstance(self._reranker, Reranker):
                    # Give reranker a wider pool for better results
                    wider = [block for block, _ in scored[:limit * 3]]
                    results = self._reranker.rerank(text_search, wider, limit)
                    return results
            except Exception:
                pass  # reranker failure should not break queries

        # Apply limit
        results = [block for block, _ in scored[:limit]]
        return results

    def _normalize_hybrid_scores(self, hybrid_boost: dict[str, float]) -> dict[str, float]:
        """Normalize RRF scores to 0-1 range for meaningful relevance contribution."""
        if not hybrid_boost:
            return {}
        max_score = max(hybrid_boost.values())
        if max_score <= 0:
            return {}
        return {bid: score / max_score for bid, score in hybrid_boost.items()}

    def _hybrid_search(
        self,
        query: str,
        fts_candidates: list[Block],
        limit: int,
    ) -> dict[str, float]:
        """
        Run vector search and merge with FTS results using Weighted Reciprocal Rank Fusion.

        Returns a dict of block_id -> RRF boost score.
        """
        from memblock.embeddings import (
            EmbeddingProvider,
            cosine_similarity,
            weighted_rrf_merge,
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

            # FTS ranked list
            fts_ids = [b.id for b in fts_candidates[:limit]]

            # Try server-side vector search first (pgvector)
            from memblock.embeddings import pack_embedding
            query_bytes = pack_embedding(query_vec)
            server_results = self.storage.search_similar_embeddings(
                query_bytes, limit
            )
            if server_results:
                vec_ids = [bid for bid, _ in server_results]
                vec_scores = {bid: score for bid, score in server_results}
                merged = weighted_rrf_merge(fts_ids, vec_ids, vec_scores)
                return dict(merged)

            # Fallback: brute-force cosine similarity in Python
            all_embeddings = self.storage.get_all_embeddings()
            if not all_embeddings:
                return {}

            vec_scored: list[tuple[str, float]] = []
            vec_scores_map: dict[str, float] = {}
            for block_id, emb_bytes in all_embeddings:
                emb = unpack_embedding(emb_bytes)
                score = cosine_similarity(query_vec, emb)
                vec_scored.append((block_id, score))
                vec_scores_map[block_id] = score

            vec_scored.sort(key=lambda x: x[1], reverse=True)
            vec_ids = [bid for bid, _ in vec_scored[:limit]]

            merged = weighted_rrf_merge(fts_ids, vec_ids, vec_scores_map)
            return dict(merged)

        except Exception:
            return {}
