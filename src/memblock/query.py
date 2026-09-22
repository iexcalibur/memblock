"""Query engine — structured retrieval combining filters, graph, decay, and hybrid search."""

from __future__ import annotations

from datetime import timezone

from memblock.block import Block
from memblock.decay import DecayEngine
from memblock.embeddings import weighted_rrf_merge
from memblock.graph import GraphIndex
from memblock.storage.base import StorageAdapter
from typing import Any

from memblock.types import BlockType


# Engine-level sort_by -> storage sort_by for the text-search candidate pool.
# Explicit recency / access_count sorts ask storage for the same ordering, so
# the pool holds the newest / most-accessed matches (v0.13.1 semantics,
# bounded to the pool); "relevance" and "strength" take the best keyword
# matches.
_STORAGE_SORT_FOR_TEXT_SEARCH: dict[str, str] = {
    "recency": "created_at",
    "access_count": "access_count",
}

# RRF constant for the FTS-only relevance curve (no vector signal). The
# default k=60 is so flat after normalisation (rank 0 -> 1.0, rank 1 -> 0.984)
# that the 0.70-weighted keyword term moved the final score by ~0.01 per rank
# while recency + strength alone can swing it by 0.10, so any fresh one-token
# match outranked an older exact match. With k=3 (rank 1 -> 0.80, rank 2 ->
# 0.67, rank 9 -> 0.31) the best keyword match beats recency/strength at any
# age for equal confidence; a much higher-confidence block can still edge
# out an adjacent rank. The hybrid path keeps the standard k=60 because the
# vector side supplies score magnitude there.
_FTS_ONLY_RRF_K = 3


def _block_matches_filters(block: Block, filters: dict) -> bool:
    """Apply the SQL-level filters to a block fetched outside the candidate
    query (vector-only hybrid hits, graph neighbours), so a session- or
    type-scoped query never returns blocks from another scope."""
    md = block.metadata
    t = filters.get("type")
    if t is not None:
        tv = t.value if hasattr(t, "value") else t
        bv = block.type.value if hasattr(block.type, "value") else block.type
        if bv != tv:
            return False
    if "parent_id" in filters and block.parent_id != filters["parent_id"]:
        return False
    for key in ("session_id", "org_id", "project_id", "agent_id"):
        if key in filters and getattr(md, key, None) != filters[key]:
            return False
    if "min_confidence" in filters and (getattr(md, "confidence", 0.0) or 0.0) < filters["min_confidence"]:
        return False
    tags = filters.get("tags")
    if tags and not any(t in (block.tags or []) for t in tags):
        return False
    mf = filters.get("metadata_filters")
    if mf:
        custom = getattr(md, "custom_metadata", None) or {}
        for k, v in mf.items():
            if str(custom.get(k)) != str(v):
                return False
    happened = getattr(md, "happened_at", None)
    if "happened_after" in filters and (happened is None or happened < filters["happened_after"]):
        return False
    if "happened_before" in filters and (happened is None or happened > filters["happened_before"]):
        return False
    if "temporal_range" in filters:
        start, end = filters["temporal_range"]
        if happened is None or happened < start or happened > end:
            return False
    return True


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
            sort_by: 'relevance', 'recency', 'access_count', 'strength'.
                With text_search the candidates are the top max(limit*5, 50)
                keyword-ranked matches (newest / most-accessed matches for
                'recency' / 'access_count'); 'strength' considers every match.
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
        # For text searches the storage layer ranks by full-text relevance
        # (bm25 / ts_rank) and returns only the top `pool` rows — the same
        # pool _hybrid_search re-ranks. Non-text queries keep the full set:
        # the Python-side strength sort needs it.
        pool = max(limit * 5, 50)
        filters: dict = {}
        if type is not None:
            filters["type"] = type
        if tags:
            filters["tags"] = tags
        if cleaned_text_search:
            filters["text_search"] = cleaned_text_search
            filters["sort_by"] = _STORAGE_SORT_FOR_TEXT_SEARCH.get(sort_by, "relevance")
            if sort_by != "strength":
                # "strength" has no SQL equivalent and must see every match
                # (v0.13.1 semantics); every other sort works on the pool.
                filters["limit"] = pool
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

        # A temporal window inferred from the query text ("last week",
        # "10 days ago") is advisory, not a user-supplied filter: if it
        # leaves no candidates at all — every memory predates the window,
        # or blocks carry no happened_at — retry without it rather than
        # return nothing. Explicit filters are never relaxed.
        if temporal_filters and not candidates:
            for key in temporal_filters:
                filters.pop(key, None)
            candidates = self.storage.query_blocks(filters)

        # Step 1b: If embeddings available and text_search given, do hybrid search
        hybrid_boost: dict[str, float] = {}
        if (
            text_search
            and semantic
            and self._embedding_provider is not None
        ):
            hybrid_boost = self._hybrid_search(text_search, candidates, pool)

        # Normalize hybrid scores to 0-1 range for meaningful relevance contribution
        if hybrid_boost:
            normalized_hybrid = self._normalize_hybrid_scores(hybrid_boost)
        elif cleaned_text_search:
            # FTS-only (no provider, semantic=False, or no embeddings yet):
            # the keyword rank order from storage must still drive scoring,
            # otherwise `sem` is 0 for every block and results collapse to
            # confidence/recency order. RRF over the FTS order alone with a
            # steep k (see _FTS_ONLY_RRF_K); first candidate normalises to 1.0.
            # Kept out of `hybrid_boost` so the "add vec-only blocks" step
            # below does not re-fetch blocks the strength filter dropped.
            normalized_hybrid = self._fts_only_curve(candidates)
        else:
            normalized_hybrid = {}

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
                    if block and not block.deleted and _block_matches_filters(block, filters):
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

        # Refill: the text-search pool was cut in SQL before this Python-side
        # min_strength filter ran. If decayed top matches emptied the pool so
        # that `limit` can no longer be filled although the pool was full,
        # fall back to the full match set (v0.13.1 behaviour) so decayed
        # best matches never hide live weaker ones.
        if (
            "limit" in filters
            and min_strength > 0
            and not include_decayed
            and len(scored) < limit
            and len(candidates) >= filters["limit"]
        ):
            full_filters = {k: v for k, v in filters.items() if k != "limit"}
            candidates = self.storage.query_blocks(full_filters)
            if cleaned_text_search and not hybrid_boost:
                normalized_hybrid = self._fts_only_curve(candidates)
            scored = []
            for block in candidates:
                if block.deleted and not include_decayed:
                    continue
                strength = self.decay.calculate_strength(block)
                if strength < min_strength and not include_decayed:
                    continue
                scored.append((block, strength))

        # If hybrid search found blocks not in FTS results, add them.
        # `hybrid_boost` is in merged-score order and (on the Python vector
        # fallback) carries a cosine bonus for EVERY embedded block, so only
        # its top `pool` entries are hydrated: anything below them trails at
        # least `pool` higher-scoring blocks on the 0.70-weighted term and
        # cannot realistically reach the top `limit` (pool >= 5 * limit).
        # Without this cap a hybrid query on a non-pgvector store called
        # get_block once per embedded block (O(N) per query).
        if hybrid_boost:
            candidate_ids = {b.id for b, _ in scored}
            for block_id in list(hybrid_boost)[:pool]:
                if block_id not in candidate_ids:
                    block = self.storage.get_block(block_id)
                    if block and not block.deleted and _block_matches_filters(block, filters):
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

    def _fts_only_curve(self, candidates: list[Block]) -> dict[str, float]:
        """Relevance curve from the storage (bm25 / ts_rank) order alone.

        Used when no vector signal exists; see _FTS_ONLY_RRF_K. The first
        candidate normalises to 1.0.
        """
        fts_only = dict(weighted_rrf_merge(
            [b.id for b in candidates], [], None, k=_FTS_ONLY_RRF_K,
        ))
        return self._normalize_hybrid_scores(fts_only)

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
            brute_force_similarity,
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

            # Fallback: brute-force cosine similarity (numpy matmul when
            # available, pure Python otherwise) — all rows, sorted desc.
            all_embeddings = self.storage.get_all_embeddings()
            if not all_embeddings:
                return {}

            vec_scored = brute_force_similarity(query_vec, all_embeddings)
            vec_ids = [bid for bid, _ in vec_scored[:limit]]
            vec_scores_map = dict(vec_scored)

            merged = weighted_rrf_merge(fts_ids, vec_ids, vec_scores_map)
            return dict(merged)

        except Exception:
            return {}
