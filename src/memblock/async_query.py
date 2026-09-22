"""Async query engine — async equivalent of `query.QueryEngine`.

Mirrors the sync engine 1:1 but every storage call is awaited
through an `AsyncStorageAdapter`. Pure-Python scoring / sorting /
RRF-merge logic is reused unchanged via local imports.

Used by `AsyncMemBlock` when constructed with a `postgresql+asyncpg://`
URL — the native-async path bypasses the `asyncio.to_thread`
wrapping that the legacy AsyncMemBlock did.
"""

from __future__ import annotations

import asyncio
import time
from datetime import timezone
from typing import Any

from memblock.block import Block
from memblock.decay import DecayEngine
from memblock.embeddings import weighted_rrf_merge
from memblock.storage.async_base import AsyncStorageAdapter
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


from memblock.query import _block_matches_filters  # noqa: E402  (shared filter check)


class AsyncQueryEngine:
    """Async-native query engine.

    Same filter shape, same scoring, same hybrid (FTS + vector)
    ranking as `QueryEngine`. The only difference is `async/await`
    plumbing.

    Params:
      - `storage`: `AsyncStorageAdapter` (typically `AsyncPostgreSQLAdapter`)
      - `embedding_provider`: optional sync provider; we wrap its
        `.embed()` calls with `asyncio.to_thread` since most
        embedding APIs are HTTP and benefit from running off the
        event loop without blocking it.
      - `decay`: a regular `DecayEngine`; `calculate_strength` is a
        pure function (no I/O), reusable as-is.
      - `reranker`: optional reranker (skipped in v0; falls back to
        relevance-sorted results).
    """

    def __init__(
        self,
        storage: AsyncStorageAdapter,
        decay: DecayEngine | None = None,
        embedding_provider: object | None = None,
        reranker: object | None = None,
    ) -> None:
        self.storage = storage
        # Use a default DecayEngine when none provided. We never call
        # storage on it from the query path — only `calculate_strength`,
        # which is pure compute — so a DecayEngine pointing at a
        # non-async storage is fine here.
        self.decay = decay or DecayEngine(storage=None)  # type: ignore[arg-type]
        self._embedding_provider = embedding_provider
        self._reranker = reranker

    async def query(
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
        """Query memory blocks. See `QueryEngine.query` for full
        param semantics."""

        # ── Step 0: Temporal query parsing (extract time constraints
        # from text_search; pure Python).
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
                    cleaned_text_search = (
                        constraint.cleaned_query or text_search
                    )
            except ImportError:
                pass

        # ── Step 1: Get candidate blocks from storage
        # For text searches the storage layer ranks by full-text relevance
        # (bm25 / ts_rank) and returns only the top `pool` rows — the same
        # pool _hybrid_search re-ranks. Non-text queries keep the full set:
        # the Python-side strength sort needs it.
        pool = max(limit * 5, 50)
        filters: dict[str, Any] = {}
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
        filters.update(temporal_filters)

        candidates = await self.storage.query_blocks(filters)

        # A temporal window inferred from the query text ("last week",
        # "10 days ago") is advisory, not a user-supplied filter: if it
        # leaves no candidates at all — every memory predates the window,
        # or blocks carry no happened_at — retry without it rather than
        # return nothing. Explicit filters are never relaxed.
        if temporal_filters and not candidates:
            for key in temporal_filters:
                filters.pop(key, None)
            candidates = await self.storage.query_blocks(filters)

        # ── Step 1b: Hybrid search — FTS results + vector similarity
        hybrid_boost: dict[str, float] = {}
        if (
            text_search
            and semantic
            and self._embedding_provider is not None
        ):
            hybrid_boost = await self._hybrid_search(
                text_search, candidates, pool,
            )

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

        # ── Step 2: Graph proximity boost when `related_to` is set
        graph_boost: dict[str, float] = {}
        if related_to:
            depth_map = await self._traverse_with_depth(
                related_to, max_depth=3,
            )
            for block_id, depth in depth_map.items():
                graph_boost[block_id] = 0.3 / depth

            if not candidates:
                # Pull neighbor blocks as candidates when no other
                # filters narrowed anything down.
                fetched = await asyncio.gather(*[
                    self.storage.get_block(bid) for bid in depth_map
                ])
                for block in fetched:
                    if block and not block.deleted and _block_matches_filters(block, filters):
                        candidates.append(block)

        # ── Step 3: Strength filtering
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
            candidates = await self.storage.query_blocks(full_filters)
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

        # If hybrid found blocks not in FTS results, fetch + add them.
        # `hybrid_boost` is in merged-score order and (on the Python vector
        # fallback) carries a cosine bonus for EVERY embedded block, so only
        # its top `pool` entries are hydrated: anything below them trails at
        # least `pool` higher-scoring blocks on the 0.70-weighted term and
        # cannot realistically reach the top `limit` (pool >= 5 * limit).
        # Without this cap a hybrid query on a non-pgvector store called
        # get_block once per embedded block (O(N) per query).
        if hybrid_boost:
            candidate_ids = {b.id for b, _ in scored}
            missing_ids = [
                bid for bid in list(hybrid_boost)[:pool]
                if bid not in candidate_ids
            ]
            if missing_ids:
                fetched = await asyncio.gather(*[
                    self.storage.get_block(bid) for bid in missing_ids
                ])
                for block in fetched:
                    if block and not block.deleted and _block_matches_filters(block, filters):
                        strength = self.decay.calculate_strength(block)
                        if strength >= min_strength or include_decayed:
                            scored.append((block, strength))

        # ── Step 4: Sort
        if sort_by == "strength":
            scored.sort(key=lambda x: x[1], reverse=True)
        elif sort_by == "recency":
            scored.sort(
                key=lambda x: x[0].metadata.created_at.isoformat(),
                reverse=True,
            )
        elif sort_by == "access_count":
            scored.sort(
                key=lambda x: x[0].metadata.access_count, reverse=True,
            )
        else:  # 'relevance'
            def relevance_score(item: tuple[Block, float]) -> float:
                block, strength = item
                conf = block.metadata.confidence
                sem = normalized_hybrid.get(block.id, 0.0)
                created_ts = (
                    block.metadata.created_at
                    .astimezone(timezone.utc)
                    .timestamp()
                )
                hours_since = max(
                    0.0, (time.time() - created_ts) / 3600.0,
                )
                recency = max(0.0, 1.0 - (hours_since / (30 * 24)))
                g_boost = graph_boost.get(block.id, 0.0)
                return (
                    sem * 0.70
                    + conf * 0.10
                    + strength * 0.05
                    + recency * 0.05
                    + g_boost * 0.10
                )

            scored.sort(key=relevance_score, reverse=True)

        # ── Step 5: Reranker (best-effort, async-via-to_thread)
        if text_search and self._reranker is not None:
            try:
                from memblock.rerankers import Reranker
                if isinstance(self._reranker, Reranker):
                    wider = [block for block, _ in scored[: limit * 3]]
                    return await asyncio.to_thread(
                        self._reranker.rerank, text_search, wider, limit,
                    )
            except Exception:
                pass

        # ── Step 6: Apply limit
        return [block for block, _ in scored[:limit]]

    # ─── Internal helpers ────────────────────────────────────────────

    def _fts_only_curve(self, candidates: list[Block]) -> dict[str, float]:
        """Relevance curve from the storage (bm25 / ts_rank) order alone.

        Used when no vector signal exists; see _FTS_ONLY_RRF_K. The first
        candidate normalises to 1.0.
        """
        fts_only = dict(weighted_rrf_merge(
            [b.id for b in candidates], [], None, k=_FTS_ONLY_RRF_K,
        ))
        return self._normalize_hybrid_scores(fts_only)

    def _normalize_hybrid_scores(
        self, hybrid_boost: dict[str, float],
    ) -> dict[str, float]:
        """RRF scores → 0-1 range. Identical logic to the sync
        engine's helper of the same name."""
        if not hybrid_boost:
            return {}
        max_score = max(hybrid_boost.values())
        if max_score <= 0:
            return {}
        return {bid: score / max_score for bid, score in hybrid_boost.items()}

    async def _hybrid_search(
        self,
        query: str,
        fts_candidates: list[Block],
        limit: int,
    ) -> dict[str, float]:
        """FTS + vector hybrid search via Weighted RRF.

        Embedding is generated via the provider's sync `.embed()`
        method wrapped in `asyncio.to_thread` (the call is HTTP, so
        thread-pool is fine — keeps the event loop free).

        Vector search is done server-side via pgvector when the
        adapter supports it; otherwise falls back to brute-force
        cosine in Python over `get_all_embeddings`.
        """
        from memblock.embeddings import (
            EmbeddingProvider,
            brute_force_similarity,
            pack_embedding,
        )

        provider = self._embedding_provider
        if not isinstance(provider, EmbeddingProvider):
            return {}

        try:
            # Embed the query — sync HTTP, run in thread pool.
            query_embeddings = await asyncio.to_thread(provider.embed, [query])
            if not query_embeddings:
                return {}
            query_vec = query_embeddings[0]

            fts_ids = [b.id for b in fts_candidates[:limit]]

            # pgvector path
            query_bytes = pack_embedding(query_vec)
            server_results = await self.storage.search_similar_embeddings(
                query_bytes, limit,
            )
            if server_results:
                vec_ids = [bid for bid, _ in server_results]
                vec_scores = {bid: s for bid, s in server_results}
                merged = weighted_rrf_merge(fts_ids, vec_ids, vec_scores)
                return dict(merged)

            # Python fallback: brute-force cosine similarity (numpy matmul
            # when available, pure Python otherwise) — all rows, sorted desc.
            all_embeddings = await self.storage.get_all_embeddings()
            if not all_embeddings:
                return {}

            vec_scored = brute_force_similarity(query_vec, all_embeddings)
            vec_ids = [bid for bid, _ in vec_scored[:limit]]
            vec_scores_map = dict(vec_scored)

            merged = weighted_rrf_merge(fts_ids, vec_ids, vec_scores_map)
            return dict(merged)

        except Exception:
            return {}

    async def _traverse_with_depth(
        self,
        block_id: str,
        max_depth: int = 3,
    ) -> dict[str, int]:
        """Async BFS from `block_id` returning {block_id: depth}.

        Walks edges via `storage.get_edges(direction='both')`. Each
        depth level gathers edge fetches concurrently for speed
        — at typical depths and fanout this is much faster than
        sequential awaits.
        """
        depths: dict[str, int] = {block_id: 0}
        frontier = [block_id]

        for current_depth in range(1, max_depth + 1):
            if not frontier:
                break

            # Fetch edges for every frontier node concurrently.
            edge_lists = await asyncio.gather(*[
                self.storage.get_edges(bid, direction="both")
                for bid in frontier
            ])

            next_frontier: list[str] = []
            for edges in edge_lists:
                for edge in edges:
                    for neighbor in (edge.source_id, edge.target_id):
                        if neighbor not in depths:
                            depths[neighbor] = current_depth
                            next_frontier.append(neighbor)
            frontier = next_frontier

        # Drop the seed itself from results — caller wants "blocks
        # connected TO this one", not the one itself.
        depths.pop(block_id, None)
        return depths
