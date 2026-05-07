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
from memblock.storage.async_base import AsyncStorageAdapter
from memblock.types import BlockType


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
        filters: dict[str, Any] = {}
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
        filters.update(temporal_filters)

        candidates = await self.storage.query_blocks(filters)

        # ── Step 1b: Hybrid search — FTS results + vector similarity
        hybrid_boost: dict[str, float] = {}
        if (
            text_search
            and semantic
            and self._embedding_provider is not None
        ):
            hybrid_boost = await self._hybrid_search(
                text_search, candidates, max(limit * 5, 50),
            )
        normalized_hybrid = self._normalize_hybrid_scores(hybrid_boost)

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
                    if block and not block.deleted:
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

        # If hybrid found blocks not in FTS results, fetch + add them
        if hybrid_boost:
            candidate_ids = {b.id for b, _ in scored}
            missing_ids = [
                bid for bid in hybrid_boost if bid not in candidate_ids
            ]
            if missing_ids:
                fetched = await asyncio.gather(*[
                    self.storage.get_block(bid) for bid in missing_ids
                ])
                for block in fetched:
                    if block and not block.deleted:
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
            cosine_similarity,
            weighted_rrf_merge,
            unpack_embedding,
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

            # Python fallback
            all_embeddings = await self.storage.get_all_embeddings()
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
