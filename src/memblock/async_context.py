"""Async context builder — async equivalent of `context.ContextBuilder`.

Mirrors `ContextBuilder` 1:1 but every storage / query call is
awaited through an `AsyncQueryEngine` + `AsyncStorageAdapter`.
Pure-Python formatting / token-budget logic is reused via local
imports.

Used by `AsyncMemBlock.build_context()` when constructed with a
`postgresql+asyncpg://` URL.
"""

from __future__ import annotations

import asyncio
from typing import Any

from memblock.async_query import AsyncQueryEngine
from memblock.block import Block
from memblock.context import (
    ContextConfidence,
    estimate_tokens,
)
from memblock.decay import DecayEngine
from memblock.storage.async_base import AsyncStorageAdapter
from memblock.types import BlockType


class AsyncContextBuilder:
    """Async-native context builder.

    Same `build_context` signature, same strategies (relevance,
    graph_walk, type_grouped, adaptive), same output format. The
    difference is async I/O throughout; formatting is reused.
    """

    def __init__(
        self,
        storage: AsyncStorageAdapter,
        query_engine: AsyncQueryEngine,
        decay: DecayEngine | None = None,
    ) -> None:
        self.storage = storage
        self.query = query_engine
        self.decay = decay or DecayEngine(storage=None)  # type: ignore[arg-type]

    async def build_context(
        self,
        query: str | None = None,
        token_budget: int = 4000,
        strategy: str = "relevance",
        include_metadata: bool = True,
        block_type: BlockType | None = None,
        min_confidence: float = 0.0,
        session_id: str | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        agent_id: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> str:
        """Build context text for LLM injection. See
        `ContextBuilder.build_context` for full param semantics."""

        scope = dict(
            session_id=session_id, org_id=org_id,
            project_id=project_id, agent_id=agent_id,
            metadata_filters=metadata_filters,
        )

        if strategy == "graph_walk":
            return await self._strategy_graph_walk(
                query, token_budget, include_metadata, block_type,
                min_confidence, **scope,
            )
        if strategy == "type_grouped":
            return await self._strategy_type_grouped(
                query, token_budget, include_metadata, min_confidence, **scope,
            )
        if strategy == "adaptive":
            return await self._strategy_adaptive(
                query, token_budget, include_metadata, block_type,
                min_confidence, **scope,
            )
        return await self._strategy_relevance(
            query, token_budget, include_metadata, block_type,
            min_confidence, **scope,
        )

    async def build_context_with_confidence(
        self,
        query: str | None = None,
        token_budget: int = 4000,
        strategy: str = "relevance",
        include_metadata: bool = True,
        block_type: BlockType | None = None,
        min_confidence: float = 0.0,
        evidence_threshold: float = 0.3,
        session_id: str | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        agent_id: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> tuple[str, ContextConfidence]:
        """Async equivalent of
        `ContextBuilder.build_context_with_confidence`."""
        scope = dict(
            session_id=session_id, org_id=org_id,
            project_id=project_id, agent_id=agent_id,
            metadata_filters=metadata_filters,
        )

        blocks = await self.query.query(
            text_search=query, type=block_type,
            min_confidence=min_confidence, sort_by="relevance",
            limit=50, **scope,
        )

        confidence = ContextConfidence()
        if blocks:
            strengths = [self.decay.calculate_strength(b) for b in blocks]
            confidence.num_blocks = len(blocks)
            confidence.max_relevance = max(strengths)
            confidence.avg_relevance = sum(strengths) / len(strengths)
            confidence.coverage_types = list({b.type.value for b in blocks})
            confidence.sufficient_evidence = (
                confidence.max_relevance >= evidence_threshold
            )
        else:
            confidence.sufficient_evidence = False

        context = await self.build_context(
            query=query, token_budget=token_budget, strategy=strategy,
            include_metadata=include_metadata, block_type=block_type,
            min_confidence=min_confidence, **scope,
        )

        if not confidence.sufficient_evidence and context:
            context = context.replace(
                "## Memory Context",
                "## Memory Context (Low confidence — information may not be available)",
                1,
            )
        return context, confidence

    # ─── Strategies ──────────────────────────────────────────────────

    async def _strategy_relevance(
        self,
        query: str | None,
        token_budget: int,
        include_metadata: bool,
        block_type: BlockType | None,
        min_confidence: float,
        **scope: Any,
    ) -> str:
        blocks = await self.query.query(
            text_search=query, type=block_type,
            min_confidence=min_confidence, sort_by="relevance",
            limit=50, **scope,
        )
        return await self._fill_budget(blocks, token_budget, include_metadata)

    async def _strategy_graph_walk(
        self,
        query: str | None,
        token_budget: int,
        include_metadata: bool,
        block_type: BlockType | None,
        min_confidence: float,
        **scope: Any,
    ) -> str:
        seed_blocks = await self.query.query(
            text_search=query, type=block_type,
            min_confidence=min_confidence, sort_by="relevance",
            limit=1, **scope,
        )
        if not seed_blocks:
            return ""

        seed = seed_blocks[0]
        graph_blocks = await self._async_traverse(seed.id, max_depth=3)
        all_blocks = self._deduplicate([seed] + graph_blocks)
        return await self._fill_budget(
            all_blocks, token_budget, include_metadata,
        )

    async def _strategy_type_grouped(
        self,
        query: str | None,
        token_budget: int,
        include_metadata: bool,
        min_confidence: float,
        **scope: Any,
    ) -> str:
        type_order = [
            BlockType.FACT, BlockType.PREFERENCE, BlockType.EVENT,
            BlockType.ENTITY, BlockType.RELATION,
        ]
        # Concurrent per-type queries — significantly faster than
        # sequential sync.
        results = await asyncio.gather(*[
            self.query.query(
                text_search=query, type=bt,
                min_confidence=min_confidence, sort_by="strength",
                limit=20, **scope,
            )
            for bt in type_order
        ])
        all_blocks: list[Block] = []
        for r in results:
            all_blocks.extend(r)
        return self._fill_budget_grouped(
            all_blocks, token_budget, include_metadata,
        )

    async def _strategy_adaptive(
        self,
        query: str | None,
        token_budget: int,
        include_metadata: bool,
        block_type: BlockType | None,
        min_confidence: float,
        **scope: Any,
    ) -> str:
        seed_blocks = await self.query.query(
            text_search=query, type=block_type,
            min_confidence=min_confidence, sort_by="relevance",
            limit=20, **scope,
        )
        if not seed_blocks:
            return ""

        all_blocks: list[Block] = list(seed_blocks)
        seed_ids = {b.id for b in seed_blocks}
        support_count: dict[str, int] = {b.id: 0 for b in seed_blocks}

        # 1-hop expand: gather neighbor edges concurrently per seed.
        neighbor_groups = await asyncio.gather(*[
            self._async_neighbors(b.id) for b in seed_blocks
        ])
        for seed, neighbors in zip(seed_blocks, neighbor_groups):
            for neighbor in neighbors:
                if neighbor.id not in seed_ids:
                    all_blocks.append(neighbor)
                    seed_ids.add(neighbor.id)
                support_count[neighbor.id] = support_count.get(neighbor.id, 0) + 1

        all_blocks = self._deduplicate(all_blocks)

        def adaptive_score(block: Block) -> float:
            strength = self.decay.calculate_strength(block)
            support = support_count.get(block.id, 0)
            return strength + support * 0.1

        all_blocks.sort(key=adaptive_score, reverse=True)
        return await self._fill_budget(
            all_blocks, token_budget, include_metadata, annotate_relations=True,
        )

    # ─── Graph helpers ───────────────────────────────────────────────

    async def _async_traverse(
        self, seed_id: str, max_depth: int = 3,
    ) -> list[Block]:
        """BFS from `seed_id`, return reachable block dataclasses
        within `max_depth`. Concurrent edge fetches per depth level."""
        visited: set[str] = {seed_id}
        frontier = [seed_id]
        result_ids: list[str] = []

        for _ in range(max_depth):
            if not frontier:
                break
            edge_lists = await asyncio.gather(*[
                self.storage.get_edges(bid, direction="both")
                for bid in frontier
            ])
            next_frontier: list[str] = []
            for edges in edge_lists:
                for edge in edges:
                    for nid in (edge.source_id, edge.target_id):
                        if nid not in visited:
                            visited.add(nid)
                            next_frontier.append(nid)
                            result_ids.append(nid)
            frontier = next_frontier

        if not result_ids:
            return []
        # Concurrent block fetch
        blocks = await asyncio.gather(*[
            self.storage.get_block(bid) for bid in result_ids
        ])
        return [b for b in blocks if b is not None and not b.deleted]

    async def _async_neighbors(self, block_id: str) -> list[Block]:
        """Return blocks one hop from `block_id`."""
        edges = await self.storage.get_edges(block_id, direction="both")
        neighbor_ids = {
            edge.target_id if edge.source_id == block_id else edge.source_id
            for edge in edges
        }
        if not neighbor_ids:
            return []
        blocks = await asyncio.gather(*[
            self.storage.get_block(nid) for nid in neighbor_ids
        ])
        return [b for b in blocks if b is not None and not b.deleted]

    # ─── Pure-Python formatting (reused unchanged) ───────────────────

    def _deduplicate(self, blocks: list[Block]) -> list[Block]:
        seen: set[str] = set()
        unique: list[Block] = []
        for b in blocks:
            if b.id not in seen:
                seen.add(b.id)
                unique.append(b)
        return unique

    async def _fill_budget(
        self,
        blocks: list[Block],
        token_budget: int,
        include_metadata: bool,
        annotate_relations: bool = False,
    ) -> str:
        """Same logic as `ContextBuilder._fill_budget`. Async because
        `annotate_relations` mode awaits per-block edge fetches."""
        lines: list[str] = ["## Memory Context"]
        tokens_used = estimate_tokens(lines[0])

        for block in blocks:
            if annotate_relations:
                line = await self._format_block_with_relations(
                    block, include_metadata,
                )
            else:
                line = self._format_block(block, include_metadata)
            line_tokens = estimate_tokens(line)
            if tokens_used + line_tokens > token_budget:
                break
            lines.append(line)
            tokens_used += line_tokens

        if len(lines) == 1:
            return ""
        return "\n".join(lines)

    def _fill_budget_grouped(
        self,
        blocks: list[Block],
        token_budget: int,
        include_metadata: bool,
    ) -> str:
        """Pure Python — identical to sync version."""
        lines: list[str] = ["## Memory Context"]
        tokens_used = estimate_tokens(lines[0])
        current_type: BlockType | None = None

        for block in blocks:
            if block.type != current_type:
                type_header = f"\n### {block.type.value.title()}s"
                type_tokens = estimate_tokens(type_header)
                if tokens_used + type_tokens > token_budget:
                    break
                lines.append(type_header)
                tokens_used += type_tokens
                current_type = block.type

            line = self._format_block(block, include_metadata)
            line_tokens = estimate_tokens(line)
            if tokens_used + line_tokens > token_budget:
                break
            lines.append(line)
            tokens_used += line_tokens

        if len(lines) == 1:
            return ""
        return "\n".join(lines)

    def _format_block(self, block: Block, include_metadata: bool) -> str:
        strength = self.decay.calculate_strength(block)
        if include_metadata:
            confidence_bar = self._confidence_bar(block.metadata.confidence)
            tags_str = f" [{', '.join(block.tags)}]" if block.tags else ""
            temporal_str = ""
            if block.metadata.happened_at:
                temporal_str = (
                    f" (when: {block.metadata.happened_at.strftime('%Y-%m-%d')})"
                )
            return (
                f"- [{block.type.value.upper()}] {block.content}"
                f" {confidence_bar}"
                f" (source: {block.metadata.source.value}, strength: {strength:.2f})"
                f"{tags_str}{temporal_str}"
            )
        return f"- {block.content}"

    async def _format_block_with_relations(
        self, block: Block, include_metadata: bool,
    ) -> str:
        """Annotate a block with up to 3 graph relationships.
        Same shape as sync; awaits the edge + neighbor fetches."""
        base = self._format_block(block, include_metadata)
        try:
            edges = await self.storage.get_edges(block.id, direction="both")
            if not edges:
                return base
            rel_parts: list[str] = []
            target_ids = []
            for edge in edges[:3]:
                target_id = (
                    edge.target_id if edge.source_id == block.id
                    else edge.source_id
                )
                target_ids.append((edge, target_id))
            # Concurrent neighbor fetch
            target_blocks = await asyncio.gather(*[
                self.storage.get_block(tid) for _, tid in target_ids
            ])
            for (edge, _), target in zip(target_ids, target_blocks):
                if target and not target.deleted:
                    brief = (
                        target.content[:50] + "..."
                        if len(target.content) > 50 else target.content
                    )
                    rel_parts.append(f'{edge.relation.value}: "{brief}"')
            if rel_parts:
                base += f" | related: {'; '.join(rel_parts)}"
        except Exception:
            pass
        return base

    def _confidence_bar(self, confidence: float) -> str:
        if confidence >= 0.9:
            return "[HIGH]"
        if confidence >= 0.7:
            return "[MED]"
        if confidence >= 0.4:
            return "[LOW]"
        return "[WEAK]"
