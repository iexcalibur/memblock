"""Context builder — serialize relevant memory blocks into LLM-ready text."""

from __future__ import annotations

from dataclasses import dataclass, field
from memblock.block import Block
from memblock.decay import DecayEngine
from memblock.graph import GraphIndex
from memblock.query import QueryEngine
from memblock.storage.base import StorageAdapter
from typing import Any

from memblock.types import BlockType


# Rough token estimation: ~4 characters per token (conservative)
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length."""
    return len(text) // CHARS_PER_TOKEN


@dataclass
class ContextConfidence:
    """Metadata about retrieved context quality — helps detect unanswerable questions."""

    num_blocks: int = 0
    max_relevance: float = 0.0
    avg_relevance: float = 0.0
    sufficient_evidence: bool = True
    coverage_types: list[str] = field(default_factory=list)


class ContextBuilder:
    """
    Builds LLM-ready context from memory blocks within a token budget.

    Strategies:
    - relevance: Query by relevance, fill until budget
    - graph_walk: Start from most relevant block, walk graph outward
    - type_grouped: Group by block type (facts → preferences → events)
    - adaptive: Combines relevance + graph expansion + deduplication
    """

    def __init__(
        self,
        storage: StorageAdapter,
        query_engine: QueryEngine,
        graph: GraphIndex,
        decay: DecayEngine,
    ) -> None:
        self.storage = storage
        self.query = query_engine
        self.graph = graph
        self.decay = decay

    def build_context(
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
        """
        Build context text for LLM injection.

        Args:
            query: Search query to find relevant blocks
            token_budget: Maximum tokens in output
            strategy: 'relevance', 'graph_walk', 'type_grouped', or 'adaptive'
            include_metadata: Include confidence and source in output
            block_type: Filter by specific block type
            min_confidence: Minimum confidence threshold

        Returns:
            Formatted string ready for LLM context injection.
        """
        scope_kwargs = dict(
            session_id=session_id, org_id=org_id,
            project_id=project_id, agent_id=agent_id,
            metadata_filters=metadata_filters,
        )
        if strategy == "graph_walk":
            return self._strategy_graph_walk(
                query, token_budget, include_metadata, block_type, min_confidence,
                **scope_kwargs,
            )
        elif strategy == "type_grouped":
            return self._strategy_type_grouped(
                query, token_budget, include_metadata, min_confidence,
                **scope_kwargs,
            )
        elif strategy == "adaptive":
            return self._strategy_adaptive(
                query, token_budget, include_metadata, block_type, min_confidence,
                **scope_kwargs,
            )
        else:
            return self._strategy_relevance(
                query, token_budget, include_metadata, block_type, min_confidence,
                **scope_kwargs,
            )

    def build_context_with_confidence(
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
        """
        Build context with confidence metadata for adversarial robustness.

        Returns (context_string, ContextConfidence) tuple.
        When evidence is below threshold, the context header indicates low confidence,
        signaling to the LLM that the question may be unanswerable.
        """
        scope_kwargs = dict(
            session_id=session_id, org_id=org_id,
            project_id=project_id, agent_id=agent_id,
            metadata_filters=metadata_filters,
        )

        # Retrieve blocks to assess evidence quality
        blocks = self.query.query(
            text_search=query,
            type=block_type,
            min_confidence=min_confidence,
            sort_by="relevance",
            limit=50,
            **scope_kwargs,
        )

        # Compute confidence metrics
        confidence = ContextConfidence()
        if blocks:
            strengths = [self.decay.calculate_strength(b) for b in blocks]
            confidence.num_blocks = len(blocks)
            confidence.max_relevance = max(strengths)
            confidence.avg_relevance = sum(strengths) / len(strengths)
            confidence.coverage_types = list({b.type.value for b in blocks})
            confidence.sufficient_evidence = confidence.max_relevance >= evidence_threshold
        else:
            confidence.sufficient_evidence = False

        # Build context with appropriate header
        context = self.build_context(
            query=query,
            token_budget=token_budget,
            strategy=strategy,
            include_metadata=include_metadata,
            block_type=block_type,
            min_confidence=min_confidence,
            **scope_kwargs,
        )

        # Replace header with confidence-aware version
        if not confidence.sufficient_evidence and context:
            context = context.replace(
                "## Memory Context",
                "## Memory Context (Low confidence — information may not be available)",
                1,
            )

        return context, confidence

    def _strategy_relevance(
        self,
        query: str | None,
        token_budget: int,
        include_metadata: bool,
        block_type: BlockType | None,
        min_confidence: float,
        session_id: str | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        agent_id: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> str:
        """Fill context with most relevant blocks until budget is reached."""
        blocks = self.query.query(
            text_search=query,
            type=block_type,
            min_confidence=min_confidence,
            sort_by="relevance",
            limit=50,
            session_id=session_id,
            org_id=org_id,
            project_id=project_id,
            agent_id=agent_id,
            metadata_filters=metadata_filters,
        )

        return self._fill_budget(blocks, token_budget, include_metadata)

    def _strategy_graph_walk(
        self,
        query: str | None,
        token_budget: int,
        include_metadata: bool,
        block_type: BlockType | None,
        min_confidence: float,
        session_id: str | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        agent_id: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> str:
        """Start from most relevant block, walk graph outward."""
        seed_blocks = self.query.query(
            text_search=query,
            type=block_type,
            min_confidence=min_confidence,
            sort_by="relevance",
            limit=1,
            session_id=session_id,
            org_id=org_id,
            project_id=project_id,
            agent_id=agent_id,
            metadata_filters=metadata_filters,
        )

        if not seed_blocks:
            return ""

        seed = seed_blocks[0]

        # Walk graph from seed
        graph_blocks = self.graph.traverse(seed.id, max_depth=3)

        # Combine seed + graph neighbors, seed first, deduplicated
        all_blocks = self._deduplicate([seed] + graph_blocks)

        return self._fill_budget(all_blocks, token_budget, include_metadata)

    def _strategy_type_grouped(
        self,
        query: str | None,
        token_budget: int,
        include_metadata: bool,
        min_confidence: float,
        session_id: str | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        agent_id: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> str:
        """Group blocks by type: facts → preferences → events → entities."""
        type_order = [
            BlockType.FACT,
            BlockType.PREFERENCE,
            BlockType.EVENT,
            BlockType.ENTITY,
            BlockType.RELATION,
        ]

        all_blocks: list[Block] = []
        for bt in type_order:
            blocks = self.query.query(
                text_search=query,
                type=bt,
                min_confidence=min_confidence,
                sort_by="strength",
                limit=20,
                session_id=session_id,
                org_id=org_id,
                project_id=project_id,
                agent_id=agent_id,
                metadata_filters=metadata_filters,
            )
            all_blocks.extend(blocks)

        return self._fill_budget_grouped(all_blocks, token_budget, include_metadata)

    def _strategy_adaptive(
        self,
        query: str | None,
        token_budget: int,
        include_metadata: bool,
        block_type: BlockType | None,
        min_confidence: float,
        session_id: str | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        agent_id: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> str:
        """
        Adaptive context: relevance + 1-hop graph expansion + deduplication + type grouping.

        1. Retrieve top-K by relevance
        2. For each, pull 1-hop graph neighbors
        3. Deduplicate and score by direct_relevance + graph_support_count
        4. Group by type and fill token budget
        """
        scope = dict(
            session_id=session_id, org_id=org_id,
            project_id=project_id, agent_id=agent_id,
            metadata_filters=metadata_filters,
        )

        # Step 1: Get top relevant blocks
        seed_blocks = self.query.query(
            text_search=query,
            type=block_type,
            min_confidence=min_confidence,
            sort_by="relevance",
            limit=20,
            **scope,
        )

        if not seed_blocks:
            return ""

        # Step 2: Expand via 1-hop graph neighbors
        all_blocks: list[Block] = list(seed_blocks)
        seed_ids = {b.id for b in seed_blocks}
        support_count: dict[str, int] = {b.id: 0 for b in seed_blocks}

        for seed in seed_blocks:
            neighbors = self.graph.neighbors(seed.id)
            for neighbor in neighbors:
                if neighbor.id not in seed_ids:
                    all_blocks.append(neighbor)
                    seed_ids.add(neighbor.id)
                support_count[neighbor.id] = support_count.get(neighbor.id, 0) + 1

        # Step 3: Deduplicate
        all_blocks = self._deduplicate(all_blocks)

        # Step 4: Score and sort — prioritize blocks supported by multiple connections
        def adaptive_score(block: Block) -> float:
            strength = self.decay.calculate_strength(block)
            support = support_count.get(block.id, 0)
            return strength + (support * 0.1)

        all_blocks.sort(key=adaptive_score, reverse=True)

        # Step 5: Fill budget with relationship annotations
        return self._fill_budget(
            all_blocks, token_budget, include_metadata, annotate_relations=True,
        )

    def _deduplicate(self, blocks: list[Block]) -> list[Block]:
        """Remove duplicate blocks, keeping the first occurrence."""
        seen: set[str] = set()
        unique: list[Block] = []
        for b in blocks:
            if b.id not in seen:
                seen.add(b.id)
                unique.append(b)
        return unique

    def _fill_budget(
        self,
        blocks: list[Block],
        token_budget: int,
        include_metadata: bool,
        annotate_relations: bool = False,
    ) -> str:
        """Fill context with blocks until token budget is exhausted."""
        lines: list[str] = []
        tokens_used = 0

        # Header
        header = "## Memory Context"
        tokens_used += estimate_tokens(header)
        lines.append(header)

        for block in blocks:
            if annotate_relations:
                line = self._format_block_with_relations(block, include_metadata)
            else:
                line = self._format_block(block, include_metadata)
            line_tokens = estimate_tokens(line)

            if tokens_used + line_tokens > token_budget:
                break

            lines.append(line)
            tokens_used += line_tokens

        if len(lines) == 1:  # only header
            return ""

        return "\n".join(lines)

    def _fill_budget_grouped(
        self,
        blocks: list[Block],
        token_budget: int,
        include_metadata: bool,
    ) -> str:
        """Fill context with blocks grouped by type."""
        lines: list[str] = []
        tokens_used = 0
        current_type: BlockType | None = None

        header = "## Memory Context"
        tokens_used += estimate_tokens(header)
        lines.append(header)

        for block in blocks:
            # Add type header if changed
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
        """Format a single block for context output."""
        strength = self.decay.calculate_strength(block)

        if include_metadata:
            confidence_bar = self._confidence_bar(block.metadata.confidence)
            tags_str = f" [{', '.join(block.tags)}]" if block.tags else ""
            temporal_str = ""
            if block.metadata.happened_at:
                temporal_str = f" (when: {block.metadata.happened_at.strftime('%Y-%m-%d')})"
            return (
                f"- [{block.type.value.upper()}] {block.content}"
                f" {confidence_bar}"
                f" (source: {block.metadata.source.value}, strength: {strength:.2f})"
                f"{tags_str}{temporal_str}"
            )
        else:
            return f"- {block.content}"

    def _format_block_with_relations(self, block: Block, include_metadata: bool) -> str:
        """Format block with graph relationship annotations for richer context."""
        base = self._format_block(block, include_metadata)

        # Add up to 3 relationship annotations
        try:
            edges = self.storage.get_edges(block.id, direction="both")
            if edges:
                rel_parts = []
                for edge in edges[:3]:
                    target_id = edge.target_id if edge.source_id == block.id else edge.source_id
                    target = self.storage.get_block(target_id)
                    if target and not target.deleted:
                        # Truncate target content for brief annotation
                        brief = target.content[:50] + "..." if len(target.content) > 50 else target.content
                        rel_parts.append(f"{edge.relation.value}: \"{brief}\"")
                if rel_parts:
                    base += f" | related: {'; '.join(rel_parts)}"
        except Exception:
            pass  # relationship annotation failure should not break context

        return base

    def _confidence_bar(self, confidence: float) -> str:
        """Visual confidence indicator."""
        if confidence >= 0.9:
            return "[HIGH]"
        elif confidence >= 0.7:
            return "[MED]"
        elif confidence >= 0.4:
            return "[LOW]"
        else:
            return "[WEAK]"
