"""Context builder — serialize relevant memory blocks into LLM-ready text."""

from __future__ import annotations

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


class ContextBuilder:
    """
    Builds LLM-ready context from memory blocks within a token budget.

    Strategies:
    - relevance: Query by relevance, fill until budget
    - graph_walk: Start from most relevant block, walk graph outward
    - type_grouped: Group by block type (facts → preferences → events)
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
            strategy: 'relevance', 'graph_walk', or 'type_grouped'
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
        else:
            return self._strategy_relevance(
                query, token_budget, include_metadata, block_type, min_confidence,
                **scope_kwargs,
            )

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

        # Combine seed + graph neighbors, seed first
        all_blocks = [seed] + graph_blocks

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

    def _fill_budget(
        self,
        blocks: list[Block],
        token_budget: int,
        include_metadata: bool,
    ) -> str:
        """Fill context with blocks until token budget is exhausted."""
        lines: list[str] = []
        tokens_used = 0

        # Header
        header = "## Memory Context"
        tokens_used += estimate_tokens(header)
        lines.append(header)

        for block in blocks:
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
            return (
                f"- [{block.type.value.upper()}] {block.content}"
                f" {confidence_bar}"
                f" (source: {block.metadata.source.value}, strength: {strength:.2f})"
                f"{tags_str}"
            )
        else:
            return f"- {block.content}"

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
