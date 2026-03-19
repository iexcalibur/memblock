"""Multi-hop iterative retriever — improves recall for complex reasoning queries.

Performs multiple retrieval passes:
1. Standard hybrid search for initial results
2. Entity extraction + focused queries for each entity
3. Graph walk from all retrieved blocks with proximity boosting
4. Optional LLM sufficiency gating between hops
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from memblock.block import Block
from memblock.decay import DecayEngine
from memblock.graph import GraphIndex
from memblock.query import QueryEngine
from memblock.storage.base import StorageAdapter
from memblock.types import BlockType


@dataclass
class MultiHopResult:
    """Result of a multi-hop retrieval."""

    blocks: list[Block] = field(default_factory=list)
    hops_used: int = 0
    entities_found: list[str] = field(default_factory=list)
    graph_blocks_added: int = 0


class MultiHopRetriever:
    """
    Iterative retriever for multi-hop reasoning questions.

    Performs multiple retrieval passes, extracting entities from initial results
    and walking the knowledge graph to find connected information.
    """

    def __init__(
        self,
        query_engine: QueryEngine,
        graph: GraphIndex,
        storage: StorageAdapter,
        decay: DecayEngine,
        max_hops: int = 3,
    ) -> None:
        self._query = query_engine
        self._graph = graph
        self._storage = storage
        self._decay = decay
        self._max_hops = max_hops

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        session_id: str | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        agent_id: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> MultiHopResult:
        """
        Multi-hop retrieval with entity extraction and graph walking.

        Hop 1: Standard hybrid search -> top-K initial results
        Hop 2: Extract entities from hop-1, focused query for each entity
        Hop 3: Walk graph from all retrieved blocks, union with proximity boost
        """
        result = MultiHopResult()
        seen_ids: set[str] = set()
        all_scored: list[tuple[Block, float]] = []
        scope = dict(
            session_id=session_id, org_id=org_id,
            project_id=project_id, agent_id=agent_id,
            metadata_filters=metadata_filters,
        )

        # Hop 1: Standard retrieval
        hop1_blocks = self._query.query(
            text_search=query,
            sort_by="relevance",
            limit=limit * 2,
            **scope,
        )
        result.hops_used = 1

        for block in hop1_blocks:
            if block.id not in seen_ids:
                seen_ids.add(block.id)
                strength = self._decay.calculate_strength(block)
                all_scored.append((block, strength + 1.0))  # bonus for direct match

        if not hop1_blocks:
            result.blocks = []
            return result

        # Hop 2: Extract entities from hop-1 results and query for each
        entities = self._extract_entities(hop1_blocks)
        result.entities_found = entities
        result.hops_used = 2

        for entity in entities[:5]:  # limit entity queries to prevent explosion
            entity_blocks = self._query.query(
                text_search=entity,
                sort_by="relevance",
                limit=limit,
                **scope,
            )
            for block in entity_blocks:
                if block.id not in seen_ids:
                    seen_ids.add(block.id)
                    strength = self._decay.calculate_strength(block)
                    all_scored.append((block, strength + 0.5))  # lower bonus for entity match

        # Hop 3: Walk graph from all retrieved blocks
        result.hops_used = 3
        graph_candidates: dict[str, int] = {}  # block_id -> min_depth

        for block, _ in all_scored:
            depth_map = self._graph.traverse_with_depth(block.id, max_depth=2)
            for neighbor_id, depth in depth_map.items():
                if neighbor_id not in seen_ids:
                    if neighbor_id not in graph_candidates or depth < graph_candidates[neighbor_id]:
                        graph_candidates[neighbor_id] = depth

        for block_id, depth in graph_candidates.items():
            block = self._storage.get_block(block_id)
            if block and not block.deleted:
                seen_ids.add(block_id)
                strength = self._decay.calculate_strength(block)
                proximity_bonus = 0.3 / depth
                all_scored.append((block, strength + proximity_bonus))
                result.graph_blocks_added += 1

        # Sort by score and return top-limit
        all_scored.sort(key=lambda x: x[1], reverse=True)
        result.blocks = [block for block, _ in all_scored[:limit]]
        return result

    def _extract_entities(self, blocks: list[Block]) -> list[str]:
        """Extract entity names from block contents using simple NLP heuristics."""
        entities: list[str] = []
        seen: set[str] = set()

        for block in blocks:
            # Extract capitalized words/phrases (likely proper nouns)
            caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", block.content)
            for cap in caps:
                lower = cap.lower()
                if lower not in seen and len(cap) > 2:
                    seen.add(lower)
                    entities.append(cap)

            # Extract entity-type blocks' content as entity names
            if block.type == BlockType.ENTITY:
                name = block.content.split(".")[0].strip()  # first sentence
                lower = name.lower()
                if lower not in seen:
                    seen.add(lower)
                    entities.append(name)

            # Extract quoted strings
            quoted = re.findall(r'"([^"]+)"', block.content)
            for q in quoted:
                lower = q.lower()
                if lower not in seen and len(q) > 2:
                    seen.add(lower)
                    entities.append(q)

        return entities
