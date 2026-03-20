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

        Hop 0: Extract entities from the QUESTION itself for focused queries
        Hop 1: Standard hybrid search -> top-K initial results
        Hop 2: Extract entities from hop-1 results + question entities, query each
        Hop 3: Walk graph from all retrieved blocks with deeper traversal
        """
        result = MultiHopResult()
        seen_ids: set[str] = set()
        all_scored: list[tuple[Block, float]] = []
        scope = dict(
            session_id=session_id, org_id=org_id,
            project_id=project_id, agent_id=agent_id,
            metadata_filters=metadata_filters,
        )

        # Hop 0: Extract entities from the question itself
        question_entities = self._extract_entities_from_text(query)

        # Hop 1: Standard retrieval (broader initial search)
        hop1_blocks = self._query.query(
            text_search=query,
            sort_by="relevance",
            limit=limit * 3,  # broader initial pool
            **scope,
        )
        result.hops_used = 1

        for block in hop1_blocks:
            if block.id not in seen_ids:
                seen_ids.add(block.id)
                strength = self._decay.calculate_strength(block)
                all_scored.append((block, strength + 1.0))  # bonus for direct match

        if not hop1_blocks:
            # Even with no hop1 results, try question entities
            if question_entities:
                for entity in question_entities[:3]:
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
                            all_scored.append((block, strength + 0.8))
            if not all_scored:
                result.blocks = []
                return result

        # Hop 2: Extract entities from hop-1 results + question, query for each
        block_entities = self._extract_entities(hop1_blocks)
        # Merge question entities (prioritized) with block entities
        all_entities = list(dict.fromkeys(question_entities + block_entities))
        result.entities_found = all_entities
        result.hops_used = 2

        for entity in all_entities[:10]:  # increased from 5 to 10 for better coverage
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
                    # Higher bonus for question entities vs block entities
                    bonus = 0.8 if entity in question_entities else 0.5
                    all_scored.append((block, strength + bonus))

        # Hop 3: Walk graph from all retrieved blocks (deeper traversal)
        result.hops_used = 3
        graph_candidates: dict[str, int] = {}  # block_id -> min_depth

        for block, _ in all_scored:
            depth_map = self._graph.traverse_with_depth(block.id, max_depth=3)  # deeper walk
            for neighbor_id, depth in depth_map.items():
                if neighbor_id not in seen_ids:
                    if neighbor_id not in graph_candidates or depth < graph_candidates[neighbor_id]:
                        graph_candidates[neighbor_id] = depth

        for block_id, depth in graph_candidates.items():
            block = self._storage.get_block(block_id)
            if block and not block.deleted:
                seen_ids.add(block_id)
                strength = self._decay.calculate_strength(block)
                proximity_bonus = 0.4 / depth  # increased from 0.3
                all_scored.append((block, strength + proximity_bonus))
                result.graph_blocks_added += 1

        # Sort by score and return top-limit
        all_scored.sort(key=lambda x: x[1], reverse=True)
        result.blocks = [block for block, _ in all_scored[:limit]]
        return result

    # Common words to skip during entity extraction
    _STOP_WORDS = frozenset({
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "had", "her", "was", "one", "our", "out", "has", "his", "how",
        "its", "may", "new", "now", "old", "see", "way", "who", "did",
        "get", "let", "say", "she", "too", "use", "yes", "yet", "hey",
        "said", "also", "been", "call", "come", "each", "from", "have",
        "here", "just", "like", "made", "make", "much", "must", "name",
        "only", "over", "such", "take", "than", "that", "them", "then",
        "they", "this", "very", "when", "what", "with", "will", "well",
        "your", "about", "after", "could", "every", "first", "found",
        "great", "house", "large", "later", "never", "other", "place",
        "right", "shall", "small", "some", "still", "their", "there",
        "these", "thing", "think", "those", "three", "under", "where",
        "which", "while", "world", "would", "yeah", "okay", "sure",
        "really", "actually", "session", "user", "speaker", "unknown",
        "date", "time", "today", "yesterday", "tomorrow", "monday",
        "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "january", "february", "march", "april", "june", "july",
        "august", "september", "october", "november", "december",
        "shared", "image", "said", "told", "asked", "went", "going",
        "think", "know", "want", "need", "something", "anything",
        "everything", "nothing", "someone", "anyone", "everyone",
    })

    def _extract_entities_from_text(self, text: str) -> list[str]:
        """Extract entity names directly from a text string (e.g., a question)."""
        entities: list[str] = []
        seen: set[str] = set()

        def _add(t: str) -> None:
            lower = t.strip().lower()
            if lower not in seen and len(t.strip()) > 2 and lower not in self._STOP_WORDS:
                seen.add(lower)
                entities.append(t.strip())

        # Multi-word proper nouns
        for cap in re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text):
            _add(cap)

        # Single capitalized words
        for cap in re.findall(r"\b([A-Z][a-z]{2,})\b", text):
            _add(cap)

        # Quoted strings
        for q in re.findall(r'"([^"]{3,60})"', text):
            _add(q)

        # Key noun phrases from the question (words after question words)
        for phrase in re.findall(
            r"(?:does|did|is|was|would|could|has|have)\s+(\w+(?:\s+\w+){0,2})",
            text, re.IGNORECASE
        ):
            words = phrase.split()
            for w in words:
                if w[0].isupper() and len(w) > 2 and w.lower() not in self._STOP_WORDS:
                    _add(w)

        return entities

    def _extract_entities(self, blocks: list[Block]) -> list[str]:
        """
        Extract entity names from block contents using improved NLP heuristics.

        Extracts:
        - Multi-word proper nouns (e.g., "Olive Garden", "New York")
        - Single capitalized words that aren't stop words
        - Entity-type block content
        - Quoted strings (conversation content)
        - Key phrases after common patterns (e.g., "named X", "called X", "at X")
        """
        entities: list[str] = []
        seen: set[str] = set()

        def _add(text: str) -> None:
            lower = text.strip().lower()
            if lower not in seen and len(text.strip()) > 2 and lower not in self._STOP_WORDS:
                seen.add(lower)
                entities.append(text.strip())

        for block in blocks:
            content = block.content

            # Extract multi-word capitalized phrases (proper nouns like "New York", "Olive Garden")
            multi_caps = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", content)
            for cap in multi_caps:
                _add(cap)

            # Extract single capitalized words (filter stop words)
            single_caps = re.findall(r"\b([A-Z][a-z]{2,})\b", content)
            for cap in single_caps:
                _add(cap)

            # Extract entity-type blocks' content as entity names
            if block.type == BlockType.ENTITY:
                name = content.split(".")[0].strip()
                _add(name)

            # Extract quoted strings (often important conversation content)
            quoted = re.findall(r'"([^"]{3,80})"', content)
            for q in quoted:
                # For long quotes, extract key noun phrases instead
                if len(q) > 40:
                    inner_caps = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", q)
                    for ic in inner_caps:
                        _add(ic)
                else:
                    _add(q)

            # Extract patterns like "named X", "called X", "at X", "to X"
            named_patterns = re.findall(
                r"(?:named|called|at|to|from|with|about|visiting|visited)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                content
            )
            for np in named_patterns:
                _add(np)

            # Extract patterns with possessives: "Mike's", "Sarah's"
            possessives = re.findall(r"\b([A-Z][a-z]+)'s\b", content)
            for p in possessives:
                _add(p)

        return entities
