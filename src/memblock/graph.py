"""Graph index — knowledge graph layer for block relationships."""

from __future__ import annotations

from collections import deque

from memblock.block import Block
from memblock.ops import OpLog
from memblock.schema import BlockSchema
from memblock.storage.base import StorageAdapter
from memblock.types import Edge, EdgeRelation, OpAction, generate_edge_id, now_utc


class GraphIndex:
    """
    Knowledge graph layer managing relationships between memory blocks.

    Provides linking, unlinking, traversal, neighbor queries,
    and contradiction detection.
    """

    def __init__(self, storage: StorageAdapter, op_log: OpLog) -> None:
        self.storage = storage
        self.op_log = op_log

    def link(
        self,
        source_id: str,
        target_id: str,
        relation: EdgeRelation = EdgeRelation.RELATED_TO,
        weight: float = 1.0,
    ) -> Edge:
        """
        Create a directed edge between two blocks.

        Validates edge constraints before creating.
        """
        source = self.storage.get_block(source_id)
        target = self.storage.get_block(target_id)

        if source is None:
            raise ValueError(f"Source block {source_id} not found")
        if target is None:
            raise ValueError(f"Target block {target_id} not found")

        # Validate edge relation
        BlockSchema.validate_edge(relation, source.type, target.type)

        edge = Edge(
            id=generate_edge_id(),
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            weight=weight,
            created_at=now_utc(),
        )

        self.storage.save_edge(edge)

        # Log operation
        self.op_log.append(
            action=OpAction.LINK,
            block_id=source_id,
            data={
                "target_id": target_id,
                "relation": relation.value,
                "weight": weight,
                "edge_id": edge.id,
            },
        )

        return edge

    def unlink(
        self,
        source_id: str,
        target_id: str,
        relation: EdgeRelation | None = None,
    ) -> int:
        """
        Remove edges between two blocks.

        If relation is specified, only removes edges with that relation.
        Returns number of edges removed.
        """
        edges = self.storage.get_edges(source_id, direction="outgoing")
        removed = 0

        for edge in edges:
            if edge.target_id == target_id:
                if relation is None or edge.relation == relation:
                    self.storage.delete_edge(edge.id)
                    self.op_log.append(
                        action=OpAction.UNLINK,
                        block_id=source_id,
                        data={
                            "target_id": target_id,
                            "relation": edge.relation.value,
                            "edge_id": edge.id,
                        },
                    )
                    removed += 1

        return removed

    def neighbors(
        self,
        block_id: str,
        relation: EdgeRelation | None = None,
        direction: str = "both",
    ) -> list[Block]:
        """
        Get all blocks directly connected to a given block.

        direction: 'outgoing', 'incoming', or 'both'
        """
        edges = self.storage.get_edges(block_id, direction=direction)
        neighbor_ids: set[str] = set()

        for edge in edges:
            if relation is not None and edge.relation != relation:
                continue
            if edge.source_id == block_id:
                neighbor_ids.add(edge.target_id)
            else:
                neighbor_ids.add(edge.source_id)

        blocks = []
        for nid in neighbor_ids:
            block = self.storage.get_block(nid)
            if block and not block.deleted:
                blocks.append(block)

        return blocks

    def traverse(
        self,
        start_id: str,
        max_depth: int = 3,
        relation: EdgeRelation | None = None,
    ) -> list[Block]:
        """
        BFS traversal from a starting block.

        Returns all reachable blocks within max_depth hops.
        Excludes the starting block from results.
        """
        visited: set[str] = {start_id}
        result: list[Block] = []
        queue: deque[tuple[str, int]] = deque([(start_id, 0)])

        while queue:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue

            edges = self.storage.get_edges(current_id)
            for edge in edges:
                if relation is not None and edge.relation != relation:
                    continue

                neighbor_id = (
                    edge.target_id if edge.source_id == current_id else edge.source_id
                )

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    block = self.storage.get_block(neighbor_id)
                    if block and not block.deleted:
                        result.append(block)
                        queue.append((neighbor_id, depth + 1))

        return result

    def traverse_with_depth(
        self,
        start_id: str,
        max_depth: int = 3,
        relation: EdgeRelation | None = None,
    ) -> dict[str, int]:
        """
        BFS traversal returning hop distances from the starting block.

        Returns dict of {block_id: hop_distance} for all reachable non-deleted blocks.
        Excludes the starting block itself.
        """
        visited: set[str] = {start_id}
        result: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque([(start_id, 0)])

        while queue:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue

            edges = self.storage.get_edges(current_id)
            for edge in edges:
                if relation is not None and edge.relation != relation:
                    continue

                neighbor_id = (
                    edge.target_id if edge.source_id == current_id else edge.source_id
                )

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    block = self.storage.get_block(neighbor_id)
                    if block and not block.deleted:
                        result[neighbor_id] = depth + 1
                        queue.append((neighbor_id, depth + 1))

        return result

    def shortest_path(self, from_id: str, to_id: str) -> list[Block]:
        """
        Find the shortest path between two blocks (BFS).

        Returns the path as a list of blocks (including start and end).
        Returns empty list if no path exists.
        """
        if from_id == to_id:
            block = self.storage.get_block(from_id)
            return [block] if block else []

        visited: set[str] = {from_id}
        # parent map for path reconstruction
        parent: dict[str, str] = {}
        queue: deque[str] = deque([from_id])

        while queue:
            current_id = queue.popleft()
            edges = self.storage.get_edges(current_id)

            for edge in edges:
                neighbor_id = (
                    edge.target_id if edge.source_id == current_id else edge.source_id
                )

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    parent[neighbor_id] = current_id

                    if neighbor_id == to_id:
                        # Reconstruct path
                        path_ids = [to_id]
                        node = to_id
                        while node in parent:
                            node = parent[node]
                            path_ids.append(node)
                        path_ids.reverse()

                        path = []
                        for pid in path_ids:
                            block = self.storage.get_block(pid)
                            if block:
                                path.append(block)
                        return path

                    queue.append(neighbor_id)

        return []  # no path found

    def find_contradictions(self, block_id: str) -> list[Block]:
        """Find all blocks that contradict the given block."""
        return self.neighbors(block_id, relation=EdgeRelation.CONTRADICTS)

    def get_cluster(self, block_id: str) -> list[Block]:
        """Get all blocks connected to this block (full connected component)."""
        return self.traverse(block_id, max_depth=100)

    def get_edges_between(self, source_id: str, target_id: str) -> list[Edge]:
        """Get all edges between two specific blocks."""
        edges = self.storage.get_edges(source_id, direction="outgoing")
        return [e for e in edges if e.target_id == target_id]
