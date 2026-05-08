"""Memory decay engine — automatic memory fading based on access patterns."""

from __future__ import annotations

import math
from datetime import datetime, timezone

from memblock.block import Block
from memblock.storage.base import StorageAdapter
from memblock.types import BlockType, now_utc


# Per-type decay defaults — different block types fade at different
# rates because they have different durability semantics:
#   ENTITY      — named anchors (funds, people, places). Stable;
#                 should almost never fade.
#   FACT        — objective info ("user is 35"). Durable; slow decay.
#   RELATION    — describes a relationship between entities. Same
#                 durability profile as FACT.
#   PREFERENCE  — opinions ("user prefers index funds"). People change
#                 their minds; faster decay.
#   EVENT       — point-in-time happenings ("user asked X yesterday").
#                 Bounded relevance; fastest decay.
#
# Callers can still override per-block via `store(decay_rate=...)`.
# These tables only kick in when the caller leaves it as the default.
DEFAULT_DECAY_BY_TYPE: dict[BlockType, float] = {
    BlockType.ENTITY:     0.001,
    BlockType.FACT:       0.005,
    BlockType.RELATION:   0.005,
    BlockType.PREFERENCE: 0.020,
    BlockType.EVENT:      0.040,
}


def default_decay_rate_for(block_type: BlockType | str) -> float:
    """Return the per-type default decay rate. Falls back to FACT's
    rate (0.005) for any unknown / non-enum input."""
    if isinstance(block_type, str):
        try:
            block_type = BlockType(block_type)
        except ValueError:
            return DEFAULT_DECAY_BY_TYPE[BlockType.FACT]
    return DEFAULT_DECAY_BY_TYPE.get(
        block_type, DEFAULT_DECAY_BY_TYPE[BlockType.FACT],
    )


class DecayEngine:
    """
    Calculates memory strength and manages automatic decay.

    Strength formula:
        strength = base_confidence * decay_factor * access_boost

    Where:
        decay_factor = e^(-decay_rate * hours_since_creation)
        access_boost = 1 + log(1 + access_count) * recency_factor
        recency_factor = e^(-0.01 * hours_since_last_access)

    Memories that are frequently accessed stay strong.
    Memories that are never accessed fade over time.
    """

    def __init__(self, storage: StorageAdapter) -> None:
        self.storage = storage

    def calculate_strength(self, block: Block, at_time: datetime | None = None) -> float:
        """
        Calculate the current strength of a memory block.

        Returns a float between 0.0 and ~1.0 (can slightly exceed 1.0 with high access).
        """
        if at_time is None:
            at_time = now_utc()

        meta = block.metadata

        # Hours since creation
        hours_alive = max(
            (at_time - meta.created_at).total_seconds() / 3600.0,
            0.0,
        )

        # Base decay: exponential decay over time
        decay_factor = math.exp(-meta.decay_rate * hours_alive)

        # Access boost: frequent access strengthens memory
        access_boost = 1.0
        if meta.access_count > 0:
            # Logarithmic access boost (diminishing returns)
            log_access = math.log(1 + meta.access_count)

            # Recency of last access matters
            recency_factor = 1.0
            if meta.last_accessed:
                hours_since_access = max(
                    (at_time - meta.last_accessed).total_seconds() / 3600.0,
                    0.0,
                )
                recency_factor = math.exp(-0.01 * hours_since_access)

            access_boost = 1.0 + (log_access * recency_factor * 0.3)

        strength = meta.confidence * decay_factor * access_boost

        # Clamp to [0, 1] for consistency
        return max(0.0, min(1.0, strength))

    def apply_decay(self, blocks: list[Block] | None = None) -> list[tuple[Block, float]]:
        """
        Calculate strength for all blocks (or provided list).

        Returns list of (block, strength) tuples, sorted by strength descending.
        Does NOT modify blocks in storage — use prune() for that.
        """
        if blocks is None:
            blocks = self.storage.get_all_blocks()

        results = []
        for block in blocks:
            strength = self.calculate_strength(block)
            results.append((block, strength))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def prune(self, min_strength: float = 0.1) -> list[Block]:
        """
        Soft-delete blocks whose strength has fallen below the threshold.

        Also removes blocks with expired TTL.
        Returns list of pruned blocks.
        """
        blocks = self.storage.get_all_blocks()
        current_time = now_utc()
        pruned: list[Block] = []

        for block in blocks:
            should_prune = False

            # Check TTL expiry
            if block.metadata.ttl is not None:
                age_seconds = (current_time - block.metadata.created_at).total_seconds()
                if age_seconds > block.metadata.ttl:
                    should_prune = True

            # Check strength decay
            if not should_prune:
                strength = self.calculate_strength(block, at_time=current_time)
                if strength < min_strength:
                    should_prune = True

            if should_prune:
                self.storage.update_block(block.id, {"deleted": True})
                pruned.append(block)

        return pruned

    def touch(self, block_id: str) -> None:
        """
        Touch a block — increment access count and update last_accessed.

        This strengthens the memory, counteracting decay.
        """
        block = self.storage.get_block(block_id)
        if block is None:
            return

        block.metadata.access_count += 1
        block.metadata.last_accessed = now_utc()

        self.storage.update_block(block_id, {
            "access_count": block.metadata.access_count,
            "last_accessed": block.metadata.last_accessed,
        })

    def get_strongest(self, limit: int = 10) -> list[tuple[Block, float]]:
        """Get the strongest memories, sorted by strength."""
        results = self.apply_decay()
        return results[:limit]

    def get_weakest(self, limit: int = 10) -> list[tuple[Block, float]]:
        """Get the weakest memories (candidates for pruning)."""
        results = self.apply_decay()
        results.reverse()
        return results[:limit]
