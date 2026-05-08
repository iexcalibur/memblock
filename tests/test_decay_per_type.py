"""Tests for v0.11.0 per-type decay defaults."""

from __future__ import annotations

import pytest

from memblock.decay import (
    DEFAULT_DECAY_BY_TYPE,
    default_decay_rate_for,
)
from memblock.types import BlockType


class TestPerTypeDecayDefaults:
    def test_entity_decays_slowest(self):
        """ENTITY blocks (named anchors) should be the most durable."""
        rates = list(DEFAULT_DECAY_BY_TYPE.values())
        assert default_decay_rate_for(BlockType.ENTITY) == min(rates)

    def test_event_decays_fastest(self):
        """EVENT blocks (point-in-time) should fade quickest."""
        rates = list(DEFAULT_DECAY_BY_TYPE.values())
        assert default_decay_rate_for(BlockType.EVENT) == max(rates)

    def test_fact_durable_but_not_eternal(self):
        """FACTs are objective info, durable but not as much as anchors."""
        fact_rate = default_decay_rate_for(BlockType.FACT)
        assert default_decay_rate_for(BlockType.ENTITY) < fact_rate
        assert fact_rate < default_decay_rate_for(BlockType.PREFERENCE)

    def test_preference_decays_faster_than_fact(self):
        """People change minds; preferences should fade faster than facts."""
        assert (
            default_decay_rate_for(BlockType.PREFERENCE)
            > default_decay_rate_for(BlockType.FACT)
        )

    def test_relation_matches_fact_durability(self):
        """RELATIONs describe stable entity-entity links."""
        assert (
            default_decay_rate_for(BlockType.RELATION)
            == default_decay_rate_for(BlockType.FACT)
        )

    def test_string_input_resolves(self):
        """Accepts the string form of the enum value."""
        assert default_decay_rate_for("entity") == default_decay_rate_for(BlockType.ENTITY)
        assert default_decay_rate_for("fact")   == default_decay_rate_for(BlockType.FACT)

    def test_unknown_string_falls_back_to_fact_rate(self):
        """Unknown type → FACT rate (safe middle-ground)."""
        assert (
            default_decay_rate_for("unknown_block_type")
            == default_decay_rate_for(BlockType.FACT)
        )

    def test_concrete_values(self):
        """Pin the actual numbers — changing these is intentional, so
        a test failure is the right signal."""
        assert default_decay_rate_for(BlockType.ENTITY)     == 0.001
        assert default_decay_rate_for(BlockType.FACT)       == 0.005
        assert default_decay_rate_for(BlockType.RELATION)   == 0.005
        assert default_decay_rate_for(BlockType.PREFERENCE) == 0.020
        assert default_decay_rate_for(BlockType.EVENT)      == 0.040


class TestStoreUsesPerTypeDecay:
    """Verify MemBlock.store() / AsyncMemBlock.store() pick the
    right rate when caller doesn't specify."""

    def test_default_store_uses_per_type_rate(self):
        from memblock import MemBlock, BlockType
        mem = MemBlock(storage="sqlite:///:memory:")
        try:
            block = mem.store("a fact", type=BlockType.FACT)
            assert block.metadata.decay_rate == 0.005

            ev = mem.store("an event", type=BlockType.EVENT)
            assert ev.metadata.decay_rate == 0.040

            ent = mem.store("an entity", type=BlockType.ENTITY)
            assert ent.metadata.decay_rate == 0.001
        finally:
            mem.close()

    def test_explicit_decay_rate_overrides_default(self):
        from memblock import MemBlock, BlockType
        mem = MemBlock(storage="sqlite:///:memory:")
        try:
            block = mem.store("custom decay", type=BlockType.FACT, decay_rate=0.5)
            assert block.metadata.decay_rate == 0.5
        finally:
            mem.close()
