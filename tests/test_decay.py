"""Tests for memory decay engine."""

import pytest
from datetime import datetime, timedelta, timezone

from memblock.block import Block
from memblock.decay import DecayEngine
from memblock.storage.sqlite import SQLiteAdapter
from memblock.types import BlockMetadata, BlockType, SourceType


@pytest.fixture
def db():
    adapter = SQLiteAdapter(":memory:")
    adapter.initialize()
    yield adapter
    adapter.close()


@pytest.fixture
def decay(db):
    return DecayEngine(db)


def make_block(
    content="Test",
    confidence=1.0,
    decay_rate=0.01,
    access_count=0,
    hours_ago=0,
    last_accessed_hours_ago=None,
    ttl=None,
) -> Block:
    created = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    last_accessed = None
    if last_accessed_hours_ago is not None:
        last_accessed = datetime.now(timezone.utc) - timedelta(hours=last_accessed_hours_ago)

    return Block(
        content=content,
        type=BlockType.FACT,
        metadata=BlockMetadata(
            confidence=confidence,
            source=SourceType.EXPLICIT,
            created_at=created,
            access_count=access_count,
            last_accessed=last_accessed,
            decay_rate=decay_rate,
            ttl=ttl,
        ),
    )


class TestDecayCalculation:
    def test_fresh_block_full_strength(self, decay: DecayEngine):
        block = make_block(confidence=1.0, hours_ago=0, decay_rate=0.01)
        strength = decay.calculate_strength(block)
        assert strength >= 0.99  # fresh block, near full strength

    def test_old_block_decays(self, decay: DecayEngine):
        fresh = make_block(confidence=1.0, hours_ago=0, decay_rate=0.1)
        old = make_block(confidence=1.0, hours_ago=100, decay_rate=0.1)

        fresh_strength = decay.calculate_strength(fresh)
        old_strength = decay.calculate_strength(old)

        assert fresh_strength > old_strength

    def test_zero_decay_rate_never_decays(self, decay: DecayEngine):
        block = make_block(confidence=1.0, hours_ago=10000, decay_rate=0.0)
        strength = decay.calculate_strength(block)
        assert strength == 1.0

    def test_high_decay_rate_fades_fast(self, decay: DecayEngine):
        block = make_block(confidence=1.0, hours_ago=24, decay_rate=1.0)
        strength = decay.calculate_strength(block)
        assert strength < 0.01  # should be nearly zero

    def test_access_boosts_strength(self, decay: DecayEngine):
        no_access = make_block(confidence=0.5, hours_ago=48, decay_rate=0.01)
        with_access = make_block(
            confidence=0.5,
            hours_ago=48,
            decay_rate=0.01,
            access_count=50,
            last_accessed_hours_ago=1,
        )

        s1 = decay.calculate_strength(no_access)
        s2 = decay.calculate_strength(with_access)

        assert s2 > s1  # accessed block should be stronger

    def test_low_confidence_means_low_strength(self, decay: DecayEngine):
        high_conf = make_block(confidence=0.9, hours_ago=0)
        low_conf = make_block(confidence=0.1, hours_ago=0)

        assert decay.calculate_strength(high_conf) > decay.calculate_strength(low_conf)

    def test_strength_clamped_to_one(self, decay: DecayEngine):
        block = make_block(
            confidence=1.0,
            hours_ago=0,
            access_count=1000,
            last_accessed_hours_ago=0,
        )
        strength = decay.calculate_strength(block)
        assert strength <= 1.0


class TestDecayPrune:
    def test_prune_weak_blocks(self, db: SQLiteAdapter, decay: DecayEngine):
        # Create a block that's very old with high decay
        old_block = make_block(content="Old memory", confidence=0.1, hours_ago=1000, decay_rate=0.5)
        db.save_block(old_block)

        # Create a fresh strong block
        fresh_block = make_block(content="Fresh memory", confidence=1.0, hours_ago=0)
        db.save_block(fresh_block)

        pruned = decay.prune(min_strength=0.05)

        pruned_ids = {b.id for b in pruned}
        assert old_block.id in pruned_ids
        assert fresh_block.id not in pruned_ids

    def test_prune_expired_ttl(self, db: SQLiteAdapter, decay: DecayEngine):
        # TTL of 1 second, created 1 hour ago
        expired = make_block(content="Expired", hours_ago=1, ttl=1)
        db.save_block(expired)

        fresh = make_block(content="Fresh", hours_ago=0, ttl=86400)
        db.save_block(fresh)

        pruned = decay.prune()
        pruned_ids = {b.id for b in pruned}
        assert expired.id in pruned_ids
        assert fresh.id not in pruned_ids

    def test_touch_strengthens_block(self, db: SQLiteAdapter, decay: DecayEngine):
        block = make_block(content="Touch me", access_count=0)
        db.save_block(block)

        decay.touch(block.id)
        decay.touch(block.id)
        decay.touch(block.id)

        updated = db.get_block(block.id)
        assert updated.metadata.access_count == 3
        assert updated.metadata.last_accessed is not None


class TestDecayRanking:
    def test_get_strongest(self, db: SQLiteAdapter, decay: DecayEngine):
        for i in range(5):
            block = make_block(
                content=f"Block {i}",
                confidence=0.2 * (i + 1),  # 0.2, 0.4, 0.6, 0.8, 1.0
            )
            db.save_block(block)

        strongest = decay.get_strongest(limit=3)
        assert len(strongest) == 3

        # Should be in descending strength order
        strengths = [s for _, s in strongest]
        assert strengths == sorted(strengths, reverse=True)

    def test_get_weakest(self, db: SQLiteAdapter, decay: DecayEngine):
        for i in range(5):
            block = make_block(content=f"Block {i}", confidence=0.2 * (i + 1))
            db.save_block(block)

        weakest = decay.get_weakest(limit=2)
        assert len(weakest) == 2

        strengths = [s for _, s in weakest]
        assert strengths == sorted(strengths)  # ascending
