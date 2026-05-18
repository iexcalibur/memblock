"""Tests for hard_delete + introspect_user (G2, v0.12.0).

Soft-delete (existing `delete()`) keeps the row for audit. hard_delete
must irreversibly remove the row + edges + embedding, the way GDPR /
India DPDP "right to be forgotten" expects.

introspect_user must return every block the SDK holds for the bound
user — the API behind "what do you remember about me?" / right-to-
access disclosures.
"""

import pytest

from memblock import MemBlock, BlockType


@pytest.fixture
def mem():
    return MemBlock(storage="sqlite:///:memory:")


class TestHardDelete:
    def test_hard_delete_removes_block(self, mem):
        b = mem.store("My income is ₹X lakh/yr", type=BlockType.FACT)
        assert mem.get(b.id) is not None
        ok = mem.hard_delete(b.id)
        assert ok is True
        assert mem.get(b.id) is None

    def test_hard_delete_unknown_id_returns_false(self, mem):
        assert mem.hard_delete("blk_does_not_exist") is False

    def test_soft_delete_keeps_row_hard_delete_purges(self, mem):
        b1 = mem.store("Soft target", type=BlockType.FACT)
        b2 = mem.store("Hard target", type=BlockType.FACT)
        mem.delete(b1.id)
        mem.hard_delete(b2.id)
        # Soft-deleted block is still in storage (deleted=True) — the
        # underlying storage adapter returns it when raw-fetched.
        # Hard-deleted block is gone entirely.
        assert mem._storage.get_block(b1.id) is not None
        assert mem._storage.get_block(b2.id) is None

    def test_hard_delete_many_counts_purges(self, mem):
        ids = [
            mem.store(f"fact {i}", type=BlockType.FACT).id for i in range(3)
        ]
        # Include one unknown id to confirm it's silently skipped
        n = mem.hard_delete_many(ids + ["blk_unknown"])
        assert n == 3
        for bid in ids:
            assert mem.get(bid) is None


class TestIntrospectUser:
    def test_introspect_returns_all_blocks(self, mem):
        ids = {
            mem.store("fact-1", type=BlockType.FACT).id,
            mem.store("pref-1", type=BlockType.PREFERENCE).id,
            mem.store("ent-1", type=BlockType.ENTITY).id,
        }
        result = mem.introspect_user()
        seen = {b.id for b in result}
        assert ids.issubset(seen)

    def test_introspect_excludes_soft_deleted_by_default(self, mem):
        keep = mem.store("keep", type=BlockType.FACT)
        drop = mem.store("drop", type=BlockType.FACT)
        mem.delete(drop.id)
        result = mem.introspect_user()
        ids = {b.id for b in result}
        assert keep.id in ids
        assert drop.id not in ids

    def test_introspect_include_deleted_returns_everything(self, mem):
        keep = mem.store("keep", type=BlockType.FACT)
        drop = mem.store("drop", type=BlockType.FACT)
        mem.delete(drop.id)
        result = mem.introspect_user(include_deleted=True)
        ids = {b.id for b in result}
        assert keep.id in ids
        assert drop.id in ids

    def test_introspect_respects_limit(self, mem):
        for i in range(5):
            mem.store(f"fact-{i}", type=BlockType.FACT)
        result = mem.introspect_user(limit=2)
        assert len(result) == 2

    def test_introspect_then_forget_round_trip(self, mem):
        """The canonical 'tell me what you remember' → 'forget it all'
        flow. Use the disclosure call as the input to hard_delete_many,
        confirm nothing remains."""
        for i in range(3):
            mem.store(f"sensitive-{i}", type=BlockType.FACT)
        blocks = mem.introspect_user()
        n = mem.hard_delete_many([b.id for b in blocks])
        assert n == len(blocks)
        assert mem.introspect_user() == []
