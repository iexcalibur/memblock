"""Grouped writes, receipts and crash recovery (memblock/transactions.py)."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import pytest

from memblock import (
    BlockType,
    EdgeRelation,
    EncryptionLevel,
    EventType,
    MemBlock,
    ReceiptStatus,
    TransactionError,
    TransactionStatus,
)
from memblock.embeddings import EmbeddingProvider
from memblock.transactions import (
    TXN_KEY,
    apply_published,
    is_internal,
    is_staged,
    recover_transactions,
)


class Crash(BaseException):
    """Stands in for the process dying: a BaseException skips every cleanup path."""


class WordVectors(EmbeddingProvider):
    """Deterministic 16-dim vectors, so tests need no model download."""

    identity = "word-vectors:16"

    def embed(self, texts):
        out = []
        for text in texts:
            vec = [0.0] * 16
            for word in text.lower().split():
                vec[sum(map(ord, word)) % 16] += 1.0
            out.append(vec)
        return out

    @property
    def dimensions(self):
        return 16


@pytest.fixture
def db(tmp_path):
    return f"sqlite:///{tmp_path / 'mem.db'}"


@pytest.fixture
def mem(db):
    m = MemBlock(storage=db)
    yield m
    m.close()


def visible(mem):
    return sorted(b.content for b in mem.query(limit=200))


def hidden_rows(mem):
    """Rows nobody should be left with: staged blocks and journal records."""
    return [
        b for b in mem._storage.get_all_blocks(include_deleted=True)
        if is_internal(b) or is_staged(b)
    ]


def crash_at(mem, point):
    def hook(reached):
        if reached == point:
            raise Crash(point)
    mem._txn_fault_hook = hook


# ─── Visibility ──────────────────────────────────────────────────────────────


class TestPublishAndAbort:
    def test_staged_blocks_are_invisible_until_publish(self, mem):
        with mem.transaction() as txn:
            a = mem.store("User moved to Berlin")
            b = mem.store("User works at Acme")
            assert visible(mem) == []
            assert a.deleted and b.deleted
            assert mem._storage.get_block(a.id).metadata.custom_metadata[TXN_KEY] == txn.id
        assert visible(mem) == ["User moved to Berlin", "User works at Acme"]
        assert txn.status == TransactionStatus.PUBLISHED
        assert hidden_rows(mem) == []
        # the staging marker is gone once published
        assert TXN_KEY not in (mem.get(a.id).metadata.custom_metadata or {})

    def test_exception_discards_everything(self, mem):
        mem.store("kept from before")
        with pytest.raises(RuntimeError):
            with mem.transaction() as txn:
                mem.store("never visible 1")
                mem.store("never visible 2")
                raise RuntimeError("boom")
        assert visible(mem) == ["kept from before"]
        assert txn.status == TransactionStatus.ABORTED
        assert hidden_rows(mem) == []
        assert mem.stats()["deleted_blocks"] == 0

    def test_manual_begin_publish_abort(self, mem):
        txn = mem.transaction().begin()
        mem.store("one")
        summary = txn.publish()
        assert summary["status"] == "published" and len(summary["blocks"]) == 1
        with pytest.raises(TransactionError):
            txn.publish()
        with pytest.raises(TransactionError):
            txn.abort()

        other = mem.transaction().begin()
        mem.store("two")
        other.abort()
        other.abort()  # a second abort is harmless
        assert visible(mem) == ["one"]

    def test_changes_to_existing_blocks_are_held_back(self, mem):
        old = mem.store("User lives in Paris")
        gone = mem.store("User owns a bike")
        with mem.transaction():
            new = mem.store("User moved to Berlin")
            returned = mem.update(old.id, content="User lived in Paris until 2026", confidence=0.6)
            assert returned.content == "User lives in Paris"  # as it stands now
            assert mem.delete(gone.id) is True
            mem.link(new.id, old.id, relation="supersedes")
            assert visible(mem) == ["User lives in Paris", "User owns a bike"]
            assert mem._storage.get_edges(new.id) == []
        assert visible(mem) == ["User lived in Paris until 2026", "User moved to Berlin"]
        assert mem.get(old.id).metadata.confidence == 0.6
        edges = mem._storage.get_edges(new.id, direction="outgoing")
        assert [(e.target_id, e.relation) for e in edges] == [(old.id, EdgeRelation.SUPERSEDES)]

    def test_held_back_changes_are_dropped_on_abort(self, mem):
        old = mem.store("User lives in Paris")
        with pytest.raises(RuntimeError):
            with mem.transaction():
                mem.update(old.id, content="changed")
                mem.delete(old.id)
                raise RuntimeError
        assert visible(mem) == ["User lives in Paris"]

    def test_unlink_is_held_back(self, mem):
        a, b = mem.store("a"), mem.store("b")
        mem.link(a.id, b.id, relation="related_to")
        with mem.transaction():
            assert mem.unlink(a.id, b.id) == 0
            assert len(mem._storage.get_edges(a.id, direction="outgoing")) == 1
        assert mem._storage.get_edges(a.id, direction="outgoing") == []

    def test_bad_link_is_refused_while_it_can_still_abort(self, mem):
        a = mem.store("a")
        with pytest.raises(ValueError):
            with mem.transaction():
                mem.link(a.id, "blk_missing", relation="related_to")
        assert hidden_rows(mem) == []

    def test_deleting_a_staged_block_drops_it(self, mem):
        with mem.transaction():
            keep = mem.store("keep")
            drop = mem.store("drop")
            assert mem.delete(drop.id) is True
        assert visible(mem) == ["keep"]
        assert mem._storage.get_block(drop.id) is None
        assert keep.id

    def test_nested_transaction_joins_the_outer_one(self, mem):
        with mem.transaction() as outer:
            mem.store("outer")
            with mem.transaction() as inner:
                assert inner is outer
                mem.store("inner")
            assert visible(mem) == []  # leaving the inner block publishes nothing
        assert visible(mem) == ["inner", "outer"]

    def test_inner_failure_aborts_the_whole_transaction(self, mem):
        with mem.transaction() as outer:
            mem.store("outer")
            try:
                with mem.transaction():
                    mem.store("inner")
                    raise RuntimeError
            except RuntimeError:
                pass
        assert outer.status == TransactionStatus.ABORTED
        assert visible(mem) == []

    def test_hard_delete_is_not_allowed_inside(self, mem):
        block = mem.store("x")
        with mem.transaction():
            with pytest.raises(TransactionError):
                mem.hard_delete(block.id)
        assert visible(mem) == ["x"]

    def test_hash_chain_stays_valid(self, mem):
        with mem.transaction():
            a = mem.store("one")
            mem.store("two")
        with pytest.raises(RuntimeError):
            with mem.transaction():
                mem.store("three")
                mem.update(a.id, content="changed")
                raise RuntimeError
        assert mem.verify().valid

    def test_hooks_fire_on_publish_not_on_abort(self, mem):
        added, updated = [], []
        mem.on(EventType.ON_ADD.value, lambda data: added.append(data["content"]))
        mem.on(EventType.ON_UPDATE.value, lambda data: updated.append(data["block_id"]))
        base = mem.store("base")
        added.clear()
        with mem.transaction():
            mem.store("in txn")
            mem.update(base.id, confidence=0.5)
            assert added == [] and updated == []
        assert added == ["in txn"] and updated == [base.id]
        with pytest.raises(RuntimeError):
            with mem.transaction():
                mem.store("aborted")
                raise RuntimeError
        assert added == ["in txn"]

    def test_parent_child_list_is_cleaned_on_abort(self, mem):
        parent = mem.store("Project Alpha", type=BlockType.ENTITY)
        with pytest.raises(RuntimeError):
            with mem.transaction():
                mem.store("Alpha ships in May", parent_id=parent.id)
                raise RuntimeError
        assert mem.get(parent.id).children_ids == []

    def test_encrypted_block_in_a_transaction(self, tmp_path):
        m = MemBlock(storage=f"sqlite:///{tmp_path / 'e.db'}", encryption_key="secret")
        with m.transaction():
            block = m.store("card ends 4242", encryption_level=EncryptionLevel.SENSITIVE)
        stored = m._storage.get_block(block.id)
        assert stored.encrypted and "4242" not in stored.content
        assert m.get(block.id).content == "card ends 4242"
        m.close()

    def test_vector_search_does_not_see_staged_blocks(self, tmp_path):
        m = MemBlock(storage=f"sqlite:///{tmp_path / 'v.db'}", embeddings=WordVectors())
        with m.transaction():
            m.store("zebra crossing rules")
            assert m.query(text_search="zebra crossing", limit=5) == []
        assert [b.content for b in m.query(text_search="zebra crossing", limit=5)] == ["zebra crossing rules"]
        m.close()

    def test_another_threads_write_is_not_staged(self, mem):
        seen = {}

        def other():
            block = mem.store("from another thread")
            seen["deleted"] = block.deleted

        with mem.transaction():
            mem.store("staged")
            worker = threading.Thread(target=other)
            worker.start()
            worker.join()
            assert visible(mem) == ["from another thread"]
        assert seen["deleted"] is False

    def test_internal_records_stay_hidden_while_open(self, mem):
        mem.store("visible")
        with mem.transaction():
            mem.store("staged")
            assert [b.content for b in mem.introspect_user(include_deleted=True)] == ["visible"]
            stats = mem.stats()
            assert stats["total_blocks"] == 1
        assert mem.stats()["total_blocks"] == 2


# ─── Receipts ────────────────────────────────────────────────────────────────


class TestReceipts:
    def test_replay_returns_the_first_block_and_writes_nothing(self, mem):
        first = mem.store("User moved to Berlin", receipt="r1")
        again = mem.store("User moved to Berlin", receipt="r1")
        different_text = mem.store("something else entirely", receipt="r1")
        assert again.id == first.id == different_text.id
        assert visible(mem) == ["User moved to Berlin"]
        assert mem.receipt_status("r1") == ReceiptStatus.APPLIED
        assert mem.receipt_status("never-used") == ReceiptStatus.UNKNOWN

    def test_status_through_a_transaction(self, mem):
        with mem.transaction() as txn:
            block = mem.store("staged", receipt="r1")
            assert mem.receipt_status("r1") == ReceiptStatus.PENDING
            assert txn.receipts == {"r1": ReceiptStatus.APPLIED}
            assert mem.store("staged", receipt="r1").id == block.id  # replay inside
        assert mem.receipt_status("r1") == ReceiptStatus.APPLIED
        with mem.transaction():
            assert mem.store("staged", receipt="r1").id == block.id  # and in a later one
        assert visible(mem) == ["staged"]

    def test_aborted_receipt_can_be_used_again(self, mem):
        with pytest.raises(RuntimeError):
            with mem.transaction() as txn:
                mem.store("first try", receipt="r1")
                raise RuntimeError
        assert txn.receipts == {"r1": ReceiptStatus.ABORTED}
        assert mem.receipt_status("r1") == ReceiptStatus.UNKNOWN
        mem.store("second try", receipt="r1")
        assert visible(mem) == ["second try"]

    def test_publish_refuses_a_transaction_with_a_failed_write(self, mem, monkeypatch):
        txn = mem.transaction().begin()
        mem.store("fine", receipt="ok")
        real = mem._store.create

        def failing(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(mem._store, "create", failing)
        with pytest.raises(OSError):
            mem.store("fails", receipt="bad")
        monkeypatch.setattr(mem._store, "create", real)
        assert txn.receipts == {"ok": ReceiptStatus.APPLIED, "bad": ReceiptStatus.ABORTED}
        with pytest.raises(TransactionError, match="bad"):
            txn.publish()
        txn.abort()
        assert visible(mem) == [] and hidden_rows(mem) == []

    def test_empty_receipt_is_rejected(self, mem):
        with pytest.raises(Exception):
            mem.store("x", receipt="")

    def test_episode_receipts(self, mem):
        a = mem.store_episode("ok", role="user", session_id="s", turn=1, seq=0, receipt="s:t1:0")
        b = mem.store_episode("ok", role="user", session_id="s", turn=1, seq=0, receipt="s:t1:0")
        assert a.id == b.id
        assert len(mem.query(type=BlockType.EPISODE, limit=10)) == 1


# ─── Crash recovery ──────────────────────────────────────────────────────────


def stage_three(mem, base_id=None):
    """Open a transaction, stage three blocks (and a change), then publish."""
    with mem.transaction():
        for i in range(3):
            mem.store(f"turn part {i}", receipt=f"r{i}")
        if base_id is not None:
            mem.update(base_id, content="base (updated)")


class TestCrashRecovery:
    def test_crash_before_the_commit_point_rolls_back(self, db):
        mem = MemBlock(storage=db)
        base = mem.store("base")
        crash_at(mem, "before_commit_point")
        with pytest.raises(Crash):
            stage_three(mem, base.id)
        mem.close()

        fresh = MemBlock(storage=db)
        assert visible(fresh) == ["base"]
        # the dead writer's lease has not expired: recovery leaves it alone
        assert recover_transactions(fresh) == {"rolled_forward": 0, "rolled_back": 0, "in_flight": 1}
        assert len(hidden_rows(fresh)) == 4  # three staged blocks + the journal
        assert fresh.recover_transactions(force=True)["rolled_back"] == 1
        assert visible(fresh) == ["base"] and hidden_rows(fresh) == []
        assert fresh.verify().valid
        # the retry stores each part exactly once
        stage_three(fresh)
        stage_three(fresh)
        assert visible(fresh) == ["base", "turn part 0", "turn part 1", "turn part 2"]
        fresh.close()

    def test_expired_lease_is_rolled_back_by_the_next_transaction(self, db):
        mem = MemBlock(storage=db)
        crash_at(mem, "before_commit_point")
        with pytest.raises(Crash):
            with mem.transaction(ttl=0.05):
                mem.store("orphan")
        mem.close()
        time.sleep(0.1)

        fresh = MemBlock(storage=db)
        with fresh.transaction():  # opening one runs recovery first
            fresh.store("new work")
        assert visible(fresh) == ["new work"] and hidden_rows(fresh) == []
        fresh.close()

    def test_crash_after_the_commit_point_rolls_forward(self, db):
        mem = MemBlock(storage=db)
        base = mem.store("base")
        crash_at(mem, "after_commit_point")
        with pytest.raises(Crash):
            stage_three(mem, base.id)
        mem.close()

        fresh = MemBlock(storage=db)
        assert visible(fresh) == ["base"]  # decided, not yet applied
        assert fresh.recover_transactions() == {"rolled_forward": 1, "rolled_back": 0, "in_flight": 0}
        assert visible(fresh) == ["base (updated)", "turn part 0", "turn part 1", "turn part 2"]
        assert hidden_rows(fresh) == [] and fresh.verify().valid
        assert fresh.receipt_status("r1") == ReceiptStatus.APPLIED
        fresh.close()

    def test_crash_mid_apply_is_never_seen_half_done_on_sqlite(self, db):
        mem = MemBlock(storage=db)
        assert mem._storage.supports_atomic
        crash_at(mem, "apply:2")
        with pytest.raises(Crash):
            stage_three(mem)
        mem.close()

        fresh = MemBlock(storage=db)
        assert visible(fresh) == []  # the half-applied step was rolled back as one
        assert fresh.recover_transactions()["rolled_forward"] == 1
        assert visible(fresh) == ["turn part 0", "turn part 1", "turn part 2"]
        fresh.close()

    def test_crash_mid_apply_without_atomic_support_still_converges(self, db):
        """An adapter with no `atomic()` shows the step in progress, then recovery
        finishes it: same end state, nothing lost, nothing doubled."""
        mem = MemBlock(storage=db)

        @contextmanager
        def no_grouping():
            yield

        mem._storage.atomic = no_grouping
        crash_at(mem, "apply:2")
        with pytest.raises(Crash):
            stage_three(mem)
        mem.close()

        fresh = MemBlock(storage=db)
        assert visible(fresh) == ["turn part 0", "turn part 1"]  # the window
        assert fresh.recover_transactions()["rolled_forward"] == 1
        assert visible(fresh) == ["turn part 0", "turn part 1", "turn part 2"]
        assert hidden_rows(fresh) == []
        fresh.close()

    def test_crash_before_the_journal_is_removed(self, db):
        mem = MemBlock(storage=db)
        crash_at(mem, "before_journal_drop")
        with pytest.raises(Crash):
            stage_three(mem)
        mem.close()

        fresh = MemBlock(storage=db)
        assert fresh.recover_transactions()["rolled_forward"] == 1
        assert visible(fresh) == ["turn part 0", "turn part 1", "turn part 2"]
        assert hidden_rows(fresh) == []
        fresh.close()

    def test_applying_twice_changes_nothing(self, mem):
        base = mem.store("base")
        other = mem.store("other")
        txn = mem.transaction().begin()
        new = mem.store("new")
        mem.update(base.id, content="base (updated)")
        mem.link(new.id, other.id, relation="related_to")
        payload = txn._payload(TransactionStatus.PUBLISHED)
        txn.publish()
        version = mem.get(base.id).version
        ops = mem.stats()["total_operations"]

        apply_published(mem, payload)
        apply_published(mem, payload)
        assert mem.get(base.id).version == version
        assert mem.stats()["total_operations"] == ops
        assert len(mem._storage.get_edges(new.id, direction="outgoing")) == 1

    def test_recovery_never_touches_a_published_block_deleted_later(self, db):
        mem = MemBlock(storage=db)
        with mem.transaction():
            block = mem.store("published, then removed by its owner")
        mem.delete(block.id)
        assert not is_staged(mem._storage.get_block(block.id))
        assert mem.recover_transactions(force=True) == {"rolled_forward": 0, "rolled_back": 0, "in_flight": 0}
        stored = mem._storage.get_block(block.id)
        assert stored is not None and stored.deleted  # still there, still soft-deleted
        mem.close()

    def test_own_open_transaction_is_left_alone(self, mem):
        with mem.transaction():
            mem.store("mine")
            assert mem.recover_transactions(force=True)["in_flight"] == 1
        assert visible(mem) == ["mine"]


# ─── SQLite atomic() ─────────────────────────────────────────────────────────


class TestSqliteAtomic:
    def test_all_or_nothing(self, mem):
        storage = mem._storage
        with pytest.raises(RuntimeError):
            with storage.atomic():
                mem.store("one")
                mem.store("two")
                raise RuntimeError
        assert visible(mem) == []
        with storage.atomic():
            mem.store("one")
            with storage.atomic():
                mem.store("two")
        assert visible(mem) == ["one", "two"]
        assert mem.verify().valid
