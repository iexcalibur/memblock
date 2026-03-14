"""Tests for operation log and tamper detection."""

import pytest

from memblock.ops import OpLog
from memblock.storage.sqlite import SQLiteAdapter
from memblock.types import OpAction


@pytest.fixture
def db():
    adapter = SQLiteAdapter(":memory:")
    adapter.initialize()
    yield adapter
    adapter.close()


@pytest.fixture
def op_log(db):
    return OpLog(db, author="test_agent")


class TestOpLog:
    def test_append_operation(self, op_log: OpLog):
        op = op_log.append(OpAction.CREATE, "blk_123", {"content": "test"})
        assert op.action == OpAction.CREATE
        assert op.block_id == "blk_123"
        assert op.hash.startswith("sha256:")
        assert op.clock == 0

    def test_clock_increments(self, op_log: OpLog):
        op1 = op_log.append(OpAction.CREATE, "blk_1")
        op2 = op_log.append(OpAction.UPDATE, "blk_1")
        op3 = op_log.append(OpAction.CREATE, "blk_2")
        assert op1.clock == 0
        assert op2.clock == 1
        assert op3.clock == 2

    def test_hash_chain(self, op_log: OpLog):
        op1 = op_log.append(OpAction.CREATE, "blk_1")
        op2 = op_log.append(OpAction.UPDATE, "blk_1")

        assert op1.prev_hash == ""  # first op has no prev
        assert op2.prev_hash == op1.hash  # second links to first

    def test_verify_valid_chain(self, op_log: OpLog):
        for i in range(10):
            op_log.append(OpAction.CREATE, f"blk_{i}", {"n": i})

        report = op_log.verify()
        assert report.valid is True
        assert report.total_ops == 10

    def test_verify_empty_chain(self, op_log: OpLog):
        report = op_log.verify()
        assert report.valid is True
        assert report.total_ops == 0

    def test_verify_detects_tampered_hash(self, op_log: OpLog, db: SQLiteAdapter):
        op_log.append(OpAction.CREATE, "blk_1")
        op_log.append(OpAction.UPDATE, "blk_1")

        # Tamper: modify hash of first operation directly in DB
        db.conn.execute(
            "UPDATE operations SET hash = 'sha256:tampered' WHERE clock = 0"
        )
        db.conn.commit()

        report = op_log.verify()
        assert report.valid is False
        assert "mismatch" in report.message.lower() or "broken" in report.message.lower()

    def test_verify_detects_tampered_data(self, op_log: OpLog, db: SQLiteAdapter):
        op_log.append(OpAction.CREATE, "blk_1", {"content": "original"})
        op_log.append(OpAction.UPDATE, "blk_1")

        # Tamper: modify the data of first operation
        import json
        db.conn.execute(
            "UPDATE operations SET data = ? WHERE clock = 0",
            (json.dumps({"content": "hacked"}),),
        )
        db.conn.commit()

        report = op_log.verify()
        assert report.valid is False

    def test_get_history(self, op_log: OpLog):
        op_log.append(OpAction.CREATE, "blk_1")
        op_log.append(OpAction.UPDATE, "blk_1")
        op_log.append(OpAction.CREATE, "blk_2")
        op_log.append(OpAction.UPDATE, "blk_1")

        history = op_log.get_history("blk_1")
        assert len(history) == 3  # create + 2 updates
        assert all(op.block_id == "blk_1" for op in history)

    def test_get_version(self, op_log: OpLog):
        op_log.append(OpAction.CREATE, "blk_1")
        op_log.append(OpAction.UPDATE, "blk_1")
        op_log.append(OpAction.UPDATE, "blk_1")

        assert op_log.get_version("blk_1") == 3

    def test_get_last_hash(self, op_log: OpLog):
        assert op_log.get_last_hash() == ""
        op = op_log.append(OpAction.CREATE, "blk_1")
        assert op_log.get_last_hash() == op.hash
