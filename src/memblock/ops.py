"""Operation log — append-only log with SHA-256 hash chain for tamper detection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from memblock.storage.base import StorageAdapter
from memblock.types import OpAction, Operation, generate_op_id, now_utc


@dataclass
class TamperReport:
    """Result of a hash chain verification."""

    valid: bool
    total_ops: int
    first_tampered_op: str | None = None  # ID of first corrupted op
    message: str = ""


class OpLog:
    """
    Append-only operation log with SHA-256 hash chaining.

    Every mutation (create, update, delete, link, unlink) is recorded as an
    operation. Each operation's hash is computed from the previous hash + its
    own data, forming a tamper-evident chain.
    """

    def __init__(self, storage: StorageAdapter, author: str = "agent") -> None:
        self.storage = storage
        self.author = author
        self._clock = self._get_current_clock()

    def _get_current_clock(self) -> int:
        """Get the next clock value from the last operation."""
        last_op = self.storage.get_last_operation()
        if last_op is None:
            return 0
        return last_op.clock + 1

    def _compute_hash(self, prev_hash: str, action: str, data: dict, timestamp: str) -> str:
        """Compute SHA-256 hash for tamper detection."""
        payload = f"{prev_hash}|{action}|{json.dumps(data, sort_keys=True)}|{timestamp}"
        return f"sha256:{hashlib.sha256(payload.encode()).hexdigest()}"

    def append(
        self,
        action: OpAction,
        block_id: str,
        data: dict[str, Any] | None = None,
    ) -> Operation:
        """
        Append a new operation to the log.

        Returns the created Operation with computed hash.
        """
        if data is None:
            data = {}

        # Get previous hash for chaining
        last_op = self.storage.get_last_operation()
        prev_hash = last_op.hash if last_op else ""

        timestamp = now_utc()
        op_hash = self._compute_hash(
            prev_hash, action.value, data, timestamp.isoformat()
        )

        op = Operation(
            id=generate_op_id(),
            block_id=block_id,
            author=self.author,
            clock=self._clock,
            action=action,
            data=data,
            hash=op_hash,
            prev_hash=prev_hash,
            timestamp=timestamp,
        )

        self.storage.save_operation(op)
        self._clock += 1

        return op

    def verify(self) -> TamperReport:
        """
        Walk the entire operation chain and verify all hashes.

        Returns a TamperReport indicating if the chain is valid.
        """
        ops = self.storage.get_operations()

        if not ops:
            return TamperReport(valid=True, total_ops=0, message="No operations to verify")

        prev_hash = ""
        for op in ops:
            # Verify the prev_hash links correctly
            if op.prev_hash != prev_hash:
                return TamperReport(
                    valid=False,
                    total_ops=len(ops),
                    first_tampered_op=op.id,
                    message=f"Chain broken at op {op.id}: expected prev_hash={prev_hash!r}, got {op.prev_hash!r}",
                )

            # Recompute hash and verify
            expected_hash = self._compute_hash(
                op.prev_hash, op.action.value, op.data, op.timestamp.isoformat()
            )
            if op.hash != expected_hash:
                return TamperReport(
                    valid=False,
                    total_ops=len(ops),
                    first_tampered_op=op.id,
                    message=f"Hash mismatch at op {op.id}: expected {expected_hash}, got {op.hash}",
                )

            prev_hash = op.hash

        return TamperReport(
            valid=True,
            total_ops=len(ops),
            message=f"All {len(ops)} operations verified successfully",
        )

    def get_history(self, block_id: str) -> list[Operation]:
        """Get all operations for a specific block, ordered by clock."""
        return self.storage.get_operations(block_id=block_id)

    def get_version(self, block_id: str) -> int:
        """Get the current version number for a block (number of ops)."""
        ops = self.storage.get_operations(block_id=block_id)
        return len(ops)

    def get_last_hash(self) -> str:
        """Get the hash of the most recent operation."""
        last_op = self.storage.get_last_operation()
        return last_op.hash if last_op else ""
