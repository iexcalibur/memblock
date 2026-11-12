"""Grouped writes — stage privately, publish together, or abort.

Inside a transaction every new block is written *staged*: stored, but invisible
to queries. Changes to blocks that already exist (update, delete, link, unlink)
are held back. `publish()` makes all of it visible; `abort()` discards all of it.

    with mem.transaction() as txn:
        mem.store("User moved to Berlin", receipt="turn-7-a")
        mem.store("User works at Acme", receipt="turn-7-b")
    # both are visible here, or (if the block raised) neither is

How it stays all-or-nothing on every storage backend
----------------------------------------------------
Publishing is one write that decides the outcome, followed by work that can be
repeated safely:

1. The transaction's journal record is saved with state PUBLISHED and the full
   list of staged blocks and held-back changes. This single write is the commit
   point.
2. Staged blocks are made visible and the held-back changes are applied. Every
   step checks whether it was already done, so running it twice is harmless.
3. The journal record is removed.

A crash before step 1 leaves staged blocks that nobody can see; recovery removes
them. A crash after step 1 leaves a PUBLISHED journal; recovery finishes step 2.
On SQLite step 2 also runs inside one database transaction, so a reader never
sees it half done. On adapters without `atomic()` a concurrent reader can see
step 2 in progress for a moment, but the end state is the same.

Receipts
--------
A receipt is a caller-chosen ID for one write. Storing with a receipt that was
already applied returns the block written the first time and writes nothing, so
a retry after a timeout cannot store the same thing twice.

Leases
------
An open transaction holds a lease (`ttl` seconds, renewed as it works). Recovery
leaves an open transaction alone until its lease has expired, so it never
removes the staged work of a writer that is still running.

The journal record is stored through the ordinary adapter contract as a hidden
block (soft-deleted, tagged `_memblock:txn`), so no adapter needs a new table.
It exists only while a transaction is open or half published.
"""

from __future__ import annotations

import json
import time
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from memblock.block import Block
from memblock.errors import TransactionError, ValidationError
from memblock.types import BlockMetadata, BlockType, EdgeRelation

if TYPE_CHECKING:  # pragma: no cover
    from memblock.memblock import MemBlock


#: Tag on the hidden journal record of a transaction.
TXN_TAG = "_memblock:txn"
#: custom_metadata key on a staged block: the open transaction that wrote it.
#: Removed when the transaction publishes.
TXN_KEY = "_txn"
#: custom_metadata key on a block: the receipt it was stored under.
RECEIPT_KEY = "_receipt"
#: Tags that mark records the library keeps for itself.
INTERNAL_TAGS = frozenset({TXN_TAG, "_memblock:manifest"})

DEFAULT_LEASE_SECONDS = 300.0


class TransactionStatus(str, Enum):
    IDLE = "idle"
    OPEN = "open"
    PUBLISHED = "published"
    ABORTED = "aborted"


class ReceiptStatus(str, Enum):
    PENDING = "pending"   # the write was started inside an open transaction
    APPLIED = "applied"   # written (staged if its transaction is still open)
    ABORTED = "aborted"   # the write failed or its transaction was aborted
    UNKNOWN = "unknown"   # nothing is stored under this receipt


def is_internal(block: Block) -> bool:
    """True for records the library keeps for itself (journals, manifests)."""
    return any(tag in INTERNAL_TAGS for tag in (block.tags or []))


def is_staged(block: Block) -> bool:
    """True for a block written inside a transaction that has not published it."""
    return bool(block.deleted and (block.metadata.custom_metadata or {}).get(TXN_KEY))


# ─── Journal ─────────────────────────────────────────────────────────────────


def _save_journal(mem: "MemBlock", payload: dict[str, Any]) -> None:
    mem._storage.save_block(
        Block(
            id=payload["id"],
            type=BlockType.FACT,
            content=json.dumps(payload, sort_keys=True),
            metadata=BlockMetadata(
                confidence=1.0, decay_rate=0.0, created_by=mem._store.op_log.author,
            ),
            tags=[TXN_TAG],
            deleted=True,
        )
    )


def _load_journals(mem: "MemBlock") -> list[dict[str, Any]]:
    journals: list[dict[str, Any]] = []
    for block in mem._storage.query_blocks({"deleted": True, "tags": [TXN_TAG]}):
        if TXN_TAG not in (block.tags or []):
            continue
        try:
            payload = json.loads(block.content)
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("id") == block.id:
            journals.append(payload)
    return journals


def _fault(mem: "MemBlock", point: str) -> None:
    """Test seam: lets a test stop the process at a named point of a publish."""
    hook = getattr(mem, "_txn_fault_hook", None)
    if hook is not None:
        hook(point)


# ─── Applying a published transaction (safe to repeat) ───────────────────────


def _already_applied(block: Block, updates: dict[str, Any]) -> bool:
    current = {
        "content": block.content,
        "type": block.type.value,
        "tags": block.tags,
        "confidence": block.metadata.confidence,
        "source": block.metadata.source.value,
        "decay_rate": block.metadata.decay_rate,
        "ttl": block.metadata.ttl,
    }
    return all(key in current and current[key] == value for key, value in updates.items())


def _apply_op(mem: "MemBlock", op: dict[str, Any]) -> None:
    storage = mem._storage
    kind = op.get("op")
    if kind == "update":
        block = storage.get_block(op["block_id"])
        updates = dict(op["updates"])
        if block is None or _already_applied(block, updates):
            return
        mem._store.update(op["block_id"], **updates)
        if "content" in updates and not block.encrypted:
            mem._embed_block(op["block_id"], updates["content"])
    elif kind == "delete":
        block = storage.get_block(op["block_id"])
        if block is None or block.deleted:
            return
        if mem._store.delete(op["block_id"], cascade=bool(op.get("cascade"))):
            storage.delete_embedding(op["block_id"])
    elif kind == "link":
        relation = EdgeRelation(op["relation"])
        for edge in storage.get_edges(op["source_id"], direction="outgoing"):
            if edge.target_id == op["target_id"] and edge.relation == relation:
                return
        mem._graph.link(op["source_id"], op["target_id"], relation, float(op.get("weight", 1.0)))
    elif kind == "unlink":
        relation = EdgeRelation(op["relation"]) if op.get("relation") else None
        mem._graph.unlink(op["source_id"], op["target_id"], relation)


def apply_published(mem: "MemBlock", payload: dict[str, Any]) -> None:
    """Make a published transaction's work visible. Every step first checks
    whether it is already done, so this can be run again after a crash."""
    storage = mem._storage
    with storage.atomic():
        for index, block_id in enumerate(payload.get("staged", [])):
            _fault(mem, f"apply:{index}")
            block = storage.get_block(block_id)
            if block is None or not block.deleted:
                continue
            custom = dict(block.metadata.custom_metadata or {})
            if custom.get(TXN_KEY) != payload["id"]:
                continue  # already published, then removed by its owner
            # Clearing the marker is what makes "has a marker" mean "still
            # staged", so a published block that is soft-deleted later is
            # never mistaken for unpublished work.
            custom.pop(TXN_KEY, None)
            block.metadata.custom_metadata = custom or None
            block.deleted = False
            storage.save_block(block)
        for op in payload.get("ops", []):
            try:
                _apply_op(mem, op)
            except (ValueError, ValidationError):
                # The target vanished or changed since the change was held back.
                # Skipping keeps one bad change from blocking the rest for ever.
                continue


def _discard_staged(mem: "MemBlock", txn_id: str, known_ids: list[str] | None = None) -> int:
    """Remove the staged blocks of a transaction that will never publish."""
    storage = mem._storage
    ids = list(known_ids or [])
    for block in storage.query_blocks({"deleted": True, "metadata_filters": {TXN_KEY: txn_id}}):
        if block.id not in ids:
            ids.append(block.id)
    removed = 0
    with storage.atomic():
        for block_id in ids:
            block = storage.get_block(block_id)
            # Only ever remove a block that is still staged for this transaction.
            if block is None or not block.deleted:
                continue
            if (block.metadata.custom_metadata or {}).get(TXN_KEY) != txn_id:
                continue
            if block.parent_id:
                parent = storage.get_block(block.parent_id)
                if parent is not None and block_id in parent.children_ids:
                    parent.children_ids.remove(block_id)
                    storage.update_block(parent.id, {"children_ids": parent.children_ids})
            storage.delete_embedding(block_id)
            storage.delete_edges_for_block(block_id)
            storage.delete_block(block_id)
            removed += 1
    return removed


def recover_transactions(mem: "MemBlock", *, force: bool = False) -> dict[str, int]:
    """Finish or discard transactions left behind by a crash.

    A transaction that reached its commit point is completed. One that did not
    is discarded once its lease has expired (`force=True` discards it now; use
    that only when you know no other writer is running).

    Returns counts: `rolled_forward`, `rolled_back`, `in_flight`.
    """
    result = {"rolled_forward": 0, "rolled_back": 0, "in_flight": 0}
    live = mem._current_txn()
    now = time.time()
    for payload in _load_journals(mem):
        txn_id = payload["id"]
        if live is not None and txn_id == live.id:
            result["in_flight"] += 1
            continue
        state = payload.get("state")
        if state == TransactionStatus.PUBLISHED.value:
            apply_published(mem, payload)
            mem._storage.delete_block(txn_id)
            result["rolled_forward"] += 1
        elif force or float(payload.get("expires_at", 0.0)) <= now:
            _discard_staged(mem, txn_id, payload.get("staged"))
            mem._storage.delete_block(txn_id)
            result["rolled_back"] += 1
        else:
            result["in_flight"] += 1
    return result


# ─── Transaction ─────────────────────────────────────────────────────────────


class Transaction:
    """One group of writes. Get one from `MemBlock.transaction()`.

    Use it as a context manager (publishes on success, aborts on an exception),
    or call `begin()`, then `publish()` or `abort()` yourself.
    """

    def __init__(
        self,
        mem: "MemBlock",
        *,
        txn_id: str | None = None,
        ttl: float = DEFAULT_LEASE_SECONDS,
    ) -> None:
        if ttl <= 0:
            raise ValueError("ttl must be positive")
        self._mem = mem
        self.id = txn_id or f"txn_{uuid.uuid4().hex[:16]}"
        self.ttl = float(ttl)
        self.status = TransactionStatus.IDLE
        self.receipts: dict[str, ReceiptStatus] = {}
        self._staged: list[str] = []
        self._ops: list[dict[str, Any]] = []
        self._after: list[Callable[[], None]] = []
        self._depth = 0
        self._failed = False
        self._expires_at = 0.0

    # ── Lifecycle ────────────────────────────────────────────────────────

    def begin(self) -> "Transaction":
        if self.status != TransactionStatus.IDLE:
            raise TransactionError(f"Transaction {self.id} was already started")
        self._mem._recover_once()
        self._mem._txn_enter(self)
        try:
            self._expires_at = time.time() + self.ttl
            _save_journal(self._mem, self._payload(TransactionStatus.OPEN))
        except BaseException:
            self._mem._txn_exit(self)
            raise
        self.status = TransactionStatus.OPEN
        return self

    def renew(self) -> None:
        """Extend the lease of a long-running transaction."""
        self._require_open()
        self._expires_at = time.time() + self.ttl
        _save_journal(self._mem, self._payload(TransactionStatus.OPEN))

    def publish(self) -> dict[str, Any]:
        """Make every staged block and held-back change visible together."""
        self._require_open()
        unfinished = sorted(r for r, s in self.receipts.items() if s != ReceiptStatus.APPLIED)
        if unfinished:
            raise TransactionError(
                f"Transaction {self.id} has writes that did not complete "
                f"({', '.join(unfinished)}); abort it"
            )
        payload = self._payload(TransactionStatus.PUBLISHED)
        _fault(self._mem, "before_commit_point")
        _save_journal(self._mem, payload)  # the commit point
        self.status = TransactionStatus.PUBLISHED
        self._mem._txn_exit(self)
        _fault(self._mem, "after_commit_point")
        try:
            apply_published(self._mem, payload)
            _fault(self._mem, "before_journal_drop")
            self._mem._storage.delete_block(self.id)
        except Exception as error:
            raise TransactionError(
                f"Transaction {self.id} is published but not fully applied; "
                "recover_transactions() will complete it"
            ) from error
        for callback in self._after:
            try:
                callback()
            except Exception:
                pass  # hooks and auto-linking must not undo a published write
        return self.summary()

    def abort(self) -> None:
        """Discard every staged block and held-back change."""
        if self.status == TransactionStatus.ABORTED:
            return
        if self.status == TransactionStatus.PUBLISHED:
            raise TransactionError(f"Transaction {self.id} is already published")
        if self.status == TransactionStatus.IDLE:
            self.status = TransactionStatus.ABORTED
            return
        self._mem._txn_exit(self)
        _discard_staged(self._mem, self.id, self._staged)
        self._mem._storage.delete_block(self.id)
        self.status = TransactionStatus.ABORTED
        self.receipts = {r: ReceiptStatus.ABORTED for r in self.receipts}
        self._staged, self._ops, self._after = [], [], []

    def summary(self) -> dict[str, Any]:
        return {
            "transaction": self.id,
            "status": self.status.value,
            "blocks": list(self._staged),
            "changes": len(self._ops),
            "receipts": {r: s.value for r, s in self.receipts.items()},
        }

    def __enter__(self) -> "Transaction":
        if self._depth == 0 and self.status == TransactionStatus.IDLE:
            self.begin()
        self._depth += 1
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self._depth -= 1
        if exc_type is not None:
            self._failed = True
        if self._depth == 0 and self.status == TransactionStatus.OPEN:
            if self._failed:
                self.abort()
            else:
                try:
                    self.publish()
                except Exception:
                    if self.status == TransactionStatus.OPEN:
                        self.abort()  # never reached its commit point
                    raise
        return False

    def __repr__(self) -> str:
        return f"Transaction(id={self.id!r}, status={self.status.value}, blocks={len(self._staged)})"

    # ── Used by MemBlock while the transaction is open ───────────────────

    def _require_open(self) -> None:
        if self.status != TransactionStatus.OPEN:
            raise TransactionError(f"Transaction {self.id} is {self.status.value}, not open")

    def _payload(self, state: TransactionStatus) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": state.value,
            "owner": self._mem._instance_id,
            "expires_at": self._expires_at,
            "staged": list(self._staged),
            "ops": list(self._ops),
            "receipts": {r: s.value for r, s in self.receipts.items()},
        }

    def _stage(self, block_id: str) -> None:
        self._staged.append(block_id)
        if time.time() > self._expires_at - self.ttl / 2:
            self.renew()

    def _defer(self, op: dict[str, Any]) -> None:
        self._require_open()
        self._ops.append(op)

    def _on_publish(self, callback: Callable[[], None]) -> None:
        self._after.append(callback)

    def _receipt(self, receipt: str, status: ReceiptStatus) -> None:
        self.receipts[receipt] = status
