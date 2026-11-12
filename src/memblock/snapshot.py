"""Snapshots — save a memory to one file, restore it, or copy it elsewhere.

A snapshot carries a stamp: the format, a checksum of its contents, and the
identity of the embedding model that made its vectors. `restore` refuses a file
whose checksum does not match (corrupted or edited) and one whose vectors were
made by a different embedding model, because vectors from different models
cannot be compared. Pass `embeddings="drop"` or `"reembed"` to take the blocks
without the foreign vectors.

The same identity is kept inside the store itself (a small hidden manifest
record), so opening a store with a different embedding model is caught instead
of silently mixing vectors.

Files are written to a temporary name, flushed to disk and then moved into
place, so a crash never leaves a half-written snapshot under the real name.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memblock.block import Block
from memblock.errors import EmbeddingMismatchError, SnapshotError
from memblock.transactions import is_internal, is_staged
from memblock.types import BlockMetadata, BlockType, Edge, OpAction, Operation, now_utc

if TYPE_CHECKING:  # pragma: no cover
    from memblock.memblock import MemBlock


SNAPSHOT_FORMAT = "memblock-snapshot-v1"
MANIFEST_TAG = "_memblock:manifest"
MANIFEST_ID = "mbm_manifest"


# ─── Embedding identity ──────────────────────────────────────────────────────


def embedding_identity(provider: Any) -> str | None:
    """A short string naming the model behind an embedding provider.

    Two providers with the same identity produce comparable vectors. A custom
    provider can define its own `identity` attribute; otherwise the class name,
    model name (when the provider exposes one) and vector size are used.
    """
    if provider is None:
        return None
    explicit = getattr(provider, "identity", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    model = None
    for attr in ("_model_name", "model_name", "_model", "model"):
        value = getattr(provider, attr, None)
        if isinstance(value, str) and value:
            model = value
            break
    try:
        dims = int(provider.dimensions)
    except Exception:
        dims = 0
    return f"{type(provider).__name__}:{model or '-'}:{dims}"


def read_manifest(mem: "MemBlock") -> dict[str, Any] | None:
    block = mem._storage.get_block(MANIFEST_ID)
    if block is None or MANIFEST_TAG not in (block.tags or []):
        return None
    try:
        data = json.loads(block.content)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def write_manifest(mem: "MemBlock", identity: str | None) -> None:
    mem._storage.save_block(
        Block(
            id=MANIFEST_ID,
            type=BlockType.FACT,
            content=json.dumps({"embedding": identity}, sort_keys=True),
            metadata=BlockMetadata(
                confidence=1.0, decay_rate=0.0, created_by=mem._store.op_log.author,
            ),
            tags=[MANIFEST_TAG],
            deleted=True,
        )
    )


def check_embedding_identity(mem: "MemBlock", on_mismatch: str = "error") -> None:
    """Compare this instance's embedding model with the one the store was built
    with. Records the identity the first time a store sees a provider."""
    current = embedding_identity(mem._embedding_provider)
    if current is None:
        return
    stored = (read_manifest(mem) or {}).get("embedding")
    if stored is None:
        write_manifest(mem, current)
        return
    if stored == current:
        return
    message = (
        f"This store's vectors were made by '{stored}', but this instance is "
        f"configured with '{current}'. Vectors from different embedding models "
        "cannot be compared. Open the store with on_embedding_mismatch='ignore' "
        "and call reembed() to rebuild them."
    )
    if on_mismatch == "error":
        raise EmbeddingMismatchError(message)
    if on_mismatch == "warn":
        warnings.warn(message, stacklevel=3)


# ─── File handling ───────────────────────────────────────────────────────────


def _checksum(body: dict[str, Any]) -> str:
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _atomic_write(path: Path, text: str, overwrite: bool) -> None:
    if path.is_symlink():
        raise SnapshotError(f"Snapshot target must not be a symlink: {path}")
    if path.exists() and not overwrite:
        raise SnapshotError(f"Snapshot already exists: {path} (pass overwrite=True to replace it)")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".memblock-snapshot-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)  # fails if another writer got there first
            except FileExistsError as error:
                raise SnapshotError(f"Snapshot already exists: {path}") from error
        try:
            directory = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory = None
        if directory is not None:
            try:
                os.fsync(directory)
            except OSError:
                pass
            finally:
                os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


# ─── Write ───────────────────────────────────────────────────────────────────


def write_snapshot(
    mem: "MemBlock",
    path: str | os.PathLike[str],
    *,
    session_id: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a snapshot of the whole store, or of one session's blocks.

    A whole-store snapshot also carries the operation log, so the copy passes
    `verify()` with the original history. A one-session snapshot carries blocks,
    edges and vectors only; restoring it logs the blocks as new in the target.

    Returns the stamp, with `path` added.
    """
    from memblock import __version__

    if mem._current_txn() is not None:
        raise SnapshotError("Publish or abort the open transaction before taking a snapshot")

    storage = mem._storage
    blocks = [
        b for b in storage.get_all_blocks(include_deleted=True)
        if not is_internal(b) and not is_staged(b)
    ]
    whole_store = session_id is None
    if not whole_store:
        blocks = [b for b in blocks if b.metadata.session_id == session_id]
    blocks.sort(key=lambda b: b.id)
    ids = {b.id for b in blocks}

    edges: dict[str, dict[str, Any]] = {}
    for block in blocks:
        for edge in storage.get_edges(block.id, direction="outgoing"):
            if edge.target_id in ids:
                edges[edge.id] = edge.to_dict()

    embeddings = {
        block_id: base64.b64encode(blob).decode("ascii")
        for block_id, blob in storage.get_all_embeddings()
        if block_id in ids
    }
    operations = [op.to_dict() for op in storage.get_operations()] if whole_store else []

    body = {
        "blocks": [b.to_dict() for b in blocks],
        "edges": [edges[key] for key in sorted(edges)],
        "embeddings": embeddings,
        "operations": operations,
    }
    stamp = {
        "format": SNAPSHOT_FORMAT,
        "memblock_version": __version__,
        "created_at": now_utc().isoformat(),
        "scope": {"session_id": session_id},
        "embedding": embedding_identity(mem._embedding_provider) if embeddings else None,
        "counts": {
            "blocks": len(blocks),
            "edges": len(edges),
            "embeddings": len(embeddings),
            "operations": len(operations),
        },
        "encrypted_blocks": sum(1 for b in blocks if b.encrypted),
        "operations_head": operations[-1]["hash"] if operations else None,
        "checksum": _checksum(body),
    }
    target = Path(path)
    _atomic_write(target, json.dumps({"stamp": stamp, **body}, sort_keys=True), overwrite)
    return {**stamp, "path": str(target)}


# ─── Read and restore ────────────────────────────────────────────────────────


def read_snapshot(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a snapshot file and check its format and checksum."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SnapshotError(f"Snapshot not found: {path}") from error
    except (OSError, ValueError) as error:
        raise SnapshotError(f"Not a readable snapshot: {path}") from error
    stamp = document.get("stamp") if isinstance(document, dict) else None
    if not isinstance(stamp, dict) or stamp.get("format") != SNAPSHOT_FORMAT:
        raise SnapshotError(f"Not a {SNAPSHOT_FORMAT} file: {path}")
    body = {key: document.get(key) for key in ("blocks", "edges", "embeddings", "operations")}
    if (
        not isinstance(body["blocks"], list)
        or not isinstance(body["edges"], list)
        or not isinstance(body["embeddings"], dict)
        or not isinstance(body["operations"], list)
    ):
        raise SnapshotError(f"Snapshot is missing parts: {path}")
    if _checksum(body) != stamp.get("checksum"):
        raise SnapshotError(f"Snapshot checksum does not match its contents: {path}")
    return {"stamp": stamp, **body}


def restore_snapshot(
    mem: "MemBlock",
    path: str | os.PathLike[str],
    *,
    mode: str = "empty",
    embeddings: str = "strict",
) -> dict[str, Any]:
    """Load a snapshot into this store.

    mode:
        "empty" (default) — the store must hold no blocks. The operation log is
            restored as it was, so `verify()` passes with the original history.
        "merge" — add the snapshot's blocks to what is already here. A block
            whose ID already exists is left as it is. Added blocks are logged
            as new operations in this store's own log.
    embeddings:
        "strict" (default) — refuse a snapshot whose vectors were made by a
            different embedding model than this instance uses.
        "drop" — take the blocks, leave the vectors out.
        "reembed" — take the blocks and build fresh vectors with this
            instance's model (skips blocks whose content cannot be read).
    """
    if mode not in ("empty", "merge"):
        raise ValueError("mode must be 'empty' or 'merge'")
    if embeddings not in ("strict", "drop", "reembed"):
        raise ValueError("embeddings must be 'strict', 'drop' or 'reembed'")
    if mem._current_txn() is not None:
        raise SnapshotError("Publish or abort the open transaction before restoring")

    document = read_snapshot(path)
    stamp = document["stamp"]
    storage = mem._storage

    existing = {
        b.id for b in storage.get_all_blocks(include_deleted=True) if not is_internal(b)
    }
    if mode == "empty" and existing:
        raise SnapshotError(
            f"This store already holds {len(existing)} blocks; restore into an "
            "empty store, or pass mode='merge'"
        )

    vectors: dict[str, str] = document["embeddings"]
    current = embedding_identity(mem._embedding_provider)
    if vectors and embeddings == "strict" and stamp.get("embedding") != current:
        raise SnapshotError(
            f"The snapshot's vectors were made by '{stamp.get('embedding')}', but "
            f"this instance uses '{current}'. Pass embeddings='drop' or 'reembed'."
        )
    if vectors and embeddings == "strict":
        mem._ensure_embedding_identity()

    added: list[Block] = []
    with storage.atomic():
        for data in document["blocks"]:
            block = Block.from_dict(data)
            if block.id in existing:
                continue
            block.edges = []
            storage.save_block(block)
            added.append(block)
        added_ids = {b.id for b in added}
        known = existing | added_ids
        restored_edges = 0
        for data in document["edges"]:
            edge = Edge.from_dict(data)
            if edge.source_id in added_ids and edge.target_id in known:
                storage.save_edge(edge)
                restored_edges += 1
        keep_history = (
            mode == "empty"
            and bool(document["operations"])
            and storage.get_last_operation() is None
        )
        if keep_history:
            for data in document["operations"]:
                storage.save_operation(Operation.from_dict(data))
            mem._store.op_log._clock = mem._store.op_log._get_current_clock()
        else:
            for block in added:
                op = mem._store.op_log.append(
                    action=OpAction.CREATE,
                    block_id=block.id,
                    data={"content": block.content, "type": block.type.value, "restored": True},
                )
                storage.update_block(block.id, {"op_hash": op.hash})
        restored_vectors = 0
        if embeddings == "strict":
            for block_id, encoded in vectors.items():
                if block_id in added_ids:
                    storage.save_embedding(block_id, base64.b64decode(encoded))
                    restored_vectors += 1

    if embeddings == "reembed" and mem._embedding_provider is not None:
        mem._ensure_embedding_identity()
        for block in added:
            if block.deleted:
                continue
            text = mem._plaintext(block)
            if text is None:
                continue  # sealed, and this instance has no key for it
            mem._embed_block(block.id, text)
            restored_vectors += 1

    return {
        "path": str(path),
        "mode": mode,
        "blocks": len(added),
        "skipped": len(document["blocks"]) - len(added),
        "edges": restored_edges,
        "embeddings": restored_vectors,
        "operations": len(document["operations"]) if keep_history else len(added),
        "history_kept": keep_history,
        "stamp": stamp,
    }
