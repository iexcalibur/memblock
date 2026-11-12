"""Stamped snapshots, restore, fork and embedding identity (memblock/snapshot.py)."""

from __future__ import annotations

import json
import os
import warnings

import pytest

from memblock import (
    BlockType,
    EmbeddingMismatchError,
    EncryptionLevel,
    MemBlock,
    SnapshotError,
)
from memblock.embeddings import EmbeddingProvider
from memblock.snapshot import SNAPSHOT_FORMAT, embedding_identity, read_snapshot
from memblock.transactions import is_internal


class Vectors(EmbeddingProvider):
    """Deterministic vectors; `name` and `dims` stand in for different models."""

    def __init__(self, name="model-a", dims=8):
        self.model_name = name
        self._dims = dims

    def embed(self, texts):
        out = []
        for text in texts:
            vec = [0.0] * self._dims
            for word in text.lower().split():
                vec[sum(map(ord, word)) % self._dims] += 1.0
            out.append(vec)
        return out

    @property
    def dimensions(self):
        return self._dims


def uri(tmp_path, name):
    return f"sqlite:///{tmp_path / name}"


def contents(mem):
    return sorted(b.content for b in mem.query(limit=500))


@pytest.fixture
def source(tmp_path):
    mem = MemBlock(storage=uri(tmp_path, "source.db"))
    a = mem.store("User lives in Berlin", tags=["home"])
    b = mem.store("User works at Acme", session_id="s1")
    mem.store("User prefers tea", type=BlockType.PREFERENCE, session_id="s1")
    mem.link(a.id, b.id, relation="related_to")
    gone = mem.store("an old note")
    mem.delete(gone.id)
    yield mem
    mem.close()


class TestRoundTrip:
    def test_whole_store_into_an_empty_one(self, source, tmp_path):
        path = tmp_path / "snap.json"
        stamp = source.snapshot(str(path))
        assert stamp["format"] == SNAPSHOT_FORMAT
        assert stamp["counts"] == {"blocks": 4, "edges": 1, "embeddings": 0, "operations": 6}
        assert stamp["operations_head"] == source._storage.get_last_operation().hash

        target = MemBlock(storage=uri(tmp_path, "target.db"))
        result = target.restore(str(path))
        assert result["blocks"] == 4 and result["edges"] == 1 and result["history_kept"]
        assert contents(target) == contents(source)
        assert target.verify().valid
        assert target.stats()["total_operations"] == source.stats()["total_operations"]
        assert target.stats()["deleted_blocks"] == 1  # soft-deleted stays soft-deleted
        berlin = target.query(text_search="Berlin", limit=1)[0]
        assert berlin.tags == ["home"]
        assert len(target._storage.get_edges(berlin.id, direction="outgoing")) == 1
        # the copy keeps working: a new write extends the restored history
        target.store("a new fact")
        assert target.verify().valid
        target.close()

    def test_empty_mode_refuses_a_store_with_blocks(self, source, tmp_path):
        path = str(tmp_path / "snap.json")
        source.snapshot(path)
        target = MemBlock(storage=uri(tmp_path, "target.db"))
        target.store("already here")
        with pytest.raises(SnapshotError, match="already holds"):
            target.restore(path)
        target.close()

    def test_merge_adds_only_what_is_missing(self, source, tmp_path):
        path = str(tmp_path / "snap.json")
        source.snapshot(path)
        target = MemBlock(storage=uri(tmp_path, "target.db"))
        target.store("already here")
        first = target.restore(path, mode="merge")
        assert first["blocks"] == 4 and first["skipped"] == 0 and not first["history_kept"]
        second = target.restore(path, mode="merge")
        assert second["blocks"] == 0 and second["skipped"] == 4
        assert contents(target) == sorted(contents(source) + ["already here"])
        assert target.verify().valid  # merged blocks are logged in the target's own history
        target.close()

    def test_one_session_only(self, source, tmp_path):
        path = str(tmp_path / "s1.json")
        stamp = source.snapshot(path, session_id="s1")
        assert stamp["counts"]["blocks"] == 2 and stamp["counts"]["operations"] == 0
        assert stamp["scope"] == {"session_id": "s1"}
        target = MemBlock(storage=uri(tmp_path, "target.db"))
        target.restore(path)
        assert contents(target) == ["User prefers tea", "User works at Acme"]
        assert target.verify().valid
        target.close()


class TestRefusals:
    def test_edited_file_fails_its_checksum(self, source, tmp_path):
        path = tmp_path / "snap.json"
        source.snapshot(str(path))
        document = json.loads(path.read_text())
        document["blocks"][0]["content"] = "tampered"
        path.write_text(json.dumps(document))
        with pytest.raises(SnapshotError, match="checksum"):
            read_snapshot(str(path))
        target = MemBlock(storage=uri(tmp_path, "target.db"))
        with pytest.raises(SnapshotError, match="checksum"):
            target.restore(str(path))
        assert contents(target) == []
        target.close()

    def test_wrong_format_and_missing_file(self, tmp_path):
        target = MemBlock(storage=uri(tmp_path, "target.db"))
        other = tmp_path / "other.json"
        other.write_text(json.dumps({"stamp": {"format": "something-else"}}))
        with pytest.raises(SnapshotError, match="Not a"):
            target.restore(str(other))
        garbage = tmp_path / "garbage.json"
        garbage.write_text("not json at all")
        with pytest.raises(SnapshotError):
            target.restore(str(garbage))
        with pytest.raises(SnapshotError, match="not found"):
            target.restore(str(tmp_path / "missing.json"))
        target.close()

    def test_existing_file_is_not_replaced_unless_asked(self, source, tmp_path):
        path = tmp_path / "snap.json"
        source.snapshot(str(path))
        before = path.read_text()
        with pytest.raises(SnapshotError, match="already exists"):
            source.snapshot(str(path))
        assert path.read_text() == before
        source.store("one more")
        stamp = source.snapshot(str(path), overwrite=True)
        assert stamp["counts"]["blocks"] == 5
        # no temporary file is left next to the snapshot
        assert not [n for n in os.listdir(tmp_path) if n.startswith(".memblock-snapshot-")]

    def test_not_while_a_transaction_is_open(self, source, tmp_path):
        with source.transaction():
            source.store("staged")
            with pytest.raises(SnapshotError):
                source.snapshot(str(tmp_path / "snap.json"))

    def test_staged_and_internal_records_are_never_included(self, tmp_path):
        mem = MemBlock(storage=uri(tmp_path, "a.db"), embeddings=Vectors())
        mem.store("visible")  # also writes the embedding manifest record

        class Stop(BaseException):
            pass

        def hook(point):
            if point == "before_commit_point":
                raise Stop

        mem._txn_fault_hook = hook
        with pytest.raises(Stop):
            with mem.transaction():
                mem.store("staged, never published")
        mem._txn_local.txn = None  # the writer is gone; its rows remain
        document = read_snapshot(mem.snapshot(str(tmp_path / "snap.json"))["path"])
        assert [b["content"] for b in document["blocks"]] == ["visible"]
        mem.close()


class TestEncryption:
    def test_sealed_content_stays_sealed_in_the_file(self, tmp_path):
        mem = MemBlock(storage=uri(tmp_path, "a.db"), encryption_key="secret")
        mem.store("card ends 4242", encryption_level=EncryptionLevel.SENSITIVE)
        path = tmp_path / "snap.json"
        stamp = mem.snapshot(str(path))
        assert stamp["encrypted_blocks"] == 1
        assert "4242" not in path.read_text()

        same_key = MemBlock(storage=uri(tmp_path, "b.db"), encryption_key="secret")
        same_key.restore(str(path))
        block = same_key._storage.get_all_blocks()[0]
        assert same_key.get(block.id).content == "card ends 4242"
        mem.close()
        same_key.close()


class TestEmbeddingIdentity:
    def test_identity_string(self):
        assert embedding_identity(None) is None
        assert embedding_identity(Vectors("model-a", 8)) == "Vectors:model-a:8"
        assert embedding_identity(Vectors("model-b", 8)) != embedding_identity(Vectors("model-a", 8))

    def test_snapshot_from_another_model_is_refused(self, tmp_path):
        a = MemBlock(storage=uri(tmp_path, "a.db"), embeddings=Vectors("model-a"))
        a.store("zebra crossing rules")
        path = str(tmp_path / "snap.json")
        stamp = a.snapshot(path)
        assert stamp["embedding"] == "Vectors:model-a:8" and stamp["counts"]["embeddings"] == 1

        b = MemBlock(storage=uri(tmp_path, "b.db"), embeddings=Vectors("model-b"))
        with pytest.raises(SnapshotError, match="model-a"):
            b.restore(path)
        assert contents(b) == []

        dropped = b.restore(path, embeddings="drop")
        assert dropped["blocks"] == 1 and dropped["embeddings"] == 0
        assert b.stats()["total_embeddings"] == 0

        c = MemBlock(storage=uri(tmp_path, "c.db"), embeddings=Vectors("model-b"))
        rebuilt = c.restore(path, embeddings="reembed")
        assert rebuilt["embeddings"] == 1 and c.stats()["total_embeddings"] == 1

        same = MemBlock(storage=uri(tmp_path, "d.db"), embeddings=Vectors("model-a"))
        kept = same.restore(path)
        assert kept["embeddings"] == 1
        for mem in (a, b, c, same):
            mem.close()

    def test_store_without_embeddings_refuses_foreign_vectors(self, tmp_path):
        a = MemBlock(storage=uri(tmp_path, "a.db"), embeddings=Vectors())
        a.store("something")
        path = str(tmp_path / "snap.json")
        a.snapshot(path)
        plain = MemBlock(storage=uri(tmp_path, "b.db"))
        with pytest.raises(SnapshotError):
            plain.restore(path)
        assert plain.restore(path, embeddings="drop")["blocks"] == 1
        a.close()
        plain.close()

    def test_opening_a_store_with_another_model_is_caught(self, tmp_path):
        db = uri(tmp_path, "a.db")
        first = MemBlock(storage=db, embeddings=Vectors("model-a"))
        first.store("zebra crossing rules")
        first.close()

        other = MemBlock(storage=db, embeddings=Vectors("model-b"))
        with pytest.raises(EmbeddingMismatchError, match="model-a"):
            other.store("another fact")
        with pytest.raises(EmbeddingMismatchError):
            other.query(text_search="zebra", limit=3)
        assert other.query(text_search="zebra", semantic=False, limit=3)  # keyword search is unaffected
        other.close()

        same = MemBlock(storage=db, embeddings=Vectors("model-a"))
        same.store("another fact")
        same.close()

    def test_warn_and_ignore_then_reembed(self, tmp_path):
        db = uri(tmp_path, "a.db")
        first = MemBlock(storage=db, embeddings=Vectors("model-a"))
        first.store("zebra crossing rules")
        first.close()

        warned = MemBlock(storage=db, embeddings=Vectors("model-b"), on_embedding_mismatch="warn")
        with pytest.warns(UserWarning, match="model-a"):
            warned.store("stored with a warning")
        warned.close()

        fixing = MemBlock(storage=db, embeddings=Vectors("model-b"), on_embedding_mismatch="ignore")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert fixing.reembed() == 2
        fixing.close()

        after = MemBlock(storage=db, embeddings=Vectors("model-b"))
        after.store("no complaint now")
        assert after.embedding_identity() == "Vectors:model-b:8"
        after.close()

    def test_bad_option(self, tmp_path):
        with pytest.raises(ValueError):
            MemBlock(storage=uri(tmp_path, "a.db"), on_embedding_mismatch="maybe")

    def test_manifest_record_is_hidden(self, tmp_path):
        mem = MemBlock(storage=uri(tmp_path, "a.db"), embeddings=Vectors())
        mem.store("only this")
        assert [b.content for b in mem.introspect_user(include_deleted=True)] == ["only this"]
        assert mem.stats()["total_blocks"] == 1 and mem.stats()["deleted_blocks"] == 0
        hidden = [b for b in mem._storage.get_all_blocks(include_deleted=True) if is_internal(b)]
        assert len(hidden) == 1
        mem.close()


class TestFork:
    def test_fork_is_an_independent_copy(self, source, tmp_path):
        copy = source.fork(uri(tmp_path, "copy.db"))
        assert contents(copy) == contents(source)
        assert copy.verify().valid
        copy.store("only in the copy")
        source.store("only in the source")
        assert "only in the copy" not in contents(source)
        assert "only in the source" not in contents(copy)
        copy.close()

    def test_fork_carries_the_embedding_model(self, tmp_path):
        mem = MemBlock(storage=uri(tmp_path, "a.db"), embeddings=Vectors("model-a"))
        mem.store("zebra crossing rules")
        copy = mem.fork(uri(tmp_path, "copy.db"))
        assert copy.stats()["total_embeddings"] == 1
        assert [b.content for b in copy.query(text_search="zebra crossing", limit=1)] == ["zebra crossing rules"]
        mem.close()
        copy.close()

    def test_fork_into_a_used_store_fails_cleanly(self, source, tmp_path):
        used = MemBlock(storage=uri(tmp_path, "used.db"))
        used.store("occupied")
        used.close()
        with pytest.raises(SnapshotError):
            source.fork(uri(tmp_path, "used.db"))
