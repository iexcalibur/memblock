"""Tests for the deduplication system."""

import pytest

from memblock import MemBlock, BlockType, DuplicatePolicy
from memblock.dedup import ContentHasher
from memblock.errors import DuplicateBlockError


class TestContentHasher:
    """Test content normalization and hashing."""

    def test_identical_content_same_hash(self):
        h1 = ContentHasher.hash("Hello World")
        h2 = ContentHasher.hash("Hello World")
        assert h1 == h2

    def test_whitespace_normalization(self):
        h1 = ContentHasher.hash("hello  world")
        h2 = ContentHasher.hash("hello world")
        assert h1 == h2

    def test_case_normalization(self):
        h1 = ContentHasher.hash("Hello World")
        h2 = ContentHasher.hash("hello world")
        assert h1 == h2

    def test_strip_normalization(self):
        h1 = ContentHasher.hash("  hello world  ")
        h2 = ContentHasher.hash("hello world")
        assert h1 == h2

    def test_unicode_normalization(self):
        # NFC vs NFD forms of é
        h1 = ContentHasher.hash("caf\u00e9")  # NFC
        h2 = ContentHasher.hash("cafe\u0301")  # NFD
        assert h1 == h2

    def test_different_content_different_hash(self):
        h1 = ContentHasher.hash("hello")
        h2 = ContentHasher.hash("world")
        assert h1 != h2

    def test_hash_is_sha256_hex(self):
        h = ContentHasher.hash("test")
        assert len(h) == 64  # SHA-256 hex is 64 chars
        assert all(c in "0123456789abcdef" for c in h)


class TestDuplicatePolicyError:
    """Test ERROR policy — raises DuplicateBlockError."""

    def test_exact_duplicate_raises(self):
        mem = MemBlock(storage="sqlite:///:memory:", on_duplicate="error")
        mem.store("User prefers Python", type=BlockType.PREFERENCE)
        with pytest.raises(DuplicateBlockError, match="Duplicate content"):
            mem.store("User prefers Python", type=BlockType.PREFERENCE)
        mem.close()

    def test_normalized_duplicate_raises(self):
        mem = MemBlock(storage="sqlite:///:memory:", on_duplicate="error")
        mem.store("User prefers Python", type=BlockType.PREFERENCE)
        with pytest.raises(DuplicateBlockError):
            mem.store("  user  prefers  python  ", type=BlockType.PREFERENCE)
        mem.close()

    def test_different_content_allowed(self):
        mem = MemBlock(storage="sqlite:///:memory:", on_duplicate="error")
        mem.store("User prefers Python", type=BlockType.PREFERENCE)
        block2 = mem.store("User prefers Rust", type=BlockType.PREFERENCE)
        assert block2 is not None
        mem.close()


class TestDuplicatePolicySkip:
    """Test SKIP policy — returns None silently."""

    def test_duplicate_returns_none(self):
        mem = MemBlock(storage="sqlite:///:memory:", on_duplicate="skip")
        mem.store("User prefers Python", type=BlockType.PREFERENCE)
        result = mem.store("User prefers Python", type=BlockType.PREFERENCE)
        assert result is None
        mem.close()


class TestDuplicatePolicyReturnExisting:
    """Test RETURN_EXISTING policy — returns the original block."""

    def test_duplicate_returns_existing(self):
        mem = MemBlock(storage="sqlite:///:memory:", on_duplicate="return_existing")
        original = mem.store("User prefers Python", type=BlockType.PREFERENCE)
        duplicate = mem.store("User prefers Python", type=BlockType.PREFERENCE)
        assert duplicate.id == original.id
        mem.close()


class TestDuplicatePolicyMerge:
    """Test MERGE policy — merges tags and confidence."""

    def test_merge_combines_tags(self):
        mem = MemBlock(storage="sqlite:///:memory:", on_duplicate="merge")
        mem.store("User prefers Python", type=BlockType.PREFERENCE, tags=["lang"])
        result = mem.store(
            "User prefers Python", type=BlockType.PREFERENCE, tags=["programming"]
        )
        assert "lang" in result.tags
        assert "programming" in result.tags
        mem.close()

    def test_merge_takes_higher_confidence(self):
        mem = MemBlock(storage="sqlite:///:memory:", on_duplicate="merge")
        mem.store("User prefers Python", type=BlockType.PREFERENCE, confidence=0.5)
        result = mem.store(
            "User prefers Python", type=BlockType.PREFERENCE, confidence=0.9
        )
        assert result.metadata.confidence == 0.9
        mem.close()


class TestNoDuplicatePolicy:
    """Test that on_duplicate=None (default) skips all checks."""

    def test_duplicates_allowed_by_default(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        b1 = mem.store("User prefers Python", type=BlockType.PREFERENCE)
        b2 = mem.store("User prefers Python", type=BlockType.PREFERENCE)
        assert b1.id != b2.id  # Two separate blocks created
        mem.close()


class TestContentHashPersistence:
    """Test that content_hash is stored and retrievable."""

    def test_block_has_content_hash(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        block = mem.store("Test content", type=BlockType.FACT)
        assert block.content_hash != ""
        assert len(block.content_hash) == 64
        mem.close()

    def test_content_hash_matches_hasher(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        block = mem.store("Test content", type=BlockType.FACT)
        expected = ContentHasher.hash("Test content")
        assert block.content_hash == expected
        mem.close()

    def test_content_hash_persisted_in_db(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        block = mem.store("Test content", type=BlockType.FACT)
        retrieved = mem.get(block.id)
        assert retrieved.content_hash == block.content_hash
        mem.close()
