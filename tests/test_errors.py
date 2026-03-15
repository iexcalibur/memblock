"""Tests for the error type hierarchy."""

import pytest

from memblock import (
    MemBlock,
    BlockType,
)
from memblock.errors import (
    BlockNotFoundError,
    DuplicateBlockError,
    EncryptionError,
    ExtractionError,
    MemBlockError,
    MigrationError,
    StorageError,
    ValidationError,
)
from memblock.schema import SchemaValidationError


class TestErrorHierarchy:
    """Test that the exception hierarchy is correct."""

    def test_all_errors_inherit_from_memblock_error(self):
        for exc_class in [
            StorageError,
            MigrationError,
            ValidationError,
            BlockNotFoundError,
            DuplicateBlockError,
            ExtractionError,
            EncryptionError,
        ]:
            assert issubclass(exc_class, MemBlockError)

    def test_schema_validation_error_is_validation_error(self):
        assert issubclass(SchemaValidationError, ValidationError)
        assert issubclass(SchemaValidationError, MemBlockError)

    def test_schema_validation_error_catchable_by_old_name(self):
        with pytest.raises(SchemaValidationError):
            raise SchemaValidationError("test")

    def test_schema_validation_error_catchable_as_validation_error(self):
        with pytest.raises(ValidationError):
            raise SchemaValidationError("test")

    def test_schema_validation_error_catchable_as_memblock_error(self):
        with pytest.raises(MemBlockError):
            raise SchemaValidationError("test")

    def test_each_error_can_be_raised_and_caught(self):
        for exc_class in [
            StorageError,
            MigrationError,
            ValidationError,
            BlockNotFoundError,
            DuplicateBlockError,
            ExtractionError,
            EncryptionError,
        ]:
            with pytest.raises(exc_class):
                raise exc_class("test message")

    def test_error_messages_preserved(self):
        err = BlockNotFoundError("Block blk_123 not found")
        assert "blk_123" in str(err)


class TestErrorsInFacade:
    """Test that facade methods raise appropriate error types."""

    def test_get_or_raise_missing_block(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        with pytest.raises(BlockNotFoundError):
            mem._store.get_or_raise("nonexistent_id")
        mem.close()

    def test_extract_without_api_key_raises_extraction_error(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        with pytest.raises(ExtractionError, match="api_key"):
            mem.extract("some conversation")
        mem.close()

    def test_extract_messages_without_api_key_raises_extraction_error(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        with pytest.raises(ExtractionError, match="api_key"):
            mem.extract_messages([{"role": "user", "content": "hello"}])
        mem.close()

    def test_extract_unknown_provider_raises_extraction_error(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        with pytest.raises(ExtractionError, match="Unknown provider"):
            mem.extract("test", provider="invalid", api_key="key")
        mem.close()

    def test_invalid_block_raises_schema_validation_error(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        with pytest.raises(SchemaValidationError):
            mem.store(content="", type=BlockType.FACT)
        mem.close()


class TestErrorsImportable:
    """Test that errors are importable from the top-level package."""

    def test_import_from_memblock(self):
        from memblock import (
            MemBlockError,
            StorageError,
            MigrationError,
            ValidationError,
            BlockNotFoundError,
            DuplicateBlockError,
            ExtractionError,
            EncryptionError,
            SchemaValidationError,
        )
        # Just verify they're the right classes
        assert issubclass(SchemaValidationError, ValidationError)
