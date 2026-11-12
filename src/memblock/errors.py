"""MemBlock error hierarchy — stable exception types for the SDK."""

from __future__ import annotations


class MemBlockError(Exception):
    """Base exception for all MemBlock errors."""
    pass


class StorageError(MemBlockError):
    """Raised when a storage operation fails (DB connection, query, etc.)."""
    pass


class MigrationError(MemBlockError):
    """Raised when a schema migration fails or encounters an incompatible version."""
    pass


class ValidationError(MemBlockError):
    """Raised when a block or edge fails validation."""
    pass


class BlockNotFoundError(MemBlockError):
    """Raised when a requested block does not exist."""
    pass


class DuplicateBlockError(MemBlockError):
    """Raised when attempting to store a duplicate block (with ERROR policy)."""
    pass


class ExtractionError(MemBlockError):
    """Raised when LLM extraction fails."""
    pass


class EncryptionError(MemBlockError):
    """Raised when encryption or decryption fails."""
    pass


class LicenseError(MemBlockError):
    """Raised when license validation fails (missing, expired, or tampered)."""
    pass


class ConflictResolutionError(MemBlockError):
    """Raised when LLM-based conflict resolution fails."""
    pass


class AnalyticsError(MemBlockError):
    """Raised when analytics operations fail or analytics is not enabled."""
    pass


class TransactionError(MemBlockError):
    """Raised when a grouped write cannot be published, aborted or recovered."""
    pass


class SnapshotError(MemBlockError):
    """Raised when a snapshot cannot be written, or is refused on restore
    (wrong format, failed checksum, or an embedding model that does not match)."""
    pass


class EmbeddingMismatchError(MemBlockError):
    """Raised when stored vectors were made by a different embedding model
    than the one this instance is configured with."""
    pass


class MemorySessionError(MemBlockError):
    """Raised by `MemorySession`. `uncertain` means a commit may or may not have
    been applied: call `recover()` before writing again. `answer` carries the
    model's reply when the failure happened after generation."""

    def __init__(
        self,
        message: str,
        *,
        stage: str | None = None,
        uncertain: bool = False,
        answer: object = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.uncertain = uncertain
        self.answer = answer
