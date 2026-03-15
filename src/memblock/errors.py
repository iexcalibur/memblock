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
