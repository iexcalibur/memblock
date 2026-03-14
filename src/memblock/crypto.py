"""Encryption layer — per-block AES-256-GCM encryption with key derivation."""

from __future__ import annotations

import base64
import hashlib
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

from memblock.types import EncryptionLevel


class CryptoLayer:
    """
    Per-block encryption using AES-256-GCM.

    - Key derived from user passphrase via PBKDF2
    - Each seal() uses a unique nonce (12 bytes)
    - Supports encryption levels: NONE, STANDARD, SENSITIVE
    """

    NONCE_SIZE = 12  # 96 bits for GCM
    KEY_SIZE = 32  # 256 bits
    SALT_SIZE = 16
    ITERATIONS = 480_000  # OWASP recommendation

    def __init__(self, key: str | None = None) -> None:
        """
        Initialize with an encryption key (passphrase).

        If key is None, encryption is disabled — seal() returns plaintext.
        """
        self._enabled = key is not None
        self._aesgcm: AESGCM | None = None
        self._salt: bytes = b""

        if key:
            self._salt = os.urandom(self.SALT_SIZE)
            derived_key = self._derive_key(key, self._salt)
            self._aesgcm = AESGCM(derived_key)

    def _derive_key(self, passphrase: str, salt: bytes) -> bytes:
        """Derive a 256-bit key from passphrase using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.KEY_SIZE,
            salt=salt,
            iterations=self.ITERATIONS,
        )
        return kdf.derive(passphrase.encode("utf-8"))

    @property
    def enabled(self) -> bool:
        return self._enabled

    def seal(self, content: str, level: EncryptionLevel = EncryptionLevel.STANDARD) -> str:
        """
        Encrypt content based on encryption level.

        Returns base64-encoded string: salt + nonce + ciphertext
        If level is NONE or encryption is disabled, returns content as-is.
        """
        if level == EncryptionLevel.NONE or not self._enabled:
            return content

        nonce = os.urandom(self.NONCE_SIZE)
        ciphertext = self._aesgcm.encrypt(nonce, content.encode("utf-8"), None)

        # Pack: salt(16) + nonce(12) + ciphertext
        packed = self._salt + nonce + ciphertext
        return base64.b64encode(packed).decode("ascii")

    def open(self, encrypted_content: str, level: EncryptionLevel = EncryptionLevel.STANDARD) -> str:
        """
        Decrypt content.

        If level is NONE or encryption is disabled, returns content as-is.
        """
        if level == EncryptionLevel.NONE or not self._enabled:
            return encrypted_content

        packed = base64.b64decode(encrypted_content.encode("ascii"))

        salt = packed[: self.SALT_SIZE]
        nonce = packed[self.SALT_SIZE : self.SALT_SIZE + self.NONCE_SIZE]
        ciphertext = packed[self.SALT_SIZE + self.NONCE_SIZE :]

        # Re-derive key from stored salt (in case different from init salt)
        derived_key = self._derive_key(self._get_passphrase(), salt)
        aesgcm = AESGCM(derived_key)

        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode("utf-8")

    def _get_passphrase(self) -> str:
        """Get the original passphrase. Stored internally for re-derivation."""
        # For simplicity, we store the passphrase reference
        # In production, this would use a key management system
        raise NotImplementedError("Use CryptoLayerWithPassphrase for decryption")

    def sign(self, content: str) -> str:
        """Create a SHA-256 signature of content."""
        return f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"

    def verify_signature(self, content: str, signature: str) -> bool:
        """Verify a content signature."""
        expected = self.sign(content)
        return expected == signature


class CryptoLayerWithPassphrase(CryptoLayer):
    """
    Extended CryptoLayer that stores the passphrase for decryption.

    In production, you'd use a key vault. For the SDK, this is acceptable
    since the passphrase lives in memory only during the session.
    """

    def __init__(self, key: str | None = None) -> None:
        self._passphrase = key or ""
        super().__init__(key)

    def _get_passphrase(self) -> str:
        return self._passphrase

    def open(self, encrypted_content: str, level: EncryptionLevel = EncryptionLevel.STANDARD) -> str:
        """Decrypt content using stored passphrase."""
        if level == EncryptionLevel.NONE or not self._enabled:
            return encrypted_content

        packed = base64.b64decode(encrypted_content.encode("ascii"))

        salt = packed[: self.SALT_SIZE]
        nonce = packed[self.SALT_SIZE : self.SALT_SIZE + self.NONCE_SIZE]
        ciphertext = packed[self.SALT_SIZE + self.NONCE_SIZE :]

        derived_key = self._derive_key(self._passphrase, salt)
        aesgcm = AESGCM(derived_key)

        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode("utf-8")
