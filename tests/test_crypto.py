"""Tests for encryption layer."""

import pytest

from memblock.crypto import CryptoLayer, CryptoLayerWithPassphrase
from memblock.types import EncryptionLevel


class TestCryptoLayer:
    def test_disabled_when_no_key(self):
        crypto = CryptoLayer(key=None)
        assert crypto.enabled is False

    def test_seal_returns_plaintext_when_disabled(self):
        crypto = CryptoLayer(key=None)
        result = crypto.seal("hello world", EncryptionLevel.STANDARD)
        assert result == "hello world"

    def test_seal_returns_plaintext_for_none_level(self):
        crypto = CryptoLayerWithPassphrase(key="secret")
        result = crypto.seal("hello world", EncryptionLevel.NONE)
        assert result == "hello world"

    def test_seal_encrypts_content(self):
        crypto = CryptoLayerWithPassphrase(key="my-secret-key")
        encrypted = crypto.seal("sensitive data", EncryptionLevel.STANDARD)
        assert encrypted != "sensitive data"
        assert len(encrypted) > 0

    def test_seal_and_open_roundtrip(self):
        crypto = CryptoLayerWithPassphrase(key="my-secret-key")
        original = "This is sensitive memory content"

        encrypted = crypto.seal(original, EncryptionLevel.STANDARD)
        decrypted = crypto.open(encrypted, EncryptionLevel.STANDARD)

        assert decrypted == original

    def test_seal_different_each_time(self):
        crypto = CryptoLayerWithPassphrase(key="key123")
        e1 = crypto.seal("same content", EncryptionLevel.STANDARD)
        e2 = crypto.seal("same content", EncryptionLevel.STANDARD)
        # Different nonces → different ciphertext
        assert e1 != e2

    def test_open_none_level_returns_plaintext(self):
        crypto = CryptoLayerWithPassphrase(key="key")
        result = crypto.open("plaintext", EncryptionLevel.NONE)
        assert result == "plaintext"

    def test_sensitive_level_works(self):
        crypto = CryptoLayerWithPassphrase(key="key")
        original = "top secret tags and content"
        encrypted = crypto.seal(original, EncryptionLevel.SENSITIVE)
        decrypted = crypto.open(encrypted, EncryptionLevel.SENSITIVE)
        assert decrypted == original


class TestSignature:
    def test_sign_content(self):
        crypto = CryptoLayer()
        sig = crypto.sign("test content")
        assert sig.startswith("sha256:")

    def test_verify_valid_signature(self):
        crypto = CryptoLayer()
        content = "test content"
        sig = crypto.sign(content)
        assert crypto.verify_signature(content, sig) is True

    def test_verify_invalid_signature(self):
        crypto = CryptoLayer()
        sig = crypto.sign("original")
        assert crypto.verify_signature("tampered", sig) is False

    def test_deterministic_signatures(self):
        crypto = CryptoLayer()
        s1 = crypto.sign("same content")
        s2 = crypto.sign("same content")
        assert s1 == s2

    def test_different_content_different_signatures(self):
        crypto = CryptoLayer()
        s1 = crypto.sign("content A")
        s2 = crypto.sign("content B")
        assert s1 != s2
