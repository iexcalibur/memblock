"""Tests for the MemBlock license key system."""

from __future__ import annotations

import base64
import json
import os
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from memblock.errors import LicenseError
from memblock.licensing import (
    LICENSE_FILE,
    LicenseInfo,
    generate_license,
    get_secret,
    get_stored_license,
    store_license,
    validate_license,
)


TEST_SECRET = "test-secret-key-for-memblock"


# ─── Key Generation ───────────────────────────────────────────────────────


class TestGenerateLicense:
    def test_generates_valid_base64_key(self):
        key = generate_license("Acme Corp", TEST_SECRET, expires_days=365)
        # Should be valid base64
        raw = base64.urlsafe_b64decode(key)
        payload = json.loads(raw)
        assert payload["customer"] == "Acme Corp"
        assert payload["id"].startswith("lic_")
        assert payload["signature"]
        assert payload["issued_at"] == date.today().isoformat()

    def test_perpetual_license(self):
        key = generate_license("Acme Corp", TEST_SECRET, expires_days=None)
        raw = base64.urlsafe_b64decode(key)
        payload = json.loads(raw)
        assert payload["expires_at"] is None

    def test_custom_expiry(self):
        key = generate_license("Acme Corp", TEST_SECRET, expires_days=30)
        raw = base64.urlsafe_b64decode(key)
        payload = json.loads(raw)
        expected = (date.today() + timedelta(days=30)).isoformat()
        assert payload["expires_at"] == expected

    def test_empty_customer_raises(self):
        with pytest.raises(LicenseError, match="Customer name"):
            generate_license("", TEST_SECRET)

    def test_empty_secret_raises(self):
        with pytest.raises(LicenseError, match="Secret is required"):
            generate_license("Acme Corp", "")


# ─── Key Validation ───────────────────────────────────────────────────────


class TestValidateLicense:
    def test_valid_key_roundtrip(self):
        key = generate_license("Test Corp", TEST_SECRET, expires_days=365)
        info = validate_license(key, TEST_SECRET)
        assert info.customer == "Test Corp"
        assert info.id.startswith("lic_")
        assert info.issued_at == date.today()
        assert info.expires_at == date.today() + timedelta(days=365)

    def test_perpetual_key_roundtrip(self):
        key = generate_license("Test Corp", TEST_SECRET, expires_days=None)
        info = validate_license(key, TEST_SECRET)
        assert info.expires_at is None

    def test_tampered_key_fails(self):
        key = generate_license("Test Corp", TEST_SECRET)
        # Decode, tamper, re-encode
        raw = json.loads(base64.urlsafe_b64decode(key))
        raw["customer"] = "Evil Corp"
        tampered = base64.urlsafe_b64encode(
            json.dumps(raw, separators=(",", ":")).encode()
        ).decode()
        with pytest.raises(LicenseError, match="signature mismatch"):
            validate_license(tampered, TEST_SECRET)

    def test_wrong_secret_fails(self):
        key = generate_license("Test Corp", TEST_SECRET)
        with pytest.raises(LicenseError, match="signature mismatch"):
            validate_license(key, "wrong-secret")

    def test_expired_key_fails(self):
        # Generate a key, then manually set expiry to yesterday
        key = generate_license("Test Corp", TEST_SECRET, expires_days=365)
        raw = json.loads(base64.urlsafe_b64decode(key))
        raw["expires_at"] = (date.today() - timedelta(days=1)).isoformat()
        # Re-sign with correct secret
        from memblock.licensing import _sign

        raw["signature"] = _sign(raw, TEST_SECRET)
        expired_key = base64.urlsafe_b64encode(
            json.dumps(raw, separators=(",", ":")).encode()
        ).decode()
        with pytest.raises(LicenseError, match="expired"):
            validate_license(expired_key, TEST_SECRET)

    def test_empty_key_fails(self):
        with pytest.raises(LicenseError, match="Empty license key"):
            validate_license("", TEST_SECRET)

    def test_empty_secret_fails(self):
        key = generate_license("Test Corp", TEST_SECRET)
        with pytest.raises(LicenseError, match="MEMBLOCK_SECRET"):
            validate_license(key, "")

    def test_malformed_base64_fails(self):
        with pytest.raises(LicenseError, match="Malformed"):
            validate_license("not-valid-base64!!!", TEST_SECRET)

    def test_missing_field_fails(self):
        payload = {"id": "lic_123", "customer": "Test"}
        raw = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).decode()
        with pytest.raises(LicenseError, match="missing field"):
            validate_license(raw, TEST_SECRET)


# ─── File Storage ─────────────────────────────────────────────────────────


class TestLicenseStorage:
    def test_store_and_read(self, tmp_path: Path):
        lic_file = tmp_path / "license"
        with patch("memblock.licensing.LICENSE_DIR", tmp_path), \
             patch("memblock.licensing.LICENSE_FILE", lic_file):
            key = generate_license("Test Corp", TEST_SECRET)
            store_license(key)
            assert lic_file.exists()
            loaded = get_stored_license()
            assert loaded == key

    def test_no_stored_license(self, tmp_path: Path):
        lic_file = tmp_path / "license"
        with patch("memblock.licensing.LICENSE_FILE", lic_file):
            assert get_stored_license() is None

    def test_empty_license_file(self, tmp_path: Path):
        lic_file = tmp_path / "license"
        lic_file.write_text("   ")
        with patch("memblock.licensing.LICENSE_FILE", lic_file):
            assert get_stored_license() is None


# ─── Environment Variable ────────────────────────────────────────────────


class TestGetSecret:
    def test_reads_env_var(self):
        with patch.dict(os.environ, {"MEMBLOCK_SECRET": "my-secret"}):
            assert get_secret() == "my-secret"

    def test_missing_env_var(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_secret() is None


# ─── MemBlock Constructor Integration ────────────────────────────────────


class TestMemBlockLicenseCheck:
    """License enforcement is currently disabled (no paid tier yet).
    These tests verify the constructor works without license validation.
    Re-enable enforcement tests when paid tier is activated."""

    def test_constructor_works_without_secret(self):
        """Constructor works when no MEMBLOCK_SECRET is set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MEMBLOCK_SECRET", None)
            from memblock import MemBlock

            mem = MemBlock(storage="sqlite:///:memory:")
            assert mem._license is None
            mem.close()

    def test_constructor_works_with_secret_no_key(self):
        """Constructor works even when MEMBLOCK_SECRET is set but no key exists (enforcement disabled)."""
        with patch.dict(os.environ, {"MEMBLOCK_SECRET": TEST_SECRET}), \
             patch("memblock.memblock.get_stored_license", return_value=None):
            from memblock import MemBlock

            mem = MemBlock(storage="sqlite:///:memory:")
            assert mem._license is None
            mem.close()

    def test_constructor_works_with_any_key(self):
        """Constructor ignores license_key param (enforcement disabled)."""
        from memblock import MemBlock

        mem = MemBlock(storage="sqlite:///:memory:", license_key="anything")
        assert mem._license is None
        mem.close()


# ─── CLI Commands ─────────────────────────────────────────────────────────


class TestCLIActivate:
    def test_activate_valid_key(self, tmp_path: Path, capsys):
        key = generate_license("CLI Corp", TEST_SECRET)
        lic_file = tmp_path / "license"

        with patch("memblock.cli.get_secret", return_value=TEST_SECRET), \
             patch("memblock.licensing.LICENSE_DIR", tmp_path), \
             patch("memblock.licensing.LICENSE_FILE", lic_file):
            from memblock.cli import main

            rc = main(["activate", key, "--secret", TEST_SECRET])
            assert rc == 0
            out = capsys.readouterr().out
            assert "CLI Corp" in out
            assert "activated" in out.lower()

    def test_activate_invalid_key(self, capsys):
        from memblock.cli import main

        rc = main(["activate", "bad-key", "--secret", TEST_SECRET])
        assert rc == 1
        err = capsys.readouterr().err
        assert "Invalid" in err or "invalid" in err.lower() or "Malformed" in err

    def test_activate_no_secret(self, capsys):
        with patch("memblock.cli.get_secret", return_value=None):
            from memblock.cli import main

            rc = main(["activate", "some-key"])
            assert rc == 1
            err = capsys.readouterr().err
            assert "MEMBLOCK_SECRET" in err


class TestCLILicenseInfo:
    def test_info_shows_details(self, tmp_path: Path, capsys):
        key = generate_license("Info Corp", TEST_SECRET)
        lic_file = tmp_path / "license"
        lic_file.write_text(key)

        with patch("memblock.cli.get_secret", return_value=TEST_SECRET), \
             patch("memblock.licensing.LICENSE_FILE", lic_file):
            from memblock.cli import main

            rc = main(["license", "info", "--secret", TEST_SECRET])
            assert rc == 0
            out = capsys.readouterr().out
            assert "Info Corp" in out

    def test_info_no_license(self, tmp_path: Path, capsys):
        lic_file = tmp_path / "nonexistent"
        with patch("memblock.licensing.LICENSE_FILE", lic_file):
            from memblock.cli import main

            rc = main(["license", "info"])
            assert rc == 1


class TestCLILicenseGenerate:
    def test_generate_outputs_key(self, capsys):
        from memblock.cli import main

        rc = main(["license", "generate", "--customer", "Gen Corp", "--secret", TEST_SECRET])
        assert rc == 0
        key = capsys.readouterr().out.strip()
        # Validate the generated key
        info = validate_license(key, TEST_SECRET)
        assert info.customer == "Gen Corp"

    def test_generate_perpetual(self, capsys):
        from memblock.cli import main

        rc = main(["license", "generate", "--customer", "Perp Corp", "--secret", TEST_SECRET, "--days", "0"])
        assert rc == 0
        key = capsys.readouterr().out.strip()
        info = validate_license(key, TEST_SECRET)
        assert info.expires_at is None

    def test_generate_no_secret(self, capsys):
        with patch("memblock.cli.get_secret", return_value=None):
            from memblock.cli import main

            rc = main(["license", "generate", "--customer", "Test"])
            assert rc == 1
            err = capsys.readouterr().err
            assert "MEMBLOCK_SECRET" in err
