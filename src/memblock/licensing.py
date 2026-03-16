"""MemBlock licensing — HMAC-signed offline license key generation & validation."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from memblock.errors import LicenseError


# ─── Constants ────────────────────────────────────────────────────────────

LICENSE_DIR = Path.home() / ".memblock"
LICENSE_FILE = LICENSE_DIR / "license"
SECRET_ENV_VAR = "MEMBLOCK_SECRET"


# ─── Data ─────────────────────────────────────────────────────────────────


@dataclass
class LicenseInfo:
    """Decoded & validated license information."""

    id: str
    customer: str
    issued_at: date
    expires_at: date | None  # None = perpetual
    tier: str = "community"  # "community" or "pro"


# ─── Key Generation ───────────────────────────────────────────────────────


def _sign(payload: dict, secret: str) -> str:
    """Compute HMAC-SHA256 signature over the canonical payload fields."""
    tier = payload.get("tier", "community")
    msg = f"{payload['id']}|{payload['customer']}|{payload['issued_at']}|{payload['expires_at']}|{tier}"
    return hmac.new(
        secret.encode("utf-8"),
        msg.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def generate_license(
    customer: str,
    secret: str,
    expires_days: int | None = 365,
    tier: str = "community",
) -> str:
    """
    Generate a signed license key.

    Args:
        customer: Customer name / company.
        secret: The HMAC shared secret.
        expires_days: Days until expiry (None = perpetual).
        tier: License tier — "community" (free) or "pro" (paid).

    Returns:
        Base64-encoded license key string.
    """
    if not customer:
        raise LicenseError("Customer name is required")
    if not secret:
        raise LicenseError("Secret is required to generate a license")

    today = date.today()
    expires_at: str | None = None
    if expires_days is not None:
        from datetime import timedelta

        expires_at = (today + timedelta(days=expires_days)).isoformat()

    payload: dict = {
        "id": f"lic_{uuid.uuid4().hex[:12]}",
        "customer": customer,
        "issued_at": today.isoformat(),
        "expires_at": expires_at,
        "tier": tier,
    }

    payload["signature"] = _sign(payload, secret)

    raw = json.dumps(payload, separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")


# ─── Key Validation ───────────────────────────────────────────────────────


def validate_license(key_str: str, secret: str) -> LicenseInfo:
    """
    Validate a license key string.

    Args:
        key_str: Base64-encoded license key.
        secret: The HMAC shared secret.

    Returns:
        LicenseInfo on success.

    Raises:
        LicenseError: If key is invalid, tampered, or expired.
    """
    if not key_str or not key_str.strip():
        raise LicenseError("Empty license key")
    if not secret:
        raise LicenseError(
            "MEMBLOCK_SECRET environment variable is required for license validation"
        )

    # Decode
    try:
        raw = base64.urlsafe_b64decode(key_str.strip()).decode("utf-8")
        payload = json.loads(raw)
    except Exception as e:
        raise LicenseError(f"Malformed license key: {e}") from e

    # Check required fields
    for field in ("id", "customer", "issued_at", "expires_at", "signature"):
        if field not in payload:
            raise LicenseError(f"License key missing field: {field}")

    # Verify HMAC signature
    expected_sig = _sign(payload, secret)
    if not hmac.compare_digest(payload["signature"], expected_sig):
        raise LicenseError("Invalid license key (signature mismatch)")

    # Check expiry
    expires_at: date | None = None
    if payload["expires_at"] is not None:
        try:
            expires_at = date.fromisoformat(payload["expires_at"])
        except ValueError as e:
            raise LicenseError(f"Invalid expiry date: {e}") from e

        if expires_at < date.today():
            raise LicenseError(
                f"License expired on {expires_at.isoformat()}. "
                "Contact the author for a renewal."
            )

    issued_at = date.fromisoformat(payload["issued_at"])

    return LicenseInfo(
        id=payload["id"],
        customer=payload["customer"],
        issued_at=issued_at,
        expires_at=expires_at,
        tier=payload.get("tier", "community"),
    )


# ─── File Storage ─────────────────────────────────────────────────────────


def store_license(key_str: str) -> Path:
    """
    Write a license key to ~/.memblock/license.

    Returns:
        Path to the stored license file.
    """
    LICENSE_DIR.mkdir(parents=True, exist_ok=True)
    LICENSE_FILE.write_text(key_str.strip(), encoding="utf-8")
    return LICENSE_FILE


def get_stored_license() -> str | None:
    """
    Read the stored license key from ~/.memblock/license.

    Returns:
        The key string, or None if no license file exists.
    """
    if not LICENSE_FILE.exists():
        return None
    content = LICENSE_FILE.read_text(encoding="utf-8").strip()
    return content if content else None


def get_secret() -> str | None:
    """Read the HMAC secret from the MEMBLOCK_SECRET env var."""
    return os.environ.get(SECRET_ENV_VAR)
