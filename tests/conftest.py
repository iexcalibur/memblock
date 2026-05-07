"""Shared test fixtures.

Native-async Postgres tests need a live database and gracefully skip
when one isn't available. Set `MEMBLOCK_TEST_DB_URL` to point at a
Postgres you're willing to have tables created in.

Example:
    export MEMBLOCK_TEST_DB_URL=postgresql://shubham:@localhost:5432/memblock_test
    pytest tests/test_async_native.py
"""

from __future__ import annotations

import os
import uuid

import pytest


# ─── Live-DB gating ──────────────────────────────────────────────────


def _detect_postgres_url() -> str | None:
    """Return a Postgres URL if one is configured, else None.

    Priority:
      1. `MEMBLOCK_TEST_DB_URL` (explicit opt-in for CI / local)
      2. None — tests that need a DB will skip
    """
    return os.environ.get("MEMBLOCK_TEST_DB_URL")


@pytest.fixture(scope="session")
def postgres_sync_url() -> str:
    """Sync `postgresql://` URL or skip the test."""
    url = _detect_postgres_url()
    if not url:
        pytest.skip("MEMBLOCK_TEST_DB_URL not set — live Postgres test skipped")
    # Strip any +asyncpg driver suffix the user passed.
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


@pytest.fixture(scope="session")
def postgres_async_url(postgres_sync_url: str) -> str:
    """Native-async `postgresql+asyncpg://` URL.

    Constructed from `postgres_sync_url` so a single env var configures
    both. Skips inheriting from `postgres_sync_url`'s skip.
    """
    return postgres_sync_url.replace("postgresql://", "postgresql+asyncpg://", 1)


# ─── Schema isolation per test ───────────────────────────────────────


@pytest.fixture
async def fresh_schema(postgres_sync_url: str):
    """Yields a unique Postgres schema name; drops it after the test.

    Lets every test run in its own pristine namespace so tests don't
    pollute each other or `public`.
    """
    import asyncpg

    schema = f"mb_test_{uuid.uuid4().hex[:10]}"
    yield schema

    # Cleanup — best-effort; if asyncpg connect fails we just leak the
    # schema and let the next dev rerun GC it.
    try:
        conn = await asyncpg.connect(postgres_sync_url)
        try:
            await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await conn.close()
    except Exception:
        pass


@pytest.fixture
async def mb_native(postgres_async_url: str, fresh_schema: str):
    """Yields an `AsyncMemBlock` in native-async mode against a unique
    schema. Closes the pool after the test.

    Uses `embeddings=False, extract=False` for fast tests — individual
    tests that need embeddings construct their own instance.
    """
    from memblock import AsyncMemBlock

    mb = AsyncMemBlock(
        storage=postgres_async_url,
        embeddings=False,
        extract=False,
        schema=fresh_schema,
    )
    yield mb

    # Close the asyncpg pool cleanly.
    if hasattr(mb, "close"):
        try:
            await mb.close()
        except Exception:
            pass
