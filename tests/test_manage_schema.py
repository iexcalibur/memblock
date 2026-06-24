"""Tests for the ``manage_schema`` flag.

``manage_schema=False`` lets memblock run against a schema that was
provisioned out of band (see ``sql/``), under a least-privilege DML-only
database role. The contract is strict: when it is False, the adapter must
emit ZERO DDL (no CREATE / ALTER / DROP / TRUNCATE) on any code path —
``initialize()``, the migration runner, and the lazy pgvector index.

The unit tests below prove that invariant deterministically with spy
connections (no database required). A gated live-DB integration test
proves it end-to-end: a ``manage_schema=False`` client does full CRUD
against a pre-provisioned schema while adding zero schema objects.
"""

from __future__ import annotations

import re

import pytest

from memblock.storage.postgresql import HAS_PSYCOPG, PostgreSQLAdapter
from memblock.storage.async_postgresql import HAS_ASYNCPG, AsyncPostgreSQLAdapter


# Anything that changes schema shape. The app role must never issue these.
_DDL = re.compile(r"\b(CREATE|ALTER|DROP|TRUNCATE)\b", re.IGNORECASE)


def _ddl_statements(log: list[str]) -> list[str]:
    return [s for s in log if _DDL.search(s)]


# ─── Sync spy connection (no real psycopg needed) ────────────────────


class _SyncSpyCursor:
    """Records every executed SQL string. Returns just enough for the
    sync MigrationRunner to believe the schema is already at the latest
    version, so the manage_schema=True contrast path runs without a DB."""

    def __init__(self, log: list[str]) -> None:
        self.log = log
        self._last = ""

    def __enter__(self) -> "_SyncSpyCursor":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, sql: str, params: object = None) -> None:
        self._last = sql
        self.log.append(sql)

    def fetchone(self) -> object:
        s = self._last.lower()
        if "pg_extension" in s:
            return None  # pgvector not installed in this fake
        if "information_schema" in s or "exists" in s:
            return {"exists": True}
        if "select version" in s:
            from memblock.migrations import SCHEMA_VERSION

            return {"version": SCHEMA_VERSION}
        return None

    def fetchall(self) -> list:
        return []


class _SyncSpyConn:
    def __init__(self) -> None:
        self.log: list[str] = []
        self.closed = False

    def cursor(self) -> _SyncSpyCursor:
        return _SyncSpyCursor(self.log)

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass


# ─── Async spy pool/connection ───────────────────────────────────────


class _AsyncSpyConn:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    async def execute(self, sql: str, *args: object) -> None:
        self.log.append(sql)

    async def fetchval(self, sql: str, *args: object) -> object:
        self.log.append(sql)
        return None  # pgvector not installed in this fake

    async def fetchrow(self, sql: str, *args: object) -> object:
        self.log.append(sql)
        return None

    async def fetch(self, sql: str, *args: object) -> list:
        self.log.append(sql)
        return []


class _AsyncAcquire:
    def __init__(self, conn: _AsyncSpyConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _AsyncSpyConn:
        return self._conn

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _AsyncSpyPool:
    def __init__(self) -> None:
        self.log: list[str] = []

    def acquire(self) -> _AsyncAcquire:
        return _AsyncAcquire(_AsyncSpyConn(self.log))

    async def close(self) -> None:
        pass


# ─── Flag plumbing (no DB) ───────────────────────────────────────────


@pytest.mark.skipif(not HAS_PSYCOPG, reason="psycopg not installed")
class TestFlagPlumbingSync:
    def test_default_is_true(self):
        adapter = PostgreSQLAdapter(dsn="postgresql://x/y")
        assert adapter._manage_schema is True

    def test_explicit_false_stored(self):
        adapter = PostgreSQLAdapter(dsn="postgresql://x/y", manage_schema=False)
        assert adapter._manage_schema is False

    def test_memblock_threads_flag_to_pg_adapter(self):
        # MemBlock(...).__init__ would connect; go through the factory
        # directly — PostgreSQLAdapter.__init__ does NOT open a connection.
        from memblock import MemBlock

        adapter = MemBlock._create_storage(
            "postgresql://x/y", manage_schema=False
        )
        assert adapter._manage_schema is False
        adapter_default = MemBlock._create_storage("postgresql://x/y")
        assert adapter_default._manage_schema is True


@pytest.mark.skipif(not HAS_ASYNCPG, reason="asyncpg not installed")
class TestFlagPlumbingAsync:
    def test_default_is_true(self):
        adapter = AsyncPostgreSQLAdapter(dsn="postgresql://x/y")
        assert adapter._manage_schema is True

    def test_explicit_false_stored(self):
        adapter = AsyncPostgreSQLAdapter(
            dsn="postgresql://x/y", manage_schema=False
        )
        assert adapter._manage_schema is False

    def test_async_memblock_native_threads_flag(self):
        # Native-async __init__ builds the adapter but does NOT connect
        # (the pool is created lazily on first await).
        from memblock import AsyncMemBlock

        mb = AsyncMemBlock(
            storage="postgresql+asyncpg://x/y",
            embeddings=False,
            manage_schema=False,
        )
        assert mb._async_storage._manage_schema is False


# ─── The core invariant: no DDL when manage_schema=False ─────────────


@pytest.mark.skipif(not HAS_PSYCOPG, reason="psycopg not installed")
class TestNoDDLSync:
    def test_initialize_emits_no_ddl_when_false(self):
        adapter = PostgreSQLAdapter(dsn="postgresql://x/y", manage_schema=False)
        spy = _SyncSpyConn()
        adapter._conn = spy  # inject; .conn property returns it (not closed)
        adapter.initialize()
        assert _ddl_statements(spy.log) == [], (
            "manage_schema=False must emit zero DDL on initialize(); "
            f"got: {_ddl_statements(spy.log)}"
        )

    def test_initialize_DOES_emit_ddl_when_true(self):
        # Contrast: the default path provisions the schema (proves the
        # flag is what gates DDL, not some unrelated short-circuit).
        adapter = PostgreSQLAdapter(dsn="postgresql://x/y", manage_schema=True)
        spy = _SyncSpyConn()
        adapter._conn = spy
        adapter.initialize()
        ddl = _ddl_statements(spy.log)
        assert any("CREATE TABLE" in s.upper() for s in ddl), (
            "manage_schema=True should still create tables; "
            f"DDL seen: {ddl[:3]}"
        )

    def test_pgvector_index_emits_no_ddl_when_false(self):
        # The lazy vector-index path is a SEPARATE runtime DDL surface —
        # it must also be silenced.
        adapter = PostgreSQLAdapter(dsn="postgresql://x/y", manage_schema=False)
        adapter._has_pgvector = True
        spy = _SyncSpyConn()
        adapter._conn = spy
        adapter._ensure_pgvector_index(384)
        assert _ddl_statements(spy.log) == [], (
            "manage_schema=False must not CREATE INDEX on the vector table; "
            f"got: {_ddl_statements(spy.log)}"
        )
        # dims are still cached so it won't re-check on every write.
        assert adapter._embedding_dims == 384


@pytest.mark.skipif(not HAS_ASYNCPG, reason="asyncpg not installed")
class TestNoDDLAsync:
    async def test_initialize_emits_no_ddl_when_false(self):
        adapter = AsyncPostgreSQLAdapter(
            dsn="postgresql://x/y", manage_schema=False
        )
        pool = _AsyncSpyPool()
        adapter._pool = pool  # inject so _ensure_pool returns it (no connect)
        await adapter.initialize()
        assert _ddl_statements(pool.log) == [], (
            "manage_schema=False must emit zero DDL on async initialize(); "
            f"got: {_ddl_statements(pool.log)}"
        )
        assert adapter._initialized is True

    async def test_pgvector_index_emits_no_ddl_when_false(self):
        adapter = AsyncPostgreSQLAdapter(
            dsn="postgresql://x/y", manage_schema=False
        )
        adapter._has_pgvector = True
        pool = _AsyncSpyPool()
        adapter._pool = pool
        await adapter._ensure_pgvector_index(384)
        assert _ddl_statements(pool.log) == []
        assert adapter._embedding_dims == 384


# ─── Live-DB end-to-end proof (gated on MEMBLOCK_TEST_DB_URL) ─────────


async def _count_schema_objects(sync_url: str, schema: str) -> dict[str, int]:
    """Count tables/indexes/functions/triggers in a schema."""
    import asyncpg

    conn = await asyncpg.connect(sync_url)
    try:
        row = await conn.fetchrow(
            """
            SELECT
              (SELECT count(*) FROM information_schema.tables
                 WHERE table_schema = $1) AS tables,
              (SELECT count(*) FROM pg_indexes
                 WHERE schemaname = $1) AS indexes,
              (SELECT count(*) FROM information_schema.routines
                 WHERE specific_schema = $1) AS functions,
              (SELECT count(*) FROM pg_trigger t
                 JOIN pg_class c ON t.tgrelid = c.oid
                 JOIN pg_namespace n ON c.relnamespace = n.oid
                 WHERE n.nspname = $1 AND NOT t.tgisinternal) AS triggers
            """,
            schema,
        )
        return dict(row)
    finally:
        await conn.close()


class TestManageSchemaLiveDB:
    """End-to-end: a manage_schema=False client does real CRUD against a
    pre-provisioned schema and adds ZERO schema objects."""

    async def test_no_ddl_against_provisioned_schema(
        self, postgres_async_url: str, postgres_sync_url: str, fresh_schema: str
    ):
        from memblock import AsyncMemBlock, BlockType

        # 1. Provision the schema with a normal (manage_schema=True) client.
        provisioner = AsyncMemBlock(
            storage=postgres_async_url,
            embeddings=False,
            schema=fresh_schema,
        )
        await provisioner.store("seed", type=BlockType.FACT)
        await provisioner.close()

        before = await _count_schema_objects(postgres_sync_url, fresh_schema)
        assert before["tables"] > 0, "provisioning should have created tables"

        # 2. A manage_schema=False client must do CRUD with no DDL.
        app = AsyncMemBlock(
            storage=postgres_async_url,
            embeddings=False,
            schema=fresh_schema,
            manage_schema=False,
        )
        stored = await app.store("app-role write", type=BlockType.FACT)
        assert stored is not None
        fetched = await app.get(stored.id)
        assert fetched is not None and fetched.content == "app-role write"
        results = await app.query(type=BlockType.FACT)
        assert any(b.id == stored.id for b in results)
        await app.delete(stored.id)
        await app.close()

        # 3. The app client must have added zero schema objects.
        after = await _count_schema_objects(postgres_sync_url, fresh_schema)
        assert after == before, (
            "manage_schema=False added schema objects (DDL leaked): "
            f"before={before} after={after}"
        )
