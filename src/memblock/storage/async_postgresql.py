"""Async PostgreSQL storage adapter for MemBlock.

Asyncpg-backed equivalent of `PostgreSQLAdapter`. Same schema,
same query patterns, but every I/O method is `async def` and runs
through `asyncpg.Pool` for proper async I/O on the event loop.

Why this exists:
  - Async chat handlers (FastAPI / aiohttp / asyncio) under load
    suffer thread-pool overhead when wrapping the sync psycopg
    adapter via `AsyncMemBlock`'s `asyncio.to_thread()` shim.
  - Sharing one Postgres connection per `MemBlock` instance hits
    poisoned-transaction cascades when one nested call errors.
    asyncpg.Pool gives us per-operation connection acquisition,
    isolating failure to the operation that caused it.
  - Production deployments at 10k+ users want a single async
    driver pool that integrates with the rest of the app.

Install: `pip install memblock[postgres-async]`

Usage:
    from memblock.storage.async_postgresql import AsyncPostgreSQLAdapter
    adapter = AsyncPostgreSQLAdapter(
        dsn="postgresql://user:pass@host:5432/db",
        user_id="u_123",
    )
    await adapter.initialize()  # creates memblock_* tables

Or via `AsyncMemBlock` with a `postgresql+asyncpg://` URL — that's
the standard entry point; this class is the underlying transport.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from memblock.block import Block
from memblock.storage.async_base import AsyncStorageAdapter
from memblock.types import (
    BlockMetadata,
    BlockType,
    Edge,
    EdgeRelation,
    EncryptionLevel,
    Operation,
    OpAction,
    SourceType,
)

try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

try:
    from pgvector.asyncpg import register_vector
    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False


# ─── Driver-qualifier stripping ──────────────────────────────────────
#
# SQLAlchemy-style `postgresql+asyncpg://...` URLs carry a `+driver`
# qualifier that asyncpg's parser doesn't recognize. Same problem the
# Alembic bootstrap migration fixes; we mirror the same logic so
# AsyncPostgreSQLAdapter accepts both shapes.

def _strip_sqlalchemy_driver(dsn: str) -> str:
    """Drop any `+<driver>` qualifier from a Postgres DSN."""
    return re.sub(r"^postgresql\+\w+", "postgresql", dsn)


class AsyncPostgreSQLAdapter(AsyncStorageAdapter):
    """asyncpg-backed storage. Manages an `asyncpg.Pool` for proper
    async I/O.

    Connection pool defaults are conservative (min=2, max=20) so
    chat workloads have headroom without exhausting Postgres.
    Override via the `pool` kwarg with a pre-built `asyncpg.Pool` if
    you want to share a pool across the app.
    """

    # pgvector HNSW indexes top out at 2000 dims. Embeddings above
    # that ceiling (e.g. Gemini's `gemini-embedding-001` at 3072 dims)
    # need IVFFlat instead, which has no upper bound.
    _HNSW_MAX_DIMS = 2000

    def __init__(
        self,
        dsn: str,
        user_id: str = "default",
        schema: str = "public",
        pool: Any = None,
        pool_min_size: int = 2,
        pool_max_size: int = 20,
    ) -> None:
        if not HAS_ASYNCPG:
            raise ImportError(
                "AsyncPostgreSQLAdapter requires asyncpg. "
                "Install with: pip install memblock[postgres-async]"
            )

        self.dsn = _strip_sqlalchemy_driver(dsn)
        self.user_id = user_id
        self.schema = schema
        self._pool: asyncpg.Pool | None = pool
        self._pool_owned = pool is None  # we created it -> we close it
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size
        self._has_pgvector: bool = False
        self._embedding_dims: int | None = None
        self._initialized = False

    @property
    def adapter_type(self) -> str:
        return "postgresql_async"

    async def _ensure_pool(self) -> "asyncpg.Pool":
        """Lazily create the pool on first use. Idempotent."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.dsn,
                min_size=self._pool_min_size,
                max_size=self._pool_max_size,
                # Per-connection setup: register pgvector codec when
                # the extension is available so `vector` columns
                # round-trip cleanly. Wrapped in try/except — pgvector
                # is optional.
                init=self._init_connection,
            )
        return self._pool

    async def _init_connection(self, conn: "asyncpg.Connection") -> None:
        """Per-connection setup hook — runs on every new connection
        the pool creates."""
        if HAS_PGVECTOR:
            try:
                await register_vector(conn)
            except Exception:
                # pgvector extension not installed in this DB; the
                # adapter degrades to BYTEA-only embeddings.
                pass

    # ─── Lifecycle ────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Create all memblock_* tables, indexes, FTS triggers, and
        the pgvector embeddings table when the extension is present.
        Idempotent — `CREATE TABLE IF NOT EXISTS` throughout."""
        if self._initialized:
            return

        pool = await self._ensure_pool()

        async with pool.acquire() as conn:
            # ── Schema (no-op for `public`; needed for custom schemas
            # so users can pass `schema='tenant_xyz'` and have the
            # adapter create it on first use).
            if self.schema and self.schema.lower() != "public":
                # Quote to allow case-sensitive / hyphenated names.
                await conn.execute(
                    f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"'
                )

            # ── Core tables ─────────────────────────────────────────
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.memblock_blocks (
                    id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL DEFAULT '',
                    encryption_level TEXT NOT NULL DEFAULT 'none',
                    encrypted BOOLEAN NOT NULL DEFAULT FALSE,
                    parent_id TEXT,
                    children_ids JSONB NOT NULL DEFAULT '[]',
                    version INTEGER NOT NULL DEFAULT 1,
                    op_hash TEXT NOT NULL DEFAULT '',
                    tags JSONB NOT NULL DEFAULT '[]',
                    deleted BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    content_tsv TSVECTOR,
                    content_hash TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (id, user_id)
                )
            """)

            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.memblock_metadata (
                    block_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    source TEXT NOT NULL DEFAULT 'explicit',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    created_by TEXT NOT NULL DEFAULT 'agent',
                    access_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed TIMESTAMPTZ,
                    decay_rate REAL NOT NULL DEFAULT 0.01,
                    ttl INTEGER,
                    session_id TEXT,
                    org_id TEXT,
                    project_id TEXT,
                    agent_id TEXT,
                    custom_metadata JSONB,
                    happened_at TIMESTAMPTZ,
                    happened_at_end TIMESTAMPTZ,
                    temporal_precision TEXT NOT NULL DEFAULT 'exact',
                    PRIMARY KEY (block_id, user_id),
                    FOREIGN KEY (block_id, user_id)
                        REFERENCES {self.schema}.memblock_blocks(id, user_id) ON DELETE CASCADE
                )
            """)

            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.memblock_edges (
                    id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 1.0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (id, user_id)
                )
            """)

            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.memblock_operations (
                    id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    block_id TEXT NOT NULL DEFAULT '',
                    author TEXT NOT NULL DEFAULT 'agent',
                    clock INTEGER NOT NULL DEFAULT 0,
                    action TEXT NOT NULL,
                    data JSONB NOT NULL DEFAULT '{{}}',
                    hash TEXT NOT NULL DEFAULT '',
                    prev_hash TEXT NOT NULL DEFAULT '',
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (id, user_id)
                )
            """)

            # ── Indexes ─────────────────────────────────────────────
            for stmt in [
                f"CREATE INDEX IF NOT EXISTS idx_mb_blocks_user_type "
                f"ON {self.schema}.memblock_blocks(user_id, type)",
                f"CREATE INDEX IF NOT EXISTS idx_mb_blocks_user_deleted "
                f"ON {self.schema}.memblock_blocks(user_id, deleted)",
                f"CREATE INDEX IF NOT EXISTS idx_mb_blocks_parent "
                f"ON {self.schema}.memblock_blocks(user_id, parent_id)",
                f"CREATE INDEX IF NOT EXISTS idx_mb_blocks_tsv "
                f"ON {self.schema}.memblock_blocks USING GIN(content_tsv)",
                f"CREATE INDEX IF NOT EXISTS idx_mb_blocks_content_hash "
                f"ON {self.schema}.memblock_blocks(user_id, content_hash)",
                f"CREATE INDEX IF NOT EXISTS idx_mb_edges_source "
                f"ON {self.schema}.memblock_edges(user_id, source_id)",
                f"CREATE INDEX IF NOT EXISTS idx_mb_edges_target "
                f"ON {self.schema}.memblock_edges(user_id, target_id)",
                f"CREATE INDEX IF NOT EXISTS idx_mb_ops_block "
                f"ON {self.schema}.memblock_operations(user_id, block_id)",
                f"CREATE INDEX IF NOT EXISTS idx_mb_ops_clock "
                f"ON {self.schema}.memblock_operations(user_id, clock)",
                f"CREATE INDEX IF NOT EXISTS idx_mb_meta_confidence "
                f"ON {self.schema}.memblock_metadata(user_id, confidence)",
            ]:
                await conn.execute(stmt)

            # ── FTS trigger to maintain content_tsv ────────────────
            await conn.execute(f"""
                CREATE OR REPLACE FUNCTION {self.schema}.memblock_update_tsv()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.content_tsv := to_tsvector('english', COALESCE(NEW.content, ''));
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
            """)
            # Trigger name is unique PER TABLE, so we scope the
            # existence check to {schema}.memblock_blocks. Without
            # the tgrelid scope, a `public.memblock_blocks` trigger
            # would short-circuit creation here and leave the custom
            # schema's content_tsv permanently NULL — breaking FTS.
            await conn.execute(f"""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_trigger
                         WHERE tgname = 'trg_memblock_tsv'
                           AND tgrelid = '{self.schema}.memblock_blocks'::regclass
                    ) THEN
                        CREATE TRIGGER trg_memblock_tsv
                        BEFORE INSERT OR UPDATE OF content
                        ON {self.schema}.memblock_blocks
                        FOR EACH ROW
                        EXECUTE FUNCTION {self.schema}.memblock_update_tsv();
                    END IF;
                END;
                $$
            """)

            # ── Embeddings (BYTEA always) ──────────────────────────
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.memblock_embeddings (
                    block_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    embedding BYTEA NOT NULL,
                    PRIMARY KEY (block_id, user_id),
                    FOREIGN KEY (block_id, user_id)
                        REFERENCES {self.schema}.memblock_blocks(id, user_id) ON DELETE CASCADE
                )
            """)

            # ── pgvector dual-write (when extension installed) ─────
            if HAS_PGVECTOR:
                pgvector_present = await conn.fetchval(
                    "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
                )
                if pgvector_present:
                    self._has_pgvector = True
                    await conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS
                          {self.schema}.memblock_embeddings_vec (
                            block_id TEXT NOT NULL,
                            user_id TEXT NOT NULL,
                            embedding vector NOT NULL,
                            PRIMARY KEY (block_id, user_id),
                            FOREIGN KEY (block_id, user_id)
                                REFERENCES {self.schema}.memblock_blocks(id, user_id)
                                ON DELETE CASCADE
                        )
                    """)

            # ── Schema-version tracking table ──────────────────────
            # The sync adapter creates this via `MigrationRunner._run_postgres`
            # after the core CREATE TABLE pass. We create it inline + seed
            # with the current SCHEMA_VERSION so a fresh native-async
            # schema is immediately at the latest version with no pending
            # migrations to run. Existing DBs initialized by the sync path
            # already have this table — `IF NOT EXISTS` is a no-op there.
            from memblock.migrations import SCHEMA_VERSION
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS
                    {self.schema}.memblock_schema_version (
                    version INTEGER NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            # Seed the version row only when empty (first init).
            seeded = await conn.fetchval(
                f"SELECT COUNT(*) FROM {self.schema}.memblock_schema_version"
            )
            if not seeded:
                await conn.execute(
                    f"INSERT INTO {self.schema}.memblock_schema_version "
                    f"(version) VALUES ($1)",
                    SCHEMA_VERSION,
                )

            # ── Analytics tables (org-level question tracking) ───
            # The sync adapter creates these on demand via
            # `initialize_analytics_tables()` when `enable_analytics=True`.
            # In the async path we create them eagerly so the analytics
            # methods (`upsert_question_async`, etc.) work without an
            # explicit toggle — these tables are tiny when unused and
            # avoid silent "table doesn't exist" failures if a consumer
            # calls `log_question` later.
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS
                    {self.schema}.memblock_org_questions (
                    id TEXT PRIMARY KEY,
                    org_id TEXT NOT NULL,
                    normalized_text TEXT NOT NULL,
                    frequency_count INTEGER NOT NULL DEFAULT 1,
                    unique_user_count INTEGER NOT NULL DEFAULT 1,
                    first_asked TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_asked TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(org_id, normalized_text)
                )
            """)
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS
                    {self.schema}.memblock_org_question_users (
                    org_id TEXT NOT NULL,
                    question_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    PRIMARY KEY (org_id, question_id, user_id),
                    FOREIGN KEY (question_id)
                        REFERENCES {self.schema}.memblock_org_questions(id)
                        ON DELETE CASCADE
                )
            """)
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS
                    {self.schema}.memblock_org_question_events (
                    id BIGSERIAL PRIMARY KEY,
                    org_id TEXT NOT NULL,
                    question_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    asked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    FOREIGN KEY (question_id)
                        REFERENCES {self.schema}.memblock_org_questions(id)
                        ON DELETE CASCADE
                )
            """)
            for stmt in [
                f"CREATE INDEX IF NOT EXISTS idx_mb_org_questions_org "
                f"ON {self.schema}.memblock_org_questions(org_id)",
                f"CREATE INDEX IF NOT EXISTS idx_mb_org_questions_freq "
                f"ON {self.schema}.memblock_org_questions(org_id, frequency_count DESC)",
                f"CREATE INDEX IF NOT EXISTS idx_mb_org_question_events_time "
                f"ON {self.schema}.memblock_org_question_events(org_id, asked_at)",
            ]:
                await conn.execute(stmt)

        self._initialized = True

        # ── External migrations registry ──────────────────────────
        # We seed the schema_version row above with the latest
        # SCHEMA_VERSION, so a freshly-initialized async schema has
        # no pending migrations to run. When the SDK ships a new
        # column-level change, the migration must also be added to
        # the async path (TODO: port `MigrationRunner._run_postgres`
        # to async). Until then, the async adapter is the source of
        # truth for the latest schema layout.

    async def close(self) -> None:
        """Release the pool when we own it. No-op when an external
        pool was injected (caller's responsibility)."""
        if self._pool is not None and self._pool_owned:
            await self._pool.close()
            self._pool = None

    async def run_migration_sql(
        self, sql: str, params: tuple | None = None,
    ) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            if params:
                await conn.execute(sql, *params)
            else:
                await conn.execute(sql)

    # ─── Block Operations ─────────────────────────────────────────────

    async def save_block(self, block: Block) -> None:
        """Persist a block + its metadata in one transaction."""
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(f"""
                    INSERT INTO {self.schema}.memblock_blocks
                        (id, user_id, type, content, encryption_level, encrypted,
                         parent_id, children_ids, version, op_hash, tags,
                         deleted, created_at, content_hash)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10,
                            $11::jsonb, $12, $13, $14)
                    ON CONFLICT (id, user_id) DO UPDATE SET
                        type = EXCLUDED.type,
                        content = EXCLUDED.content,
                        encryption_level = EXCLUDED.encryption_level,
                        encrypted = EXCLUDED.encrypted,
                        parent_id = EXCLUDED.parent_id,
                        children_ids = EXCLUDED.children_ids,
                        version = EXCLUDED.version,
                        op_hash = EXCLUDED.op_hash,
                        tags = EXCLUDED.tags,
                        deleted = EXCLUDED.deleted,
                        content_hash = EXCLUDED.content_hash
                """,
                    block.id,
                    self.user_id,
                    block.type.value,
                    block.content,
                    block.encryption_level.value,
                    block.encrypted,
                    block.parent_id,
                    json.dumps(block.children_ids),
                    block.version,
                    block.op_hash,
                    json.dumps(block.tags),
                    block.deleted,
                    block.metadata.created_at,
                    block.content_hash,
                )

                await conn.execute(f"""
                    INSERT INTO {self.schema}.memblock_metadata
                        (block_id, user_id, confidence, source, created_at, created_by,
                         access_count, last_accessed, decay_rate, ttl, session_id,
                         org_id, project_id, agent_id, custom_metadata,
                         happened_at, happened_at_end, temporal_precision)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                            $12, $13, $14, $15::jsonb, $16, $17, $18)
                    ON CONFLICT (block_id, user_id) DO UPDATE SET
                        confidence = EXCLUDED.confidence,
                        source = EXCLUDED.source,
                        created_by = EXCLUDED.created_by,
                        access_count = EXCLUDED.access_count,
                        last_accessed = EXCLUDED.last_accessed,
                        decay_rate = EXCLUDED.decay_rate,
                        ttl = EXCLUDED.ttl,
                        session_id = EXCLUDED.session_id,
                        org_id = EXCLUDED.org_id,
                        project_id = EXCLUDED.project_id,
                        agent_id = EXCLUDED.agent_id,
                        custom_metadata = EXCLUDED.custom_metadata,
                        happened_at = EXCLUDED.happened_at,
                        happened_at_end = EXCLUDED.happened_at_end,
                        temporal_precision = EXCLUDED.temporal_precision
                """,
                    block.id,
                    self.user_id,
                    block.metadata.confidence,
                    block.metadata.source.value,
                    block.metadata.created_at,
                    block.metadata.created_by,
                    block.metadata.access_count,
                    block.metadata.last_accessed,
                    block.metadata.decay_rate,
                    block.metadata.ttl,
                    block.metadata.session_id,
                    block.metadata.org_id,
                    block.metadata.project_id,
                    block.metadata.agent_id,
                    json.dumps(block.metadata.custom_metadata) if block.metadata.custom_metadata else None,
                    block.metadata.happened_at,
                    block.metadata.happened_at_end,
                    block.metadata.temporal_precision,
                )

    async def get_block(self, block_id: str) -> Block | None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(f"""
                SELECT b.*,
                       m.confidence, m.source, m.created_at AS m_created_at,
                       m.created_by, m.access_count, m.last_accessed,
                       m.decay_rate, m.ttl, m.session_id, m.org_id,
                       m.project_id, m.agent_id, m.custom_metadata,
                       m.happened_at, m.happened_at_end, m.temporal_precision
                FROM {self.schema}.memblock_blocks b
                LEFT JOIN {self.schema}.memblock_metadata m
                    ON b.id = m.block_id AND b.user_id = m.user_id
                WHERE b.id = $1 AND b.user_id = $2
            """, block_id, self.user_id)
        return self._row_to_block(row) if row else None

    async def delete_block(self, block_id: str) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"""
                DELETE FROM {self.schema}.memblock_blocks
                WHERE id = $1 AND user_id = $2
            """, block_id, self.user_id)

    async def update_block(
        self, block_id: str, updates: dict[str, Any],
    ) -> None:
        block = await self.get_block(block_id)
        if block is None:
            return

        for key, value in updates.items():
            if key == "content":
                block.content = value
            elif key == "type":
                block.type = value if isinstance(value, BlockType) else BlockType(value)
            elif key == "tags":
                block.tags = value
            elif key == "version":
                block.version = value
            elif key == "op_hash":
                block.op_hash = value
            elif key == "parent_id":
                block.parent_id = value
            elif key == "children_ids":
                block.children_ids = value
            elif key == "deleted":
                block.deleted = value
            elif key == "encryption_level":
                block.encryption_level = (
                    value if isinstance(value, EncryptionLevel) else EncryptionLevel(value)
                )
            elif key == "encrypted":
                block.encrypted = value
            elif key == "confidence":
                block.metadata.confidence = value
            elif key == "access_count":
                block.metadata.access_count = value
            elif key == "last_accessed":
                block.metadata.last_accessed = value
            elif key == "decay_rate":
                block.metadata.decay_rate = value

        await self.save_block(block)

    async def query_blocks(
        self, filters: dict[str, Any],
    ) -> list[Block]:
        """Query blocks. Mirrors `PostgreSQLAdapter.query_blocks`'
        filter shape exactly.

        asyncpg uses `$1, $2, ...` placeholders (Postgres native);
        we build the parameter list and substitute placeholders by
        index into the SQL string.
        """
        # Build conditions + parameter list
        conditions: list[str] = ["b.user_id = $1"]
        params: list[Any] = [self.user_id]

        def add(condition_template: str, *values: Any) -> None:
            """Append a condition with positional params, swapping
            the placeholder-token `?` for `$N` per asyncpg."""
            n_params = condition_template.count("?")
            placeholders = [f"${len(params) + 1 + i}" for i in range(n_params)]
            # Replace the i-th `?` with $N (one at a time to preserve order)
            condition = condition_template
            for ph in placeholders:
                condition = condition.replace("?", ph, 1)
            conditions.append(condition)
            params.extend(values)

        if not filters.get("deleted", False):
            conditions.append("b.deleted = FALSE")

        if "type" in filters:
            block_type = filters["type"]
            if isinstance(block_type, BlockType):
                block_type = block_type.value
            add("b.type = ?", block_type)

        if "parent_id" in filters:
            add("b.parent_id = ?", filters["parent_id"])

        if "min_confidence" in filters:
            add("m.confidence >= ?", filters["min_confidence"])

        if "session_id" in filters:
            add("m.session_id = ?", filters["session_id"])

        if "org_id" in filters:
            add("m.org_id = ?", filters["org_id"])

        if "project_id" in filters:
            add("m.project_id = ?", filters["project_id"])

        if "agent_id" in filters:
            add("m.agent_id = ?", filters["agent_id"])

        if "metadata_filters" in filters:
            for key, value in filters["metadata_filters"].items():
                add(
                    "m.custom_metadata->>? = ?",
                    key,
                    str(value) if not isinstance(value, str) else value,
                )

        if "tags" in filters:
            tag_list = filters["tags"]
            tag_conditions: list[str] = []
            for tag in tag_list:
                placeholder = f"${len(params) + 1}"
                tag_conditions.append(f"b.tags::text LIKE {placeholder}")
                params.append(f"%{tag}%")
            if tag_conditions:
                conditions.append(f"({' OR '.join(tag_conditions)})")

        if "happened_after" in filters:
            add("m.happened_at >= ?", filters["happened_after"])

        if "happened_before" in filters:
            add("m.happened_at <= ?", filters["happened_before"])

        if "temporal_range" in filters:
            start, end = filters["temporal_range"]
            add("m.happened_at >= ?", start)
            add("m.happened_at <= ?", end)

        if "text_search" in filters:
            raw_query = filters["text_search"]
            words = re.findall(r"\w+", raw_query)
            if words:
                ts_query = " | ".join(words)
                add("b.content_tsv @@ to_tsquery('english', ?)", ts_query)

        where_clause = " AND ".join(conditions)

        sort_by = filters.get("sort_by", "created_at")
        sort_map = {
            "created_at": "b.created_at DESC",
            "access_count": "m.access_count DESC",
            "confidence": "m.confidence DESC",
        }
        order = sort_map.get(sort_by, "b.created_at DESC")

        sql = f"""
            SELECT b.*,
                   m.confidence, m.source, m.created_at AS m_created_at,
                   m.created_by, m.access_count, m.last_accessed,
                   m.decay_rate, m.ttl, m.session_id, m.org_id,
                   m.project_id, m.agent_id, m.custom_metadata,
                   m.happened_at, m.happened_at_end, m.temporal_precision
            FROM {self.schema}.memblock_blocks b
            LEFT JOIN {self.schema}.memblock_metadata m
                ON b.id = m.block_id AND b.user_id = m.user_id
            WHERE {where_clause}
            ORDER BY {order}
        """
        if "limit" in filters:
            sql += f" LIMIT ${len(params) + 1}"
            params.append(filters["limit"])

        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        return [self._row_to_block(row) for row in rows]

    async def get_all_blocks(
        self, include_deleted: bool = False,
    ) -> list[Block]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            if include_deleted:
                rows = await conn.fetch(f"""
                    SELECT b.*,
                           m.confidence, m.source, m.created_at AS m_created_at,
                           m.created_by, m.access_count, m.last_accessed,
                           m.decay_rate, m.ttl, m.session_id, m.org_id,
                           m.project_id, m.agent_id, m.custom_metadata,
                           m.happened_at, m.happened_at_end, m.temporal_precision
                    FROM {self.schema}.memblock_blocks b
                    LEFT JOIN {self.schema}.memblock_metadata m
                        ON b.id = m.block_id AND b.user_id = m.user_id
                    WHERE b.user_id = $1
                    ORDER BY b.created_at DESC
                """, self.user_id)
            else:
                rows = await conn.fetch(f"""
                    SELECT b.*,
                           m.confidence, m.source, m.created_at AS m_created_at,
                           m.created_by, m.access_count, m.last_accessed,
                           m.decay_rate, m.ttl, m.session_id, m.org_id,
                           m.project_id, m.agent_id, m.custom_metadata,
                           m.happened_at, m.happened_at_end, m.temporal_precision
                    FROM {self.schema}.memblock_blocks b
                    LEFT JOIN {self.schema}.memblock_metadata m
                        ON b.id = m.block_id AND b.user_id = m.user_id
                    WHERE b.user_id = $1 AND b.deleted = FALSE
                    ORDER BY b.created_at DESC
                """, self.user_id)
        return [self._row_to_block(row) for row in rows]

    async def get_block_by_content_hash(
        self, content_hash: str,
    ) -> Block | None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(f"""
                SELECT b.*,
                       m.confidence, m.source, m.created_at AS m_created_at,
                       m.created_by, m.access_count, m.last_accessed,
                       m.decay_rate, m.ttl, m.session_id, m.org_id,
                       m.project_id, m.agent_id, m.custom_metadata,
                       m.happened_at, m.happened_at_end, m.temporal_precision
                FROM {self.schema}.memblock_blocks b
                LEFT JOIN {self.schema}.memblock_metadata m
                    ON b.id = m.block_id AND b.user_id = m.user_id
                WHERE b.content_hash = $1 AND b.user_id = $2 AND b.deleted = FALSE
                LIMIT 1
            """, content_hash, self.user_id)
        return self._row_to_block(row) if row else None

    # ─── Edge Operations ──────────────────────────────────────────────

    async def save_edge(self, edge: Edge) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"""
                INSERT INTO {self.schema}.memblock_edges
                    (id, user_id, source_id, target_id, relation, weight, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (id, user_id) DO UPDATE SET
                    source_id = EXCLUDED.source_id,
                    target_id = EXCLUDED.target_id,
                    relation = EXCLUDED.relation,
                    weight = EXCLUDED.weight
            """,
                edge.id, self.user_id, edge.source_id, edge.target_id,
                edge.relation.value, edge.weight, edge.created_at,
            )

    async def get_edges(
        self, block_id: str, direction: str = "both",
    ) -> list[Edge]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            if direction == "outgoing":
                rows = await conn.fetch(f"""
                    SELECT * FROM {self.schema}.memblock_edges
                    WHERE user_id = $1 AND source_id = $2
                """, self.user_id, block_id)
            elif direction == "incoming":
                rows = await conn.fetch(f"""
                    SELECT * FROM {self.schema}.memblock_edges
                    WHERE user_id = $1 AND target_id = $2
                """, self.user_id, block_id)
            else:
                rows = await conn.fetch(f"""
                    SELECT * FROM {self.schema}.memblock_edges
                    WHERE user_id = $1 AND (source_id = $2 OR target_id = $2)
                """, self.user_id, block_id)
        return [self._row_to_edge(row) for row in rows]

    async def delete_edge(self, edge_id: str) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"""
                DELETE FROM {self.schema}.memblock_edges
                WHERE id = $1 AND user_id = $2
            """, edge_id, self.user_id)

    async def delete_edges_for_block(self, block_id: str) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"""
                DELETE FROM {self.schema}.memblock_edges
                WHERE user_id = $1 AND (source_id = $2 OR target_id = $2)
            """, self.user_id, block_id)

    # ─── Operation Log ────────────────────────────────────────────────

    async def save_operation(self, op: Operation) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"""
                INSERT INTO {self.schema}.memblock_operations
                    (id, user_id, block_id, author, clock, action, data,
                     hash, prev_hash, timestamp)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10)
            """,
                op.id, self.user_id, op.block_id, op.author, op.clock,
                op.action.value, json.dumps(op.data), op.hash, op.prev_hash,
                op.timestamp,
            )

    async def get_operations(
        self, block_id: str | None = None,
    ) -> list[Operation]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            if block_id:
                rows = await conn.fetch(f"""
                    SELECT * FROM {self.schema}.memblock_operations
                    WHERE user_id = $1 AND block_id = $2
                    ORDER BY clock ASC
                """, self.user_id, block_id)
            else:
                rows = await conn.fetch(f"""
                    SELECT * FROM {self.schema}.memblock_operations
                    WHERE user_id = $1
                    ORDER BY clock ASC
                """, self.user_id)
        return [self._row_to_operation(row) for row in rows]

    async def get_last_operation(self) -> Operation | None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(f"""
                SELECT * FROM {self.schema}.memblock_operations
                WHERE user_id = $1
                ORDER BY clock DESC
                LIMIT 1
            """, self.user_id)
        return self._row_to_operation(row) if row else None

    # ─── Embedding Operations ────────────────────────────────────────

    async def _ensure_pgvector_index(self, dims: int) -> None:
        """HNSW for ≤2000 dims, IVFFlat for >2000. Failures rolled
        back via try/except so the connection stays usable."""
        if self._embedding_dims is not None:
            return
        self._embedding_dims = dims

        index_type = "hnsw" if dims <= self._HNSW_MAX_DIMS else "ivfflat"
        index_name = f"idx_mb_embeddings_vec_{index_type}"

        pool = await self._ensure_pool()
        try:
            async with pool.acquire() as conn:
                if index_type == "hnsw":
                    await conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS {index_name}
                        ON {self.schema}.memblock_embeddings_vec
                        USING hnsw (embedding vector_cosine_ops)
                    """)
                else:
                    await conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS {index_name}
                        ON {self.schema}.memblock_embeddings_vec
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100)
                    """)
        except Exception:
            # Don't poison subsequent operations. Vector index is a
            # perf hint; cosine_similarity over BYTEA still works.
            self._embedding_dims = None

    async def save_embedding(
        self, block_id: str, embedding: bytes,
    ) -> None:
        pool = await self._ensure_pool()
        # BYTEA write — source of truth.
        async with pool.acquire() as conn:
            await conn.execute(f"""
                INSERT INTO {self.schema}.memblock_embeddings
                    (block_id, user_id, embedding)
                VALUES ($1, $2, $3)
                ON CONFLICT (block_id, user_id)
                DO UPDATE SET embedding = EXCLUDED.embedding
            """, block_id, self.user_id, embedding)

        # pgvector dual-write — wrapped so failure doesn't propagate.
        # pgvector.asyncpg's codec accepts a Python `list[float]` and
        # encodes it via Postgres' binary protocol — pass the list
        # directly, NOT a str() cast (the latter only works under
        # psycopg's text protocol).
        if self._has_pgvector:
            from memblock.embeddings import unpack_embedding
            vec = unpack_embedding(embedding)
            await self._ensure_pgvector_index(len(vec))
            try:
                async with pool.acquire() as conn:
                    await conn.execute(f"""
                        INSERT INTO {self.schema}.memblock_embeddings_vec
                            (block_id, user_id, embedding)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (block_id, user_id)
                        DO UPDATE SET embedding = EXCLUDED.embedding
                    """, block_id, self.user_id, vec)
            except Exception:
                # Failure here doesn't affect correctness — BYTEA is
                # already persisted. cosine_similarity fallback works.
                pass

    async def get_embedding(self, block_id: str) -> bytes | None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(f"""
                SELECT embedding FROM {self.schema}.memblock_embeddings
                WHERE block_id = $1 AND user_id = $2
            """, block_id, self.user_id)
        return bytes(row["embedding"]) if row else None

    async def get_all_embeddings(self) -> list[tuple[str, bytes]]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT e.block_id, e.embedding
                FROM {self.schema}.memblock_embeddings e
                INNER JOIN {self.schema}.memblock_blocks b
                    ON e.block_id = b.id AND e.user_id = b.user_id
                WHERE e.user_id = $1 AND b.deleted = FALSE
            """, self.user_id)
        return [(row["block_id"], bytes(row["embedding"])) for row in rows]

    async def delete_embedding(self, block_id: str) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"""
                DELETE FROM {self.schema}.memblock_embeddings
                WHERE block_id = $1 AND user_id = $2
            """, block_id, self.user_id)
            if self._has_pgvector:
                await conn.execute(f"""
                    DELETE FROM {self.schema}.memblock_embeddings_vec
                    WHERE block_id = $1 AND user_id = $2
                """, block_id, self.user_id)

    async def search_similar_embeddings(
        self, query_embedding: bytes, limit: int = 20,
    ) -> list[tuple[str, float]]:
        """Server-side cosine similarity via pgvector. Returns
        [(block_id, score), ...] sorted by similarity desc.
        Empty list when pgvector is unavailable — caller falls back
        to Python cosine over `get_all_embeddings`.

        Pass the query vector as a Python `list[float]`; pgvector's
        asyncpg codec handles the binary encoding. The `::vector`
        cast (used by the sync psycopg adapter) is unnecessary here.
        """
        if not self._has_pgvector:
            return []

        from memblock.embeddings import unpack_embedding
        query_vec = unpack_embedding(query_embedding)

        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT e.block_id,
                       1 - (e.embedding <=> $1) AS similarity
                FROM {self.schema}.memblock_embeddings_vec e
                INNER JOIN {self.schema}.memblock_blocks b
                    ON e.block_id = b.id AND e.user_id = b.user_id
                WHERE e.user_id = $2 AND b.deleted = FALSE
                ORDER BY e.embedding <=> $1
                LIMIT $3
            """, query_vec, self.user_id, limit)
        return [(row["block_id"], float(row["similarity"])) for row in rows]

    # ─── Analytics (org-level question tracking) ────────────────────
    #
    # The schema for `memblock_org_questions`, `memblock_org_question_users`,
    # and `memblock_org_question_events` was already created during
    # `initialize()` (see schema migrations above). These methods are
    # the async equivalents of `PostgreSQLAdapter.upsert_question`,
    # `get_top_questions`, etc.

    async def upsert_question_async(
        self,
        org_id: str,
        normalized_text: str,
        user_id: str,
        asked_at: Any,
    ) -> dict[str, Any]:
        """Upsert a question (frequency + unique-user counters) and
        record an event row. Returns the question record."""
        pool = await self._ensure_pool()
        # Stable id derived from (org_id, normalized_text). Lower 12
        # hex chars of SHA-1 — identical scheme to the sync adapter.
        import hashlib
        question_id = (
            "q_" + hashlib.sha1(
                f"{org_id}|{normalized_text}".encode()
            ).hexdigest()[:12]
        )

        async with pool.acquire() as conn:
            async with conn.transaction():
                # 1. Upsert the question
                row = await conn.fetchrow(f"""
                    INSERT INTO {self.schema}.memblock_org_questions
                        (id, org_id, normalized_text, frequency_count,
                         unique_user_count, first_asked, last_asked)
                    VALUES ($1, $2, $3, 1, 1, $4, $4)
                    ON CONFLICT (org_id, normalized_text) DO UPDATE SET
                        frequency_count = memblock_org_questions.frequency_count + 1,
                        last_asked = EXCLUDED.last_asked
                    RETURNING id, org_id, normalized_text,
                              frequency_count, unique_user_count,
                              first_asked, last_asked
                """, question_id, org_id, normalized_text, asked_at)

                # 2. Track unique user — INSERT IGNORE pattern
                user_row = await conn.fetchrow(f"""
                    INSERT INTO {self.schema}.memblock_org_question_users
                        (org_id, question_id, user_id)
                    VALUES ($1, $2, $3)
                    ON CONFLICT DO NOTHING
                    RETURNING 1
                """, org_id, row["id"], user_id)

                if user_row is not None:
                    # First time this user asked it — bump unique_user_count
                    await conn.execute(f"""
                        UPDATE {self.schema}.memblock_org_questions
                        SET unique_user_count = unique_user_count + 1
                        WHERE id = $1
                    """, row["id"])

                # 3. Append event row for time-window analytics
                await conn.execute(f"""
                    INSERT INTO {self.schema}.memblock_org_question_events
                        (org_id, question_id, user_id, asked_at)
                    VALUES ($1, $2, $3, $4)
                """, org_id, row["id"], user_id, asked_at)

        return {
            "id": row["id"],
            "org_id": row["org_id"],
            "normalized_text": row["normalized_text"],
            "frequency_count": row["frequency_count"] + 1,
            "unique_user_count": row["unique_user_count"]
                + (1 if user_row else 0),
            "first_asked": row["first_asked"],
            "last_asked": asked_at,
        }

    async def get_top_questions_async(
        self,
        org_id: str,
        limit: int = 20,
        since: Any = None,
        until: Any = None,
    ) -> list[dict[str, Any]]:
        """Top-N questions for an org by frequency, optionally
        windowed by `since` / `until` (uses event-table counts when
        windowed; aggregate counts otherwise)."""
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            if since is None and until is None:
                rows = await conn.fetch(f"""
                    SELECT id, org_id, normalized_text,
                           frequency_count, unique_user_count,
                           first_asked, last_asked
                    FROM {self.schema}.memblock_org_questions
                    WHERE org_id = $1
                    ORDER BY frequency_count DESC, last_asked DESC
                    LIMIT $2
                """, org_id, limit)
            else:
                # Windowed — count via events table
                params: list[Any] = [org_id]
                conditions = ["e.org_id = $1"]
                if since is not None:
                    conditions.append(f"e.asked_at >= ${len(params) + 1}")
                    params.append(since)
                if until is not None:
                    conditions.append(f"e.asked_at <= ${len(params) + 1}")
                    params.append(until)
                where = " AND ".join(conditions)
                params.append(limit)
                rows = await conn.fetch(f"""
                    SELECT q.id, q.org_id, q.normalized_text,
                           COUNT(*) AS frequency_count,
                           COUNT(DISTINCT e.user_id) AS unique_user_count,
                           q.first_asked, MAX(e.asked_at) AS last_asked
                    FROM {self.schema}.memblock_org_question_events e
                    INNER JOIN {self.schema}.memblock_org_questions q
                        ON e.question_id = q.id
                    WHERE {where}
                    GROUP BY q.id, q.org_id, q.normalized_text, q.first_asked
                    ORDER BY frequency_count DESC, last_asked DESC
                    LIMIT ${len(params)}
                """, *params)

        return [
            {
                "id": r["id"],
                "org_id": r["org_id"],
                "normalized_text": r["normalized_text"],
                "frequency_count": int(r["frequency_count"]),
                "unique_user_count": int(r["unique_user_count"]),
                "first_asked": r["first_asked"],
                "last_asked": r["last_asked"],
            }
            for r in rows
        ]

    async def question_stats_async(
        self, org_id: str,
    ) -> dict[str, Any]:
        """Aggregate stats for an org: total questions, total events,
        unique askers, top-1 frequency."""
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(f"""
                SELECT
                    (SELECT COUNT(*)
                       FROM {self.schema}.memblock_org_questions
                       WHERE org_id = $1) AS distinct_questions,
                    (SELECT COUNT(*)
                       FROM {self.schema}.memblock_org_question_events
                       WHERE org_id = $1) AS total_events,
                    (SELECT COUNT(DISTINCT user_id)
                       FROM {self.schema}.memblock_org_question_users
                       WHERE org_id = $1) AS unique_users,
                    (SELECT MAX(frequency_count)
                       FROM {self.schema}.memblock_org_questions
                       WHERE org_id = $1) AS top_frequency
            """, org_id)
        return {
            "org_id": org_id,
            "distinct_questions": int(row["distinct_questions"] or 0),
            "total_events": int(row["total_events"] or 0),
            "unique_users": int(row["unique_users"] or 0),
            "top_frequency": int(row["top_frequency"] or 0),
        }

    async def get_question_breakdown_async(
        self,
        org_id: str,
        question_text: str,
        granularity: str = "daily",
    ) -> dict[str, Any]:
        """Frequency breakdown for one normalized question, bucketed
        by daily / weekly / monthly. Mirrors the sync analytics
        method for parity."""
        bucket_expr = {
            "daily": "DATE_TRUNC('day', asked_at)",
            "weekly": "DATE_TRUNC('week', asked_at)",
            "monthly": "DATE_TRUNC('month', asked_at)",
        }.get(granularity, "DATE_TRUNC('day', asked_at)")

        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            q_row = await conn.fetchrow(f"""
                SELECT id, frequency_count, first_asked, last_asked
                FROM {self.schema}.memblock_org_questions
                WHERE org_id = $1 AND normalized_text = $2
            """, org_id, question_text)
            if q_row is None:
                return {
                    "question": question_text,
                    "buckets": [],
                    "total": 0,
                }
            bucket_rows = await conn.fetch(f"""
                SELECT {bucket_expr} AS bucket, COUNT(*) AS count
                FROM {self.schema}.memblock_org_question_events
                WHERE org_id = $1 AND question_id = $2
                GROUP BY bucket
                ORDER BY bucket ASC
            """, org_id, q_row["id"])

        return {
            "question": question_text,
            "total": int(q_row["frequency_count"]),
            "first_asked": q_row["first_asked"],
            "last_asked": q_row["last_asked"],
            "granularity": granularity,
            "buckets": [
                {"bucket": r["bucket"], "count": int(r["count"])}
                for r in bucket_rows
            ],
        }

    # ─── Internal Helpers ────────────────────────────────────────────

    def _row_to_block(self, row: Any) -> Block:
        # asyncpg.Record supports `row["key"]` / `row.get(key)` access
        children_ids = row["children_ids"]
        if isinstance(children_ids, str):
            children_ids = json.loads(children_ids)

        tags = row["tags"]
        if isinstance(tags, str):
            tags = json.loads(tags)

        m_created_at = row.get("m_created_at") if hasattr(row, "get") else row["m_created_at"]
        if m_created_at is None:
            m_created_at = row["created_at"]

        custom_meta = row.get("custom_metadata") if hasattr(row, "get") else row["custom_metadata"]
        if isinstance(custom_meta, str):
            try:
                custom_meta = json.loads(custom_meta)
            except (TypeError, ValueError):
                custom_meta = None

        return Block(
            id=row["id"],
            type=BlockType(row["type"]),
            content=row["content"],
            metadata=BlockMetadata(
                confidence=row["confidence"] if row["confidence"] is not None else 1.0,
                source=SourceType(row["source"]) if row["source"] else SourceType.EXPLICIT,
                created_at=m_created_at,
                created_by=row["created_by"] if row["created_by"] else "agent",
                access_count=row["access_count"] if row["access_count"] is not None else 0,
                last_accessed=row["last_accessed"],
                decay_rate=row["decay_rate"] if row["decay_rate"] is not None else 0.01,
                ttl=row["ttl"],
                session_id=row["session_id"],
                org_id=row["org_id"],
                project_id=row["project_id"],
                agent_id=row["agent_id"],
                custom_metadata=custom_meta,
                happened_at=row["happened_at"],
                happened_at_end=row["happened_at_end"],
                temporal_precision=row["temporal_precision"] or "exact",
            ),
            encryption_level=EncryptionLevel(row["encryption_level"]),
            encrypted=bool(row["encrypted"]),
            parent_id=row["parent_id"],
            children_ids=children_ids,
            version=row["version"],
            op_hash=row["op_hash"],
            tags=tags,
            content_hash=row.get("content_hash", "") if hasattr(row, "get") else (row["content_hash"] or ""),
            deleted=bool(row["deleted"]),
        )

    def _row_to_edge(self, row: Any) -> Edge:
        return Edge(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            relation=EdgeRelation(row["relation"]),
            weight=row["weight"],
            created_at=row["created_at"],
        )

    def _row_to_operation(self, row: Any) -> Operation:
        data = row["data"]
        if isinstance(data, str):
            data = json.loads(data)
        return Operation(
            id=row["id"],
            block_id=row["block_id"],
            author=row["author"],
            clock=row["clock"],
            action=OpAction(row["action"]),
            data=data,
            hash=row["hash"],
            prev_hash=row["prev_hash"],
            timestamp=row["timestamp"],
        )
