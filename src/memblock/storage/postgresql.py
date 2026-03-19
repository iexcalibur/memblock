"""PostgreSQL storage adapter for MemBlock — production multi-user backend."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from memblock.block import Block
from memblock.storage.base import StorageAdapter
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
    import psycopg
    from psycopg.rows import dict_row

    HAS_PSYCOPG = True
except ImportError:
    HAS_PSYCOPG = False

try:
    from psycopg_pool import ConnectionPool as PsycopgPool

    HAS_PSYCOPG_POOL = True
except ImportError:
    HAS_PSYCOPG_POOL = False

try:
    from pgvector.psycopg import register_vector

    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False


class PostgreSQLAdapter(StorageAdapter):
    """
    PostgreSQL-based storage adapter for production multi-user deployments.

    Requires: pip install memblock[postgres]

    Features over SQLite:
    - Multi-user: all blocks scoped by user_id
    - Concurrent access safe
    - Full-text search via tsvector/tsquery
    - Scales to millions of blocks
    - Lives alongside your existing PostgreSQL tables

    Usage:
        adapter = PostgreSQLAdapter(
            dsn="postgresql://user:pass@localhost:5432/mydb",
            user_id="u_123",
        )
        adapter.initialize()  # creates tables if not exists
    """

    def __init__(
        self,
        dsn: str,
        user_id: str = "default",
        schema: str = "public",
        pool: Any = None,
    ) -> None:
        if not HAS_PSYCOPG:
            raise ImportError(
                "PostgreSQL adapter requires psycopg. "
                "Install with: pip install memblock[postgres]"
            )

        self.dsn = dsn
        self.user_id = user_id
        self.schema = schema
        self._pool = pool
        self._pool_conn = False
        self._conn: psycopg.Connection | None = None
        self._has_pgvector: bool = False
        self._embedding_dims: int | None = None

    @property
    def adapter_type(self) -> str:
        return "postgresql"

    @property
    def conn(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            if self._pool is not None:
                self._conn = self._pool.getconn()
                self._conn.row_factory = dict_row
                self._conn.autocommit = False
                self._pool_conn = True
            else:
                self._conn = psycopg.connect(self.dsn, row_factory=dict_row, autocommit=False)
                self._pool_conn = False
        return self._conn

    def initialize(self) -> None:
        """Create all MemBlock tables if they don't exist."""
        with self.conn.cursor() as cur:
            cur.execute(f"""
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
                    PRIMARY KEY (id, user_id)
                )
            """)

            cur.execute(f"""
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
                    PRIMARY KEY (block_id, user_id),
                    FOREIGN KEY (block_id, user_id)
                        REFERENCES {self.schema}.memblock_blocks(id, user_id) ON DELETE CASCADE
                )
            """)

            cur.execute(f"""
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

            cur.execute(f"""
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

            # Indexes
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_mb_blocks_user_type
                ON {self.schema}.memblock_blocks(user_id, type)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_mb_blocks_user_deleted
                ON {self.schema}.memblock_blocks(user_id, deleted)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_mb_blocks_parent
                ON {self.schema}.memblock_blocks(user_id, parent_id)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_mb_blocks_tsv
                ON {self.schema}.memblock_blocks USING GIN(content_tsv)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_mb_edges_source
                ON {self.schema}.memblock_edges(user_id, source_id)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_mb_edges_target
                ON {self.schema}.memblock_edges(user_id, target_id)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_mb_ops_block
                ON {self.schema}.memblock_operations(user_id, block_id)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_mb_ops_clock
                ON {self.schema}.memblock_operations(user_id, clock)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_mb_meta_confidence
                ON {self.schema}.memblock_metadata(user_id, confidence)
            """)

            # Trigger: auto-update tsvector on insert/update
            cur.execute(f"""
                CREATE OR REPLACE FUNCTION {self.schema}.memblock_update_tsv()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.content_tsv := to_tsvector('english', COALESCE(NEW.content, ''));
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
            """)

            cur.execute(f"""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_trigger WHERE tgname = 'trg_memblock_tsv'
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

            # Embeddings table for vector search (BYTEA fallback)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.memblock_embeddings (
                    block_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    embedding BYTEA NOT NULL,
                    PRIMARY KEY (block_id, user_id),
                    FOREIGN KEY (block_id, user_id)
                        REFERENCES {self.schema}.memblock_blocks(id, user_id) ON DELETE CASCADE
                )
            """)

            # pgvector: create vector-typed embeddings table if extension available
            if HAS_PGVECTOR:
                cur.execute(
                    "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
                )
                if cur.fetchone():
                    self._has_pgvector = True
                    register_vector(self.conn)
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self.schema}.memblock_embeddings_vec (
                            block_id TEXT NOT NULL,
                            user_id TEXT NOT NULL,
                            embedding vector NOT NULL,
                            PRIMARY KEY (block_id, user_id),
                            FOREIGN KEY (block_id, user_id)
                                REFERENCES {self.schema}.memblock_blocks(id, user_id)
                                ON DELETE CASCADE
                        )
                    """)

        self.conn.commit()

        # Run schema migrations
        from memblock.migrations import MigrationRunner
        MigrationRunner(self).run()

    def run_migration_sql(self, sql: str, params: tuple | None = None) -> None:
        with self.conn.cursor() as cur:
            if params:
                cur.execute(sql, params)
            else:
                cur.execute(sql)

    # ─── Block Operations ─────────────────────────────────────────────────

    def save_block(self, block: Block) -> None:
        with self.conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {self.schema}.memblock_blocks
                    (id, user_id, type, content, encryption_level, encrypted,
                     parent_id, children_ids, version, op_hash, tags, deleted, created_at, content_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            """, (
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
                block.metadata.created_at.isoformat(),
                block.content_hash,
            ))

            cur.execute(f"""
                INSERT INTO {self.schema}.memblock_metadata
                    (block_id, user_id, confidence, source, created_at, created_by,
                     access_count, last_accessed, decay_rate, ttl, session_id,
                     org_id, project_id, agent_id, custom_metadata,
                     happened_at, happened_at_end, temporal_precision)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            """, (
                block.id,
                self.user_id,
                block.metadata.confidence,
                block.metadata.source.value,
                block.metadata.created_at.isoformat(),
                block.metadata.created_by,
                block.metadata.access_count,
                block.metadata.last_accessed.isoformat() if block.metadata.last_accessed else None,
                block.metadata.decay_rate,
                block.metadata.ttl,
                block.metadata.session_id,
                block.metadata.org_id,
                block.metadata.project_id,
                block.metadata.agent_id,
                json.dumps(block.metadata.custom_metadata) if block.metadata.custom_metadata else None,
                block.metadata.happened_at.isoformat() if block.metadata.happened_at else None,
                block.metadata.happened_at_end.isoformat() if block.metadata.happened_at_end else None,
                block.metadata.temporal_precision,
            ))

        self.conn.commit()

    def get_block(self, block_id: str) -> Block | None:
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT b.*, m.confidence, m.source, m.created_at AS m_created_at,
                       m.created_by, m.access_count, m.last_accessed, m.decay_rate, m.ttl, m.session_id,
                       m.org_id, m.project_id, m.agent_id, m.custom_metadata,
                       m.happened_at, m.happened_at_end, m.temporal_precision
                FROM {self.schema}.memblock_blocks b
                LEFT JOIN {self.schema}.memblock_metadata m
                    ON b.id = m.block_id AND b.user_id = m.user_id
                WHERE b.id = %s AND b.user_id = %s
            """, (block_id, self.user_id))
            row = cur.fetchone()

        if row is None:
            return None
        return self._row_to_block(row)

    def delete_block(self, block_id: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(f"""
                DELETE FROM {self.schema}.memblock_blocks
                WHERE id = %s AND user_id = %s
            """, (block_id, self.user_id))
        self.conn.commit()

    def update_block(self, block_id: str, updates: dict[str, Any]) -> None:
        block = self.get_block(block_id)
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
                block.encryption_level = value if isinstance(value, EncryptionLevel) else EncryptionLevel(value)
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

        self.save_block(block)

    def query_blocks(self, filters: dict[str, Any]) -> list[Block]:
        conditions: list[str] = [f"b.user_id = %s"]
        params: list[Any] = [self.user_id]

        # Default: exclude deleted
        if not filters.get("deleted", False):
            conditions.append("b.deleted = FALSE")

        if "type" in filters:
            block_type = filters["type"]
            if isinstance(block_type, BlockType):
                block_type = block_type.value
            conditions.append("b.type = %s")
            params.append(block_type)

        if "parent_id" in filters:
            conditions.append("b.parent_id = %s")
            params.append(filters["parent_id"])

        if "min_confidence" in filters:
            conditions.append("m.confidence >= %s")
            params.append(filters["min_confidence"])

        if "session_id" in filters:
            conditions.append("m.session_id = %s")
            params.append(filters["session_id"])

        if "org_id" in filters:
            conditions.append("m.org_id = %s")
            params.append(filters["org_id"])

        if "project_id" in filters:
            conditions.append("m.project_id = %s")
            params.append(filters["project_id"])

        if "agent_id" in filters:
            conditions.append("m.agent_id = %s")
            params.append(filters["agent_id"])

        if "metadata_filters" in filters:
            for key, value in filters["metadata_filters"].items():
                conditions.append("m.custom_metadata->>%s = %s")
                params.append(key)
                params.append(str(value) if not isinstance(value, str) else value)

        if "tags" in filters:
            tag_list = filters["tags"]
            tag_conditions = []
            for tag in tag_list:
                tag_conditions.append("b.tags::text LIKE %s")
                params.append(f"%{tag}%")
            if tag_conditions:
                conditions.append(f"({' OR '.join(tag_conditions)})")

        if "happened_after" in filters:
            conditions.append("m.happened_at >= %s")
            params.append(filters["happened_after"].isoformat())

        if "happened_before" in filters:
            conditions.append("m.happened_at <= %s")
            params.append(filters["happened_before"].isoformat())

        if "temporal_range" in filters:
            start, end = filters["temporal_range"]
            conditions.append("m.happened_at >= %s AND m.happened_at <= %s")
            params.extend([start.isoformat(), end.isoformat()])

        if "text_search" in filters:
            # PostgreSQL full-text search using tsvector
            import re
            raw_query = filters["text_search"]
            words = re.findall(r'\w+', raw_query)
            if words:
                ts_query = " | ".join(words)  # OR search
                conditions.append("b.content_tsv @@ to_tsquery('english', %s)")
                params.append(ts_query)

        where_clause = " AND ".join(conditions)

        # Sort
        sort_by = filters.get("sort_by", "created_at")
        sort_map = {
            "created_at": "b.created_at DESC",
            "access_count": "m.access_count DESC",
            "confidence": "m.confidence DESC",
        }
        order = sort_map.get(sort_by, "b.created_at DESC")

        sql = f"""
            SELECT b.*, m.confidence, m.source, m.created_at AS m_created_at,
                   m.created_by, m.access_count, m.last_accessed, m.decay_rate, m.ttl, m.session_id,
                       m.org_id, m.project_id, m.agent_id, m.custom_metadata,
                       m.happened_at, m.happened_at_end, m.temporal_precision
            FROM {self.schema}.memblock_blocks b
            LEFT JOIN {self.schema}.memblock_metadata m
                ON b.id = m.block_id AND b.user_id = m.user_id
            WHERE {where_clause}
            ORDER BY {order}
        """

        if "limit" in filters:
            sql += " LIMIT %s"
            params.append(filters["limit"])

        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [self._row_to_block(row) for row in rows]

    def get_all_blocks(self, include_deleted: bool = False) -> list[Block]:
        with self.conn.cursor() as cur:
            if include_deleted:
                cur.execute(f"""
                    SELECT b.*, m.confidence, m.source, m.created_at AS m_created_at,
                           m.created_by, m.access_count, m.last_accessed, m.decay_rate, m.ttl, m.session_id,
                       m.org_id, m.project_id, m.agent_id, m.custom_metadata,
                       m.happened_at, m.happened_at_end, m.temporal_precision
                    FROM {self.schema}.memblock_blocks b
                    LEFT JOIN {self.schema}.memblock_metadata m
                        ON b.id = m.block_id AND b.user_id = m.user_id
                    WHERE b.user_id = %s
                    ORDER BY b.created_at DESC
                """, (self.user_id,))
            else:
                cur.execute(f"""
                    SELECT b.*, m.confidence, m.source, m.created_at AS m_created_at,
                           m.created_by, m.access_count, m.last_accessed, m.decay_rate, m.ttl, m.session_id,
                       m.org_id, m.project_id, m.agent_id, m.custom_metadata,
                       m.happened_at, m.happened_at_end, m.temporal_precision
                    FROM {self.schema}.memblock_blocks b
                    LEFT JOIN {self.schema}.memblock_metadata m
                        ON b.id = m.block_id AND b.user_id = m.user_id
                    WHERE b.user_id = %s AND b.deleted = FALSE
                    ORDER BY b.created_at DESC
                """, (self.user_id,))
            rows = cur.fetchall()

        return [self._row_to_block(row) for row in rows]

    # ─── Edge Operations ──────────────────────────────────────────────────

    def save_edge(self, edge: Edge) -> None:
        with self.conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {self.schema}.memblock_edges
                    (id, user_id, source_id, target_id, relation, weight, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id, user_id) DO UPDATE SET
                    source_id = EXCLUDED.source_id,
                    target_id = EXCLUDED.target_id,
                    relation = EXCLUDED.relation,
                    weight = EXCLUDED.weight
            """, (
                edge.id,
                self.user_id,
                edge.source_id,
                edge.target_id,
                edge.relation.value,
                edge.weight,
                edge.created_at.isoformat(),
            ))
        self.conn.commit()

    def get_edges(self, block_id: str, direction: str = "both") -> list[Edge]:
        with self.conn.cursor() as cur:
            if direction == "outgoing":
                cur.execute(f"""
                    SELECT * FROM {self.schema}.memblock_edges
                    WHERE user_id = %s AND source_id = %s
                """, (self.user_id, block_id))
            elif direction == "incoming":
                cur.execute(f"""
                    SELECT * FROM {self.schema}.memblock_edges
                    WHERE user_id = %s AND target_id = %s
                """, (self.user_id, block_id))
            else:
                cur.execute(f"""
                    SELECT * FROM {self.schema}.memblock_edges
                    WHERE user_id = %s AND (source_id = %s OR target_id = %s)
                """, (self.user_id, block_id, block_id))
            rows = cur.fetchall()

        return [self._row_to_edge(row) for row in rows]

    def delete_edge(self, edge_id: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(f"""
                DELETE FROM {self.schema}.memblock_edges
                WHERE id = %s AND user_id = %s
            """, (edge_id, self.user_id))
        self.conn.commit()

    def delete_edges_for_block(self, block_id: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(f"""
                DELETE FROM {self.schema}.memblock_edges
                WHERE user_id = %s AND (source_id = %s OR target_id = %s)
            """, (self.user_id, block_id, block_id))
        self.conn.commit()

    # ─── Operation Log ────────────────────────────────────────────────────

    def save_operation(self, op: Operation) -> None:
        with self.conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {self.schema}.memblock_operations
                    (id, user_id, block_id, author, clock, action, data, hash, prev_hash, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                op.id,
                self.user_id,
                op.block_id,
                op.author,
                op.clock,
                op.action.value,
                json.dumps(op.data),
                op.hash,
                op.prev_hash,
                op.timestamp.isoformat(),
            ))
        self.conn.commit()

    def get_operations(self, block_id: str | None = None) -> list[Operation]:
        with self.conn.cursor() as cur:
            if block_id:
                cur.execute(f"""
                    SELECT * FROM {self.schema}.memblock_operations
                    WHERE user_id = %s AND block_id = %s
                    ORDER BY clock ASC
                """, (self.user_id, block_id))
            else:
                cur.execute(f"""
                    SELECT * FROM {self.schema}.memblock_operations
                    WHERE user_id = %s
                    ORDER BY clock ASC
                """, (self.user_id,))
            rows = cur.fetchall()

        return [self._row_to_operation(row) for row in rows]

    def get_last_operation(self) -> Operation | None:
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT * FROM {self.schema}.memblock_operations
                WHERE user_id = %s
                ORDER BY clock DESC
                LIMIT 1
            """, (self.user_id,))
            row = cur.fetchone()

        if row is None:
            return None
        return self._row_to_operation(row)

    # ─── Embedding Operations ────────────────────────────────────────────

    def _ensure_pgvector_index(self, dims: int) -> None:
        """Create HNSW index on pgvector table once dimensions are known."""
        if self._embedding_dims is not None:
            return
        self._embedding_dims = dims
        with self.conn.cursor() as cur:
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_mb_embeddings_vec_hnsw
                ON {self.schema}.memblock_embeddings_vec
                USING hnsw (embedding vector_cosine_ops)
            """)
        self.conn.commit()

    def save_embedding(self, block_id: str, embedding: bytes) -> None:
        # Always write to BYTEA table (backward compat)
        with self.conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {self.schema}.memblock_embeddings (block_id, user_id, embedding)
                VALUES (%s, %s, %s)
                ON CONFLICT (block_id, user_id) DO UPDATE SET embedding = EXCLUDED.embedding
            """, (block_id, self.user_id, embedding))
        self.conn.commit()

        # Dual-write to pgvector table if available
        if self._has_pgvector:
            from memblock.embeddings import unpack_embedding
            vec = unpack_embedding(embedding)
            self._ensure_pgvector_index(len(vec))
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {self.schema}.memblock_embeddings_vec
                        (block_id, user_id, embedding)
                    VALUES (%s, %s, %s::vector)
                    ON CONFLICT (block_id, user_id)
                    DO UPDATE SET embedding = EXCLUDED.embedding
                """, (block_id, self.user_id, str(vec)))
            self.conn.commit()

    def get_embedding(self, block_id: str) -> bytes | None:
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT embedding FROM {self.schema}.memblock_embeddings
                WHERE block_id = %s AND user_id = %s
            """, (block_id, self.user_id))
            row = cur.fetchone()
        return bytes(row["embedding"]) if row else None

    def get_all_embeddings(self) -> list[tuple[str, bytes]]:
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT e.block_id, e.embedding
                FROM {self.schema}.memblock_embeddings e
                INNER JOIN {self.schema}.memblock_blocks b
                    ON e.block_id = b.id AND e.user_id = b.user_id
                WHERE e.user_id = %s AND b.deleted = FALSE
            """, (self.user_id,))
            rows = cur.fetchall()
        return [(row["block_id"], bytes(row["embedding"])) for row in rows]

    def delete_embedding(self, block_id: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(f"""
                DELETE FROM {self.schema}.memblock_embeddings
                WHERE block_id = %s AND user_id = %s
            """, (block_id, self.user_id))
            if self._has_pgvector:
                cur.execute(f"""
                    DELETE FROM {self.schema}.memblock_embeddings_vec
                    WHERE block_id = %s AND user_id = %s
                """, (block_id, self.user_id))
        self.conn.commit()

    def search_similar_embeddings(
        self, query_embedding: bytes, limit: int = 20,
    ) -> list[tuple[str, float]]:
        """
        Server-side cosine similarity search using pgvector.

        Returns list of (block_id, similarity_score) sorted by similarity desc.
        Returns empty list if pgvector is not available.
        """
        if not self._has_pgvector:
            return []

        from memblock.embeddings import unpack_embedding
        query_vec = unpack_embedding(query_embedding)
        vec_str = str(query_vec)

        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT e.block_id, 1 - (e.embedding <=> %s::vector) AS similarity
                FROM {self.schema}.memblock_embeddings_vec e
                INNER JOIN {self.schema}.memblock_blocks b
                    ON e.block_id = b.id AND e.user_id = b.user_id
                WHERE e.user_id = %s AND b.deleted = FALSE
                ORDER BY e.embedding <=> %s::vector
                LIMIT %s
            """, (vec_str, self.user_id, vec_str, limit))
            rows = cur.fetchall()

        return [(row["block_id"], float(row["similarity"])) for row in rows]

    # ─── Analytics Operations ─────────────────────────────────────────────

    def initialize_analytics_tables(self) -> None:
        """Create analytics tables if they don't exist."""
        with self.conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.memblock_org_questions (
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
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.memblock_org_question_users (
                    org_id TEXT NOT NULL,
                    question_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    PRIMARY KEY (org_id, question_id, user_id),
                    FOREIGN KEY (question_id)
                        REFERENCES {self.schema}.memblock_org_questions(id) ON DELETE CASCADE
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.memblock_org_question_events (
                    id BIGSERIAL PRIMARY KEY,
                    org_id TEXT NOT NULL,
                    question_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    asked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    FOREIGN KEY (question_id)
                        REFERENCES {self.schema}.memblock_org_questions(id) ON DELETE CASCADE
                )
            """)
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_mb_org_questions_org "
                f"ON {self.schema}.memblock_org_questions(org_id)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_mb_org_questions_freq "
                f"ON {self.schema}.memblock_org_questions(org_id, frequency_count DESC)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_mb_org_question_events_time "
                f"ON {self.schema}.memblock_org_question_events(org_id, asked_at)"
            )
        self.conn.commit()

    def upsert_question(
        self, org_id: str, normalized_text: str, user_id: str, asked_at: Any,
    ) -> dict[str, Any]:
        """Upsert a question and return the question record dict."""
        import uuid as _uuid

        asked_at_val = asked_at.isoformat() if hasattr(asked_at, "isoformat") else str(asked_at)

        with self.conn.cursor() as cur:
            # Upsert the question
            question_id = f"qa_{_uuid.uuid4().hex[:12]}"
            cur.execute(f"""
                INSERT INTO {self.schema}.memblock_org_questions
                    (id, org_id, normalized_text, frequency_count, unique_user_count, first_asked, last_asked)
                VALUES (%s, %s, %s, 1, 1, %s, %s)
                ON CONFLICT (org_id, normalized_text) DO UPDATE SET
                    frequency_count = {self.schema}.memblock_org_questions.frequency_count + 1,
                    last_asked = EXCLUDED.last_asked
                RETURNING id
            """, (question_id, org_id, normalized_text, asked_at_val, asked_at_val))
            question_id = cur.fetchone()["id"]

            # Insert user if new
            cur.execute(f"""
                INSERT INTO {self.schema}.memblock_org_question_users (org_id, question_id, user_id)
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (org_id, question_id, user_id))

            # Update unique_user_count
            cur.execute(f"""
                UPDATE {self.schema}.memblock_org_questions
                SET unique_user_count = (
                    SELECT COUNT(*) FROM {self.schema}.memblock_org_question_users
                    WHERE question_id = %s
                )
                WHERE id = %s
            """, (question_id, question_id))

            # Log the event
            cur.execute(f"""
                INSERT INTO {self.schema}.memblock_org_question_events
                    (org_id, question_id, user_id, asked_at)
                VALUES (%s, %s, %s, %s)
            """, (org_id, question_id, user_id, asked_at_val))

            # Return the record
            cur.execute(f"""
                SELECT * FROM {self.schema}.memblock_org_questions WHERE id = %s
            """, (question_id,))
            q = cur.fetchone()

        self.conn.commit()
        return {
            "id": q["id"],
            "org_id": q["org_id"],
            "normalized_text": q["normalized_text"],
            "frequency_count": q["frequency_count"],
            "unique_user_count": q["unique_user_count"],
            "first_asked": q["first_asked"].isoformat() if hasattr(q["first_asked"], "isoformat") else q["first_asked"],
            "last_asked": q["last_asked"].isoformat() if hasattr(q["last_asked"], "isoformat") else q["last_asked"],
        }

    def get_top_questions(
        self, org_id: str, limit: int = 20, since: Any = None, until: Any = None,
    ) -> list[dict[str, Any]]:
        """Get top questions by frequency."""
        with self.conn.cursor() as cur:
            if since or until:
                conditions = ["e.org_id = %s"]
                params: list[Any] = [org_id]
                if since:
                    conditions.append("e.asked_at >= %s")
                    params.append(since.isoformat() if hasattr(since, "isoformat") else str(since))
                if until:
                    conditions.append("e.asked_at <= %s")
                    params.append(until.isoformat() if hasattr(until, "isoformat") else str(until))
                where = " AND ".join(conditions)
                cur.execute(f"""
                    SELECT q.*, COUNT(e.id) as period_count
                    FROM {self.schema}.memblock_org_questions q
                    INNER JOIN {self.schema}.memblock_org_question_events e ON q.id = e.question_id
                    WHERE {where}
                    GROUP BY q.id
                    ORDER BY period_count DESC LIMIT %s
                """, params + [limit])
            else:
                cur.execute(f"""
                    SELECT * FROM {self.schema}.memblock_org_questions
                    WHERE org_id = %s ORDER BY frequency_count DESC LIMIT %s
                """, (org_id, limit))
            rows = cur.fetchall()

        return [
            {
                "id": r["id"],
                "org_id": r["org_id"],
                "normalized_text": r["normalized_text"],
                "frequency_count": r["frequency_count"],
                "unique_user_count": r["unique_user_count"],
                "first_asked": r["first_asked"].isoformat() if hasattr(r["first_asked"], "isoformat") else r["first_asked"],
                "last_asked": r["last_asked"].isoformat() if hasattr(r["last_asked"], "isoformat") else r["last_asked"],
            }
            for r in rows
        ]

    def get_question_events(
        self, org_id: str, question_id: str, since: Any = None, until: Any = None,
    ) -> list[dict[str, Any]]:
        """Get individual question events for time breakdown."""
        conditions = ["org_id = %s", "question_id = %s"]
        params: list[Any] = [org_id, question_id]
        if since:
            conditions.append("asked_at >= %s")
            params.append(since.isoformat() if hasattr(since, "isoformat") else str(since))
        if until:
            conditions.append("asked_at <= %s")
            params.append(until.isoformat() if hasattr(until, "isoformat") else str(until))
        where = " AND ".join(conditions)

        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT * FROM {self.schema}.memblock_org_question_events
                WHERE {where} ORDER BY asked_at ASC
            """, params)
            rows = cur.fetchall()

        return [
            {
                "id": r["id"],
                "org_id": r["org_id"],
                "question_id": r["question_id"],
                "user_id": r["user_id"],
                "asked_at": r["asked_at"].isoformat() if hasattr(r["asked_at"], "isoformat") else r["asked_at"],
            }
            for r in rows
        ]

    def get_question_by_text(
        self, org_id: str, normalized_text: str,
    ) -> dict[str, Any] | None:
        """Lookup a question by its normalized text."""
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT * FROM {self.schema}.memblock_org_questions
                WHERE org_id = %s AND normalized_text = %s
            """, (org_id, normalized_text))
            row = cur.fetchone()

        if row is None:
            return None
        return {
            "id": row["id"],
            "org_id": row["org_id"],
            "normalized_text": row["normalized_text"],
            "frequency_count": row["frequency_count"],
            "unique_user_count": row["unique_user_count"],
            "first_asked": row["first_asked"].isoformat() if hasattr(row["first_asked"], "isoformat") else row["first_asked"],
            "last_asked": row["last_asked"].isoformat() if hasattr(row["last_asked"], "isoformat") else row["last_asked"],
        }

    def get_all_questions(self, org_id: str) -> list[dict[str, Any]]:
        """Get all tracked questions for an org."""
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT * FROM {self.schema}.memblock_org_questions
                WHERE org_id = %s ORDER BY frequency_count DESC
            """, (org_id,))
            rows = cur.fetchall()

        return [
            {
                "id": r["id"],
                "org_id": r["org_id"],
                "normalized_text": r["normalized_text"],
                "frequency_count": r["frequency_count"],
                "unique_user_count": r["unique_user_count"],
                "first_asked": r["first_asked"].isoformat() if hasattr(r["first_asked"], "isoformat") else r["first_asked"],
                "last_asked": r["last_asked"].isoformat() if hasattr(r["last_asked"], "isoformat") else r["last_asked"],
            }
            for r in rows
        ]

    # ─── Lifecycle ────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            if self._pool_conn and self._pool is not None:
                self._pool.putconn(self._conn)
            else:
                self._conn.close()
            self._conn = None
            self._pool_conn = False

    # ─── Internal Helpers ─────────────────────────────────────────────────

    def _row_to_block(self, row: dict) -> Block:
        children_ids = row["children_ids"]
        if isinstance(children_ids, str):
            children_ids = json.loads(children_ids)

        tags = row["tags"]
        if isinstance(tags, str):
            tags = json.loads(tags)

        m_created_at = row.get("m_created_at") or row["created_at"]
        if isinstance(m_created_at, str):
            m_created_at = datetime.fromisoformat(m_created_at)

        created_at_val = row["created_at"]
        if isinstance(created_at_val, str):
            created_at_val = datetime.fromisoformat(created_at_val)

        last_accessed = row.get("last_accessed")
        if isinstance(last_accessed, str):
            last_accessed = datetime.fromisoformat(last_accessed)

        return Block(
            id=row["id"],
            type=BlockType(row["type"]),
            content=row["content"],
            metadata=BlockMetadata(
                confidence=row.get("confidence") or 1.0,
                source=SourceType(row["source"]) if row.get("source") else SourceType.EXPLICIT,
                created_at=m_created_at,
                created_by=row.get("created_by") or "agent",
                access_count=row.get("access_count") or 0,
                last_accessed=last_accessed,
                decay_rate=row.get("decay_rate") or 0.01,
                ttl=row.get("ttl"),
                session_id=row.get("session_id"),
                org_id=row.get("org_id"),
                project_id=row.get("project_id"),
                agent_id=row.get("agent_id"),
                custom_metadata=row.get("custom_metadata") if isinstance(row.get("custom_metadata"), dict) else (json.loads(row["custom_metadata"]) if row.get("custom_metadata") else None),
                happened_at=datetime.fromisoformat(row["happened_at"]) if (row.get("happened_at") and isinstance(row["happened_at"], str)) else (row.get("happened_at") if isinstance(row.get("happened_at"), datetime) else None),
                happened_at_end=datetime.fromisoformat(row["happened_at_end"]) if (row.get("happened_at_end") and isinstance(row["happened_at_end"], str)) else (row.get("happened_at_end") if isinstance(row.get("happened_at_end"), datetime) else None),
                temporal_precision=row.get("temporal_precision") or "exact",
            ),
            encryption_level=EncryptionLevel(row["encryption_level"]),
            encrypted=bool(row["encrypted"]),
            parent_id=row["parent_id"],
            children_ids=children_ids,
            version=row["version"],
            op_hash=row["op_hash"],
            tags=tags,
            content_hash=row.get("content_hash", ""),
            deleted=bool(row["deleted"]),
        )

    def get_block_by_content_hash(self, content_hash: str) -> Block | None:
        """Find a non-deleted block by content hash."""
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT b.*, m.confidence, m.source, m.created_at AS m_created_at,
                       m.created_by, m.access_count, m.last_accessed, m.decay_rate, m.ttl, m.session_id,
                       m.org_id, m.project_id, m.agent_id, m.custom_metadata,
                       m.happened_at, m.happened_at_end, m.temporal_precision
                FROM {self.schema}.memblock_blocks b
                LEFT JOIN {self.schema}.memblock_metadata m
                    ON b.id = m.block_id AND b.user_id = m.user_id
                WHERE b.content_hash = %s AND b.user_id = %s AND b.deleted = FALSE
                LIMIT 1
            """, (content_hash, self.user_id))
            row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_block(row)

    def _row_to_edge(self, row: dict) -> Edge:
        created_at = row["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return Edge(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            relation=EdgeRelation(row["relation"]),
            weight=row["weight"],
            created_at=created_at,
        )

    def _row_to_operation(self, row: dict) -> Operation:
        data = row["data"]
        if isinstance(data, str):
            data = json.loads(data)

        timestamp = row["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return Operation(
            id=row["id"],
            block_id=row["block_id"],
            author=row["author"],
            clock=row["clock"],
            action=OpAction(row["action"]),
            data=data,
            hash=row["hash"],
            prev_hash=row["prev_hash"],
            timestamp=timestamp,
        )
