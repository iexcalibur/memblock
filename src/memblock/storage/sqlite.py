"""SQLite storage adapter for MemBlock."""

from __future__ import annotations

import json
import sqlite3
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


class SQLiteAdapter(StorageAdapter):
    """
    SQLite-based storage adapter.

    Tables:
    - blocks: core block data
    - block_metadata: metadata for each block
    - edges: relationships between blocks
    - operations: append-only operation log
    - blocks_fts: FTS5 virtual table for text search
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    @property
    def adapter_type(self) -> str:
        return "sqlite"

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def initialize(self) -> None:
        """Create all tables and indexes."""
        cur = self.conn.cursor()

        cur.executescript("""
            CREATE TABLE IF NOT EXISTS blocks (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                encryption_level TEXT NOT NULL DEFAULT 'none',
                encrypted INTEGER NOT NULL DEFAULT 0,
                parent_id TEXT,
                children_ids TEXT NOT NULL DEFAULT '[]',
                version INTEGER NOT NULL DEFAULT 1,
                op_hash TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL DEFAULT '[]',
                deleted INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS block_metadata (
                block_id TEXT PRIMARY KEY,
                confidence REAL NOT NULL DEFAULT 1.0,
                source TEXT NOT NULL DEFAULT 'explicit',
                created_at TEXT NOT NULL,
                created_by TEXT NOT NULL DEFAULT 'agent',
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed TEXT,
                decay_rate REAL NOT NULL DEFAULT 0.01,
                ttl INTEGER,
                session_id TEXT,
                org_id TEXT,
                project_id TEXT,
                agent_id TEXT,
                custom_metadata TEXT,
                FOREIGN KEY (block_id) REFERENCES blocks(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (source_id) REFERENCES blocks(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES blocks(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS operations (
                id TEXT PRIMARY KEY,
                block_id TEXT NOT NULL DEFAULT '',
                author TEXT NOT NULL DEFAULT 'agent',
                clock INTEGER NOT NULL DEFAULT 0,
                action TEXT NOT NULL,
                data TEXT NOT NULL DEFAULT '{}',
                hash TEXT NOT NULL DEFAULT '',
                prev_hash TEXT NOT NULL DEFAULT '',
                timestamp TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_blocks_type ON blocks(type);
            CREATE INDEX IF NOT EXISTS idx_blocks_parent ON blocks(parent_id);
            CREATE INDEX IF NOT EXISTS idx_blocks_deleted ON blocks(deleted);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
            CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation);
            CREATE INDEX IF NOT EXISTS idx_ops_block ON operations(block_id);
            CREATE INDEX IF NOT EXISTS idx_ops_clock ON operations(clock);
            CREATE INDEX IF NOT EXISTS idx_metadata_confidence ON block_metadata(confidence);
        """)

        # FTS5 virtual table for text search
        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS blocks_fts
            USING fts5(block_id, content, tags, tokenize='porter')
        """)

        # Embeddings table for vector search
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS block_embeddings (
                block_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                FOREIGN KEY (block_id) REFERENCES blocks(id) ON DELETE CASCADE
            );
        """)

        self.conn.commit()

        # Run schema migrations
        from memblock.migrations import MigrationRunner
        MigrationRunner(self).run()

    def run_migration_sql(self, sql: str, params: tuple | None = None) -> None:
        cur = self.conn.cursor()
        if params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)

    # ─── Block Operations ─────────────────────────────────────────────────

    def save_block(self, block: Block) -> None:
        cur = self.conn.cursor()

        cur.execute(
            """INSERT OR REPLACE INTO blocks
               (id, type, content, encryption_level, encrypted,
                parent_id, children_ids, version, op_hash, tags, deleted, created_at, content_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                block.id,
                block.type.value,
                block.content,
                block.encryption_level.value,
                int(block.encrypted),
                block.parent_id,
                json.dumps(block.children_ids),
                block.version,
                block.op_hash,
                json.dumps(block.tags),
                int(block.deleted),
                block.metadata.created_at.isoformat(),
                block.content_hash,
            ),
        )

        cur.execute(
            """INSERT OR REPLACE INTO block_metadata
               (block_id, confidence, source, created_at, created_by,
                access_count, last_accessed, decay_rate, ttl, session_id,
                org_id, project_id, agent_id, custom_metadata,
                happened_at, happened_at_end, temporal_precision)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                block.id,
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
            ),
        )

        # Update FTS index
        cur.execute("DELETE FROM blocks_fts WHERE block_id = ?", (block.id,))
        if not block.deleted:
            cur.execute(
                "INSERT INTO blocks_fts (block_id, content, tags) VALUES (?, ?, ?)",
                (block.id, block.content, " ".join(block.tags)),
            )

        self.conn.commit()

    def get_block(self, block_id: str) -> Block | None:
        cur = self.conn.cursor()
        cur.execute(
            """SELECT b.*, m.confidence, m.source, m.created_at as m_created_at,
                      m.created_by, m.access_count, m.last_accessed, m.decay_rate, m.ttl, m.session_id,
                       m.org_id, m.project_id, m.agent_id, m.custom_metadata,
                       m.happened_at, m.happened_at_end, m.temporal_precision
               FROM blocks b
               LEFT JOIN block_metadata m ON b.id = m.block_id
               WHERE b.id = ?""",
            (block_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_block(row)

    def delete_block(self, block_id: str) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM blocks_fts WHERE block_id = ?", (block_id,))
        cur.execute("DELETE FROM block_metadata WHERE block_id = ?", (block_id,))
        cur.execute("DELETE FROM blocks WHERE id = ?", (block_id,))
        self.conn.commit()

    def update_block(self, block_id: str, updates: dict[str, Any]) -> None:
        block = self.get_block(block_id)
        if block is None:
            return

        # Apply updates to block
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
            # Metadata updates
            elif key == "confidence":
                block.metadata.confidence = value
            elif key == "access_count":
                block.metadata.access_count = value
            elif key == "last_accessed":
                block.metadata.last_accessed = value
            elif key == "decay_rate":
                block.metadata.decay_rate = value
            elif key == "happened_at":
                block.metadata.happened_at = value
            elif key == "happened_at_end":
                block.metadata.happened_at_end = value
            elif key == "temporal_precision":
                block.metadata.temporal_precision = value

        self.save_block(block)

    def query_blocks(self, filters: dict[str, Any]) -> list[Block]:
        conditions: list[str] = []
        params: list[Any] = []
        use_fts = False
        fts_query = ""

        # Default: exclude deleted
        if not filters.get("deleted", False):
            conditions.append("b.deleted = 0")

        if "type" in filters:
            block_type = filters["type"]
            if isinstance(block_type, BlockType):
                block_type = block_type.value
            conditions.append("b.type = ?")
            params.append(block_type)

        if "parent_id" in filters:
            conditions.append("b.parent_id = ?")
            params.append(filters["parent_id"])

        if "min_confidence" in filters:
            conditions.append("m.confidence >= ?")
            params.append(filters["min_confidence"])

        if "tags" in filters:
            tag_list = filters["tags"]
            tag_conditions = []
            for tag in tag_list:
                tag_conditions.append("b.tags LIKE ?")
                params.append(f"%{tag}%")
            if tag_conditions:
                conditions.append(f"({' OR '.join(tag_conditions)})")

        if "session_id" in filters:
            conditions.append("m.session_id = ?")
            params.append(filters["session_id"])

        if "org_id" in filters:
            conditions.append("m.org_id = ?")
            params.append(filters["org_id"])

        if "project_id" in filters:
            conditions.append("m.project_id = ?")
            params.append(filters["project_id"])

        if "agent_id" in filters:
            conditions.append("m.agent_id = ?")
            params.append(filters["agent_id"])

        if "metadata_filters" in filters:
            for key, value in filters["metadata_filters"].items():
                conditions.append("json_extract(m.custom_metadata, ?) = ?")
                params.append(f"$.{key}")
                params.append(str(value) if not isinstance(value, str) else value)

        if "happened_after" in filters:
            conditions.append("m.happened_at >= ?")
            params.append(filters["happened_after"].isoformat())

        if "happened_before" in filters:
            conditions.append("m.happened_at <= ?")
            params.append(filters["happened_before"].isoformat())

        if "temporal_range" in filters:
            start, end = filters["temporal_range"]
            conditions.append("m.happened_at >= ? AND m.happened_at <= ?")
            params.extend([start.isoformat(), end.isoformat()])

        if "text_search" in filters:
            use_fts = True
            # Sanitize FTS5 query: remove special characters, wrap terms in quotes
            raw_query = filters["text_search"]
            # Strip FTS5 special chars and split into words
            import re
            words = re.findall(r'\w+', raw_query)
            fts_query = " OR ".join(f'"{w}"' for w in words) if words else '""'

        # Build query
        if use_fts:
            sql = """
                SELECT b.*, m.confidence, m.source, m.created_at as m_created_at,
                       m.created_by, m.access_count, m.last_accessed, m.decay_rate, m.ttl, m.session_id,
                       m.org_id, m.project_id, m.agent_id, m.custom_metadata,
                       m.happened_at, m.happened_at_end, m.temporal_precision
                FROM blocks b
                LEFT JOIN block_metadata m ON b.id = m.block_id
                INNER JOIN blocks_fts fts ON b.id = fts.block_id
                WHERE fts.blocks_fts MATCH ?
            """
            params.insert(0, fts_query)
            if conditions:
                sql += " AND " + " AND ".join(conditions)
        else:
            sql = """
                SELECT b.*, m.confidence, m.source, m.created_at as m_created_at,
                       m.created_by, m.access_count, m.last_accessed, m.decay_rate, m.ttl, m.session_id,
                       m.org_id, m.project_id, m.agent_id, m.custom_metadata,
                       m.happened_at, m.happened_at_end, m.temporal_precision
                FROM blocks b
                LEFT JOIN block_metadata m ON b.id = m.block_id
            """
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)

        # Sort
        sort_by = filters.get("sort_by", "created_at")
        sort_map = {
            "created_at": "b.created_at DESC",
            "access_count": "m.access_count DESC",
            "confidence": "m.confidence DESC",
        }
        sql += f" ORDER BY {sort_map.get(sort_by, 'b.created_at DESC')}"

        # Limit
        if "limit" in filters:
            sql += " LIMIT ?"
            params.append(filters["limit"])

        cur = self.conn.cursor()
        cur.execute(sql, params)
        return [self._row_to_block(row) for row in cur.fetchall()]

    def get_all_blocks(self, include_deleted: bool = False) -> list[Block]:
        cur = self.conn.cursor()
        if include_deleted:
            cur.execute(
                """SELECT b.*, m.confidence, m.source, m.created_at as m_created_at,
                          m.created_by, m.access_count, m.last_accessed, m.decay_rate, m.ttl, m.session_id,
                       m.org_id, m.project_id, m.agent_id, m.custom_metadata,
                       m.happened_at, m.happened_at_end, m.temporal_precision
                   FROM blocks b
                   LEFT JOIN block_metadata m ON b.id = m.block_id
                   ORDER BY b.created_at DESC"""
            )
        else:
            cur.execute(
                """SELECT b.*, m.confidence, m.source, m.created_at as m_created_at,
                          m.created_by, m.access_count, m.last_accessed, m.decay_rate, m.ttl, m.session_id,
                       m.org_id, m.project_id, m.agent_id, m.custom_metadata,
                       m.happened_at, m.happened_at_end, m.temporal_precision
                   FROM blocks b
                   LEFT JOIN block_metadata m ON b.id = m.block_id
                   WHERE b.deleted = 0
                   ORDER BY b.created_at DESC"""
            )
        return [self._row_to_block(row) for row in cur.fetchall()]

    # ─── Edge Operations ──────────────────────────────────────────────────

    def save_edge(self, edge: Edge) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """INSERT OR REPLACE INTO edges (id, source_id, target_id, relation, weight, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                edge.id,
                edge.source_id,
                edge.target_id,
                edge.relation.value,
                edge.weight,
                edge.created_at.isoformat(),
            ),
        )
        self.conn.commit()

    def get_edges(self, block_id: str, direction: str = "both") -> list[Edge]:
        cur = self.conn.cursor()

        if direction == "outgoing":
            cur.execute("SELECT * FROM edges WHERE source_id = ?", (block_id,))
        elif direction == "incoming":
            cur.execute("SELECT * FROM edges WHERE target_id = ?", (block_id,))
        else:  # both
            cur.execute(
                "SELECT * FROM edges WHERE source_id = ? OR target_id = ?",
                (block_id, block_id),
            )

        return [self._row_to_edge(row) for row in cur.fetchall()]

    def delete_edge(self, edge_id: str) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
        self.conn.commit()

    def delete_edges_for_block(self, block_id: str) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "DELETE FROM edges WHERE source_id = ? OR target_id = ?",
            (block_id, block_id),
        )
        self.conn.commit()

    # ─── Operation Log ────────────────────────────────────────────────────

    def save_operation(self, op: Operation) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """INSERT INTO operations (id, block_id, author, clock, action, data, hash, prev_hash, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                op.id,
                op.block_id,
                op.author,
                op.clock,
                op.action.value,
                json.dumps(op.data),
                op.hash,
                op.prev_hash,
                op.timestamp.isoformat(),
            ),
        )
        self.conn.commit()

    def get_operations(self, block_id: str | None = None) -> list[Operation]:
        cur = self.conn.cursor()
        if block_id:
            cur.execute(
                "SELECT * FROM operations WHERE block_id = ? ORDER BY clock ASC",
                (block_id,),
            )
        else:
            cur.execute("SELECT * FROM operations ORDER BY clock ASC")
        return [self._row_to_operation(row) for row in cur.fetchall()]

    def get_last_operation(self) -> Operation | None:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM operations ORDER BY clock DESC LIMIT 1")
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_operation(row)

    # ─── Embedding Operations ────────────────────────────────────────────

    def save_embedding(self, block_id: str, embedding: bytes) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO block_embeddings (block_id, embedding) VALUES (?, ?)",
            (block_id, embedding),
        )
        self.conn.commit()

    def get_embedding(self, block_id: str) -> bytes | None:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT embedding FROM block_embeddings WHERE block_id = ?",
            (block_id,),
        )
        row = cur.fetchone()
        return bytes(row["embedding"]) if row else None

    def get_all_embeddings(self) -> list[tuple[str, bytes]]:
        cur = self.conn.cursor()
        cur.execute(
            """SELECT e.block_id, e.embedding
               FROM block_embeddings e
               INNER JOIN blocks b ON e.block_id = b.id
               WHERE b.deleted = 0"""
        )
        return [(row["block_id"], bytes(row["embedding"])) for row in cur.fetchall()]

    def delete_embedding(self, block_id: str) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM block_embeddings WHERE block_id = ?", (block_id,))
        self.conn.commit()

    # ─── Analytics Operations ─────────────────────────────────────────────

    def initialize_analytics_tables(self) -> None:
        """Create analytics tables if they don't exist (for in-memory DBs that skip migrations)."""
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS org_questions (
                id TEXT PRIMARY KEY,
                org_id TEXT NOT NULL,
                normalized_text TEXT NOT NULL,
                frequency_count INTEGER NOT NULL DEFAULT 1,
                unique_user_count INTEGER NOT NULL DEFAULT 1,
                first_asked TEXT NOT NULL,
                last_asked TEXT NOT NULL,
                UNIQUE(org_id, normalized_text)
            );

            CREATE TABLE IF NOT EXISTS org_question_users (
                org_id TEXT NOT NULL,
                question_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                PRIMARY KEY (org_id, question_id, user_id),
                FOREIGN KEY (question_id) REFERENCES org_questions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS org_question_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                org_id TEXT NOT NULL,
                question_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                asked_at TEXT NOT NULL,
                FOREIGN KEY (question_id) REFERENCES org_questions(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_org_questions_org ON org_questions(org_id);
            CREATE INDEX IF NOT EXISTS idx_org_questions_freq ON org_questions(org_id, frequency_count DESC);
            CREATE INDEX IF NOT EXISTS idx_org_question_events_time ON org_question_events(org_id, asked_at);
        """)
        self.conn.commit()

    def upsert_question(
        self, org_id: str, normalized_text: str, user_id: str, asked_at: Any,
    ) -> dict[str, Any]:
        """Upsert a question and return the question record dict."""
        import uuid as _uuid

        cur = self.conn.cursor()
        asked_at_str = asked_at.isoformat() if hasattr(asked_at, "isoformat") else str(asked_at)

        # Try to find existing question
        cur.execute(
            "SELECT id, frequency_count FROM org_questions WHERE org_id = ? AND normalized_text = ?",
            (org_id, normalized_text),
        )
        row = cur.fetchone()

        if row:
            question_id = row["id"]
            # Increment frequency and update last_asked
            cur.execute(
                "UPDATE org_questions SET frequency_count = frequency_count + 1, last_asked = ? WHERE id = ?",
                (asked_at_str, question_id),
            )
            # Insert user if new (ignore if already exists)
            cur.execute(
                "INSERT OR IGNORE INTO org_question_users (org_id, question_id, user_id) VALUES (?, ?, ?)",
                (org_id, question_id, user_id),
            )
            # Update unique_user_count from the join table
            cur.execute(
                "UPDATE org_questions SET unique_user_count = "
                "(SELECT COUNT(*) FROM org_question_users WHERE question_id = ?) WHERE id = ?",
                (question_id, question_id),
            )
        else:
            question_id = f"qa_{_uuid.uuid4().hex[:12]}"
            cur.execute(
                "INSERT INTO org_questions (id, org_id, normalized_text, frequency_count, unique_user_count, first_asked, last_asked) "
                "VALUES (?, ?, ?, 1, 1, ?, ?)",
                (question_id, org_id, normalized_text, asked_at_str, asked_at_str),
            )
            cur.execute(
                "INSERT INTO org_question_users (org_id, question_id, user_id) VALUES (?, ?, ?)",
                (org_id, question_id, user_id),
            )

        # Always log the event
        cur.execute(
            "INSERT INTO org_question_events (org_id, question_id, user_id, asked_at) VALUES (?, ?, ?, ?)",
            (org_id, question_id, user_id, asked_at_str),
        )
        self.conn.commit()

        # Return the updated record
        cur.execute("SELECT * FROM org_questions WHERE id = ?", (question_id,))
        q = cur.fetchone()
        return {
            "id": q["id"],
            "org_id": q["org_id"],
            "normalized_text": q["normalized_text"],
            "frequency_count": q["frequency_count"],
            "unique_user_count": q["unique_user_count"],
            "first_asked": q["first_asked"],
            "last_asked": q["last_asked"],
        }

    def get_top_questions(
        self, org_id: str, limit: int = 20, since: Any = None, until: Any = None,
    ) -> list[dict[str, Any]]:
        """Get top questions by frequency."""
        cur = self.conn.cursor()

        if since or until:
            # Filter by events in time range, then aggregate
            conditions = ["e.org_id = ?"]
            params: list[Any] = [org_id]
            if since:
                conditions.append("e.asked_at >= ?")
                params.append(since.isoformat() if hasattr(since, "isoformat") else str(since))
            if until:
                conditions.append("e.asked_at <= ?")
                params.append(until.isoformat() if hasattr(until, "isoformat") else str(until))

            where = " AND ".join(conditions)
            cur.execute(
                f"SELECT q.*, COUNT(e.id) as period_count "
                f"FROM org_questions q "
                f"INNER JOIN org_question_events e ON q.id = e.question_id "
                f"WHERE {where} "
                f"GROUP BY q.id "
                f"ORDER BY period_count DESC LIMIT ?",
                params + [limit],
            )
        else:
            cur.execute(
                "SELECT * FROM org_questions WHERE org_id = ? ORDER BY frequency_count DESC LIMIT ?",
                (org_id, limit),
            )

        results = []
        for row in cur.fetchall():
            results.append({
                "id": row["id"],
                "org_id": row["org_id"],
                "normalized_text": row["normalized_text"],
                "frequency_count": row["frequency_count"],
                "unique_user_count": row["unique_user_count"],
                "first_asked": row["first_asked"],
                "last_asked": row["last_asked"],
            })
        return results

    def get_question_events(
        self, org_id: str, question_id: str, since: Any = None, until: Any = None,
    ) -> list[dict[str, Any]]:
        """Get individual question events for time breakdown."""
        cur = self.conn.cursor()
        conditions = ["org_id = ?", "question_id = ?"]
        params: list[Any] = [org_id, question_id]

        if since:
            conditions.append("asked_at >= ?")
            params.append(since.isoformat() if hasattr(since, "isoformat") else str(since))
        if until:
            conditions.append("asked_at <= ?")
            params.append(until.isoformat() if hasattr(until, "isoformat") else str(until))

        where = " AND ".join(conditions)
        cur.execute(
            f"SELECT * FROM org_question_events WHERE {where} ORDER BY asked_at ASC",
            params,
        )
        return [
            {
                "id": row["id"],
                "org_id": row["org_id"],
                "question_id": row["question_id"],
                "user_id": row["user_id"],
                "asked_at": row["asked_at"],
            }
            for row in cur.fetchall()
        ]

    def get_question_by_text(
        self, org_id: str, normalized_text: str,
    ) -> dict[str, Any] | None:
        """Lookup a question by its normalized text."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM org_questions WHERE org_id = ? AND normalized_text = ?",
            (org_id, normalized_text),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "org_id": row["org_id"],
            "normalized_text": row["normalized_text"],
            "frequency_count": row["frequency_count"],
            "unique_user_count": row["unique_user_count"],
            "first_asked": row["first_asked"],
            "last_asked": row["last_asked"],
        }

    def get_all_questions(self, org_id: str) -> list[dict[str, Any]]:
        """Get all tracked questions for an org."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM org_questions WHERE org_id = ? ORDER BY frequency_count DESC",
            (org_id,),
        )
        return [
            {
                "id": row["id"],
                "org_id": row["org_id"],
                "normalized_text": row["normalized_text"],
                "frequency_count": row["frequency_count"],
                "unique_user_count": row["unique_user_count"],
                "first_asked": row["first_asked"],
                "last_asked": row["last_asked"],
            }
            for row in cur.fetchall()
        ]

    # ─── Lifecycle ────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ─── Internal Helpers ─────────────────────────────────────────────────

    def _row_to_block(self, row: sqlite3.Row) -> Block:
        """Convert a database row to a Block object."""
        return Block(
            id=row["id"],
            type=BlockType(row["type"]),
            content=row["content"],
            metadata=BlockMetadata(
                confidence=row["confidence"] if row["confidence"] is not None else 1.0,
                source=SourceType(row["source"]) if row["source"] else SourceType.EXPLICIT,
                created_at=datetime.fromisoformat(row["m_created_at"]) if row["m_created_at"] else datetime.fromisoformat(row["created_at"]),
                created_by=row["created_by"] if row["created_by"] else "agent",
                access_count=row["access_count"] if row["access_count"] is not None else 0,
                last_accessed=datetime.fromisoformat(row["last_accessed"]) if row["last_accessed"] else None,
                decay_rate=row["decay_rate"] if row["decay_rate"] is not None else 0.01,
                ttl=row["ttl"],
                session_id=row["session_id"] if "session_id" in row.keys() else None,
                org_id=row["org_id"] if "org_id" in row.keys() else None,
                project_id=row["project_id"] if "project_id" in row.keys() else None,
                agent_id=row["agent_id"] if "agent_id" in row.keys() else None,
                custom_metadata=json.loads(row["custom_metadata"]) if ("custom_metadata" in row.keys() and row["custom_metadata"]) else None,
                happened_at=datetime.fromisoformat(row["happened_at"]) if ("happened_at" in row.keys() and row["happened_at"]) else None,
                happened_at_end=datetime.fromisoformat(row["happened_at_end"]) if ("happened_at_end" in row.keys() and row["happened_at_end"]) else None,
                temporal_precision=row["temporal_precision"] if ("temporal_precision" in row.keys() and row["temporal_precision"]) else "exact",
            ),
            encryption_level=EncryptionLevel(row["encryption_level"]),
            encrypted=bool(row["encrypted"]),
            parent_id=row["parent_id"],
            children_ids=json.loads(row["children_ids"]),
            version=row["version"],
            op_hash=row["op_hash"],
            tags=json.loads(row["tags"]),
            content_hash=row["content_hash"] if "content_hash" in row.keys() else "",
            deleted=bool(row["deleted"]),
        )

    def get_block_by_content_hash(self, content_hash: str) -> Block | None:
        """Find a non-deleted block by content hash."""
        cur = self.conn.cursor()
        cur.execute(
            """SELECT b.*, m.confidence, m.source, m.created_at as m_created_at,
                      m.created_by, m.access_count, m.last_accessed, m.decay_rate, m.ttl, m.session_id,
                       m.org_id, m.project_id, m.agent_id, m.custom_metadata,
                       m.happened_at, m.happened_at_end, m.temporal_precision
               FROM blocks b
               LEFT JOIN block_metadata m ON b.id = m.block_id
               WHERE b.content_hash = ? AND b.deleted = 0
               LIMIT 1""",
            (content_hash,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_block(row)

    def _row_to_edge(self, row: sqlite3.Row) -> Edge:
        """Convert a database row to an Edge object."""
        return Edge(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            relation=EdgeRelation(row["relation"]),
            weight=row["weight"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _row_to_operation(self, row: sqlite3.Row) -> Operation:
        """Convert a database row to an Operation object."""
        return Operation(
            id=row["id"],
            block_id=row["block_id"],
            author=row["author"],
            clock=row["clock"],
            action=OpAction(row["action"]),
            data=json.loads(row["data"]),
            hash=row["hash"],
            prev_hash=row["prev_hash"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )
