"""Tests for the schema migration system."""

import sqlite3
import pytest

from memblock.errors import MigrationError
from memblock.migrations import SCHEMA_VERSION, MigrationRunner, MIGRATIONS
from memblock.storage.sqlite import SQLiteAdapter


class TestMigrationRunner:
    """Test migration runner with SQLite."""

    def test_fresh_db_reaches_latest_version(self):
        """A fresh database should be at the latest schema version after initialize."""
        adapter = SQLiteAdapter(":memory:")
        adapter.initialize()

        cur = adapter.conn.cursor()
        cur.execute("SELECT version FROM schema_version LIMIT 1")
        row = cur.fetchone()
        assert row is not None
        assert row[0] == SCHEMA_VERSION
        adapter.close()

    def test_fresh_db_has_content_hash_column(self):
        """After migration, blocks table should have content_hash column."""
        adapter = SQLiteAdapter(":memory:")
        adapter.initialize()

        cur = adapter.conn.cursor()
        cur.execute("PRAGMA table_info(blocks)")
        columns = {row[1] for row in cur.fetchall()}
        assert "content_hash" in columns
        adapter.close()

    def test_fresh_db_has_content_hash_index(self):
        """After migration, content_hash index should exist."""
        adapter = SQLiteAdapter(":memory:")
        adapter.initialize()

        cur = adapter.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_blocks_content_hash'")
        assert cur.fetchone() is not None
        adapter.close()

    def test_v010_db_detected_and_migrated(self):
        """A v0.1.0 database (tables but no version table) should be detected and migrated."""
        # Simulate v0.1.0: create tables manually without version table
        conn = sqlite3.connect(":memory:")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("""
            CREATE TABLE blocks (
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
            )
        """)
        conn.execute("""
            CREATE TABLE block_metadata (
                block_id TEXT PRIMARY KEY,
                confidence REAL NOT NULL DEFAULT 1.0,
                source TEXT NOT NULL DEFAULT 'explicit',
                created_at TEXT NOT NULL,
                created_by TEXT NOT NULL DEFAULT 'agent',
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed TEXT,
                decay_rate REAL NOT NULL DEFAULT 0.01,
                ttl INTEGER,
                FOREIGN KEY (block_id) REFERENCES blocks(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE edges (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE operations (
                id TEXT PRIMARY KEY,
                block_id TEXT NOT NULL DEFAULT '',
                author TEXT NOT NULL DEFAULT 'agent',
                clock INTEGER NOT NULL DEFAULT 0,
                action TEXT NOT NULL,
                data TEXT NOT NULL DEFAULT '{}',
                hash TEXT NOT NULL DEFAULT '',
                prev_hash TEXT NOT NULL DEFAULT '',
                timestamp TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS blocks_fts
            USING fts5(block_id, content, tags, tokenize='porter')
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS block_embeddings (
                block_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                FOREIGN KEY (block_id) REFERENCES blocks(id) ON DELETE CASCADE
            )
        """)
        # Insert a test block to verify backfill
        conn.execute(
            "INSERT INTO blocks (id, type, content, created_at) VALUES (?, ?, ?, ?)",
            ("blk_test123", "fact", "Test content", "2024-01-01T00:00:00"),
        )
        conn.commit()

        # Now create adapter that points to this connection
        adapter = SQLiteAdapter(":memory:")
        adapter._conn = conn
        adapter._conn.row_factory = sqlite3.Row

        # Run migrations
        MigrationRunner(adapter).run()

        cur = conn.cursor()
        # Check version is latest
        cur.execute("SELECT version FROM schema_version LIMIT 1")
        assert cur.fetchone()[0] == SCHEMA_VERSION

        # Check content_hash column was added and backfilled
        cur.execute("SELECT content_hash FROM blocks WHERE id = 'blk_test123'")
        row = cur.fetchone()
        assert row is not None
        assert row[0] != ""  # Should have a hash value

        conn.close()

    def test_idempotent_initialize(self):
        """Running initialize() twice should not fail."""
        adapter = SQLiteAdapter(":memory:")
        adapter.initialize()
        # Second call should be a no-op
        adapter.initialize()

        cur = adapter.conn.cursor()
        cur.execute("SELECT version FROM schema_version LIMIT 1")
        assert cur.fetchone()[0] == SCHEMA_VERSION
        adapter.close()

    def test_future_version_guard(self):
        """Database with version > SCHEMA_VERSION should raise MigrationError."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")

        # Create blocks table and version table with future version
        conn.execute("""
            CREATE TABLE blocks (
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
                created_at TEXT NOT NULL,
                content_hash TEXT DEFAULT ''
            )
        """)
        conn.execute(
            "CREATE TABLE schema_version (version INTEGER NOT NULL, updated_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO schema_version (version, updated_at) VALUES (?, datetime('now'))",
            (999,),
        )
        conn.commit()

        adapter = SQLiteAdapter(":memory:")
        adapter._conn = conn

        with pytest.raises(MigrationError, match="newer than supported"):
            MigrationRunner(adapter).run()

        conn.close()

    def test_corrupted_db_raises_error(self):
        """A blocks table missing expected columns should raise MigrationError."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")

        # Create a blocks table that's missing required columns
        conn.execute("CREATE TABLE blocks (id TEXT PRIMARY KEY, foo TEXT)")
        conn.commit()

        adapter = SQLiteAdapter(":memory:")
        adapter._conn = conn

        with pytest.raises(MigrationError, match="missing expected columns"):
            MigrationRunner(adapter).run()

        conn.close()

    def test_migration_count_matches_schema_version(self):
        """The number of migrations should be consistent with SCHEMA_VERSION."""
        # Version 1 is the base (v0.1.0), migrations start at 2
        assert len(MIGRATIONS) == SCHEMA_VERSION - 1
        for i, m in enumerate(MIGRATIONS):
            assert m.version == i + 2
