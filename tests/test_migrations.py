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

    def test_failed_migration_rolls_back(self):
        """A migration failure should roll back and raise MigrationError."""
        adapter = SQLiteAdapter(":memory:")
        adapter.initialize()

        # Manually set version back to 1 to force migrations to re-run
        adapter.conn.execute("UPDATE schema_version SET version = 1")
        adapter.conn.commit()

        # Patch migration 2 to fail
        original_up = MIGRATIONS[0].up_sqlite
        MIGRATIONS[0].up_sqlite = lambda cur: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            with pytest.raises(MigrationError, match="failed"):
                MigrationRunner(adapter).run()

            # Version should still be 1 (rolled back)
            cur = adapter.conn.cursor()
            cur.execute("SELECT version FROM schema_version LIMIT 1")
            assert cur.fetchone()[0] == 1
        finally:
            MIGRATIONS[0].up_sqlite = original_up
            adapter.close()


class TestPostgreSQLMigrationPaths:
    """Test PostgreSQL migration paths using mocks (no real DB needed)."""

    def _make_mock_adapter(self):
        """Create a mock adapter that behaves like PostgreSQLAdapter."""
        from unittest.mock import MagicMock

        adapter = MagicMock()
        adapter.adapter_type = "postgresql"
        adapter.schema = "public"

        # Mock cursor as a context manager
        cursor = MagicMock()
        adapter.conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        adapter.conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        return adapter, cursor

    def test_postgres_future_version_guard(self):
        """PostgreSQL: version > SCHEMA_VERSION should raise MigrationError."""
        adapter, cursor = self._make_mock_adapter()

        cursor.fetchone.side_effect = [
            {"exists": True},       # has_version_table
            {"version": 999},       # current_version
            None,                   # advisory_unlock
        ]

        with pytest.raises(MigrationError, match="newer than supported"):
            MigrationRunner(adapter).run()

    def test_postgres_v010_detection_valid(self):
        """PostgreSQL: v0.1.0 DB with valid columns should migrate."""
        adapter, cursor = self._make_mock_adapter()

        # No version table, has blocks table, valid columns
        cursor.fetchone.side_effect = [
            {"exists": False},       # no version table
            {"exists": True},        # has blocks table
            None,                    # advisory_unlock
        ]
        # First fetchall: column validation; second: backfill query (no existing rows)
        cursor.fetchall.side_effect = [
            [
                {"column_name": "id"},
                {"column_name": "type"},
                {"column_name": "content"},
                {"column_name": "created_at"},
                {"column_name": "user_id"},
            ],
            [],  # backfill: no existing blocks to hash
        ]

        # Should not raise — runs migrations successfully
        MigrationRunner(adapter).run()
        adapter.conn.commit.assert_called_once()

    def test_postgres_v010_detection_missing_columns(self):
        """PostgreSQL: blocks table missing expected columns should raise."""
        adapter, cursor = self._make_mock_adapter()

        cursor.fetchone.side_effect = [
            {"exists": False},       # no version table
            {"exists": True},        # has blocks table
            None,                    # advisory_unlock
        ]
        cursor.fetchall.return_value = [
            {"column_name": "id"},
            {"column_name": "foo"},
        ]

        with pytest.raises(MigrationError, match="missing expected columns"):
            MigrationRunner(adapter).run()

    def test_postgres_migration_failure_raises(self):
        """PostgreSQL: a migration that fails should raise MigrationError."""
        adapter, cursor = self._make_mock_adapter()

        cursor.fetchone.side_effect = [
            {"exists": True},       # has_version_table
            {"version": 1},         # current_version
            None,                   # advisory_unlock
        ]

        original_up = MIGRATIONS[0].up_postgres
        MIGRATIONS[0].up_postgres = lambda cur, schema: (_ for _ in ()).throw(
            RuntimeError("pg error")
        )
        try:
            with pytest.raises(MigrationError, match="failed"):
                MigrationRunner(adapter).run()
        finally:
            MIGRATIONS[0].up_postgres = original_up

    def test_postgres_advisory_lock_released_on_success(self):
        """PostgreSQL: advisory lock should be released after successful migration."""
        adapter, cursor = self._make_mock_adapter()

        cursor.fetchone.side_effect = [
            {"exists": True},       # has_version_table
            {"version": SCHEMA_VERSION},  # already up to date
            None,                   # advisory_unlock
        ]

        MigrationRunner(adapter).run()

        # Verify lock was acquired and released
        lock_calls = [
            str(c) for c in cursor.execute.call_args_list
            if "pg_advisory" in str(c)
        ]
        assert len(lock_calls) == 2  # lock + unlock
