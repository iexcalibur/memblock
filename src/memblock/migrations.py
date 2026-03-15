"""Schema migration system for MemBlock storage backends."""

from __future__ import annotations

import hashlib
import unicodedata
import re
from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from memblock.storage.base import StorageAdapter


# The latest schema version. Bump this when adding new migrations.
SCHEMA_VERSION = 3


@dataclass
class Migration:
    """A single schema migration step."""

    version: int
    description: str
    up_sqlite: Callable[[Any], None]
    up_postgres: Callable[[Any, str], None]


def _backfill_content_hash_sqlite(cursor: Any) -> None:
    """Compute and store content_hash for all existing blocks (SQLite)."""
    cursor.execute("SELECT id, content FROM blocks")
    rows = cursor.fetchall()
    for row in rows:
        block_id = row[0] if isinstance(row, (list, tuple)) else row["id"]
        content = row[1] if isinstance(row, (list, tuple)) else row["content"]
        content_hash = _compute_content_hash(content)
        cursor.execute(
            "UPDATE blocks SET content_hash = ? WHERE id = ?",
            (content_hash, block_id),
        )


def _backfill_content_hash_postgres(cursor: Any, schema: str) -> None:
    """Compute and store content_hash for all existing blocks (PostgreSQL)."""
    cursor.execute(f"SELECT id, user_id, content FROM {schema}.memblock_blocks")
    rows = cursor.fetchall()
    for row in rows:
        content_hash = _compute_content_hash(row["content"])
        cursor.execute(
            f"UPDATE {schema}.memblock_blocks SET content_hash = %s WHERE id = %s AND user_id = %s",
            (content_hash, row["id"], row["user_id"]),
        )


def _compute_content_hash(content: str) -> str:
    """Normalize content and compute SHA-256 hash."""
    normalized = unicodedata.normalize("NFC", content)
    normalized = normalized.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ─── Migration Registry ──────────────────────────────────────────────────────

MIGRATIONS: list[Migration] = [
    Migration(
        version=2,
        description="Add content_hash column to blocks for deduplication",
        up_sqlite=lambda cur: (
            cur.execute("ALTER TABLE blocks ADD COLUMN content_hash TEXT DEFAULT ''"),
            _backfill_content_hash_sqlite(cur),
        ),
        up_postgres=lambda cur, schema: (
            cur.execute(
                f"ALTER TABLE {schema}.memblock_blocks ADD COLUMN IF NOT EXISTS content_hash TEXT DEFAULT ''"
            ),
            _backfill_content_hash_postgres(cur, schema),
        ),
    ),
    Migration(
        version=3,
        description="Add index on content_hash for fast duplicate lookups",
        up_sqlite=lambda cur: cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_blocks_content_hash ON blocks(content_hash)"
        ),
        up_postgres=lambda cur, schema: cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_mb_blocks_content_hash "
            f"ON {schema}.memblock_blocks(content_hash)"
        ),
    ),
]


# ─── Migration Runner ────────────────────────────────────────────────────────


class MigrationRunner:
    """
    Runs pending schema migrations on a storage adapter.

    Handles:
    - Fresh databases (no tables) → creates version table, runs all migrations
    - v0.1.0 databases (tables exist, no version table) → seeds version=1, runs pending
    - Up-to-date databases → no-op
    - Future databases (version > SCHEMA_VERSION) → raises MigrationError
    - Concurrent protection via exclusive transactions
    """

    def __init__(self, adapter: StorageAdapter) -> None:
        self.adapter = adapter

    def run(self) -> None:
        """Run all pending migrations."""
        from memblock.errors import MigrationError

        adapter_type = self.adapter.adapter_type

        if adapter_type == "sqlite":
            self._run_sqlite()
        elif adapter_type == "postgresql":
            self._run_postgres()
        else:
            raise MigrationError(f"Unknown adapter type: {adapter_type}")

    def _run_sqlite(self) -> None:
        from memblock.errors import MigrationError

        conn = self.adapter._conn  # type: ignore[attr-defined]
        if conn is None:
            return

        # Use BEGIN EXCLUSIVE for migration lock
        conn.execute("BEGIN EXCLUSIVE")
        try:
            cur = conn.cursor()

            # Check if schema_version table exists
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            )
            has_version_table = cur.fetchone() is not None

            if not has_version_table:
                # Create version table
                cur.execute(
                    "CREATE TABLE schema_version (version INTEGER NOT NULL, updated_at TEXT NOT NULL)"
                )

                # Check if this is a v0.1.0 DB (has blocks table) or fresh
                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='blocks'"
                )
                has_blocks = cur.fetchone() is not None

                if has_blocks:
                    # v0.1.0 DB — validate it has expected columns
                    cur.execute("PRAGMA table_info(blocks)")
                    columns = {row[1] for row in cur.fetchall()}
                    expected = {"id", "type", "content", "created_at"}
                    if not expected.issubset(columns):
                        conn.rollback()
                        raise MigrationError(
                            "Detected blocks table but missing expected columns. "
                            "Database may be corrupted or from an incompatible source."
                        )
                    current_version = 1
                else:
                    # Fresh DB
                    current_version = 1  # Base tables already created by initialize()

                cur.execute(
                    "INSERT INTO schema_version (version, updated_at) VALUES (?, datetime('now'))",
                    (current_version,),
                )
            else:
                cur.execute("SELECT version FROM schema_version LIMIT 1")
                row = cur.fetchone()
                current_version = row[0] if row else 0

            # Future version guard
            if current_version > SCHEMA_VERSION:
                conn.rollback()
                raise MigrationError(
                    f"Database schema version {current_version} is newer than "
                    f"supported version {SCHEMA_VERSION}. Upgrade the memblock package."
                )

            # Run pending migrations
            for migration in MIGRATIONS:
                if migration.version > current_version:
                    try:
                        migration.up_sqlite(cur)
                        cur.execute(
                            "UPDATE schema_version SET version = ?, updated_at = datetime('now')",
                            (migration.version,),
                        )
                    except Exception as e:
                        conn.rollback()
                        raise MigrationError(
                            f"Migration to version {migration.version} failed: {e}"
                        ) from e

            conn.commit()
        except MigrationError:
            raise
        except Exception as e:
            conn.rollback()
            raise

    def _run_postgres(self) -> None:
        from memblock.errors import MigrationError

        adapter = self.adapter
        conn = adapter.conn  # type: ignore[attr-defined]
        schema = adapter.schema  # type: ignore[attr-defined]

        with conn.cursor() as cur:
            # Advisory lock to prevent concurrent migrations
            cur.execute("SELECT pg_advisory_lock(2718281828)")

            try:
                # Check if version table exists
                cur.execute(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = %s AND table_name = 'memblock_schema_version')",
                    (schema,),
                )
                has_version_table = cur.fetchone()["exists"]

                if not has_version_table:
                    cur.execute(f"""
                        CREATE TABLE {schema}.memblock_schema_version (
                            version INTEGER NOT NULL,
                            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        )
                    """)

                    # Check if this is a v0.1.0 DB
                    cur.execute(
                        "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                        "WHERE table_schema = %s AND table_name = 'memblock_blocks')",
                        (schema,),
                    )
                    has_blocks = cur.fetchone()["exists"]

                    if has_blocks:
                        # Validate expected columns
                        cur.execute(
                            "SELECT column_name FROM information_schema.columns "
                            "WHERE table_schema = %s AND table_name = 'memblock_blocks'",
                            (schema,),
                        )
                        columns = {row["column_name"] for row in cur.fetchall()}
                        expected = {"id", "type", "content", "created_at"}
                        if not expected.issubset(columns):
                            raise MigrationError(
                                "Detected memblock_blocks table but missing expected columns."
                            )
                        current_version = 1
                    else:
                        current_version = 1

                    cur.execute(
                        f"INSERT INTO {schema}.memblock_schema_version (version) VALUES (%s)",
                        (current_version,),
                    )
                else:
                    cur.execute(f"SELECT version FROM {schema}.memblock_schema_version LIMIT 1")
                    row = cur.fetchone()
                    current_version = row["version"] if row else 0

                # Future version guard
                if current_version > SCHEMA_VERSION:
                    raise MigrationError(
                        f"Database schema version {current_version} is newer than "
                        f"supported version {SCHEMA_VERSION}. Upgrade the memblock package."
                    )

                # Run pending migrations
                for migration in MIGRATIONS:
                    if migration.version > current_version:
                        try:
                            migration.up_postgres(cur, schema)
                            cur.execute(
                                f"UPDATE {schema}.memblock_schema_version "
                                f"SET version = %s, updated_at = NOW()",
                                (migration.version,),
                            )
                        except MigrationError:
                            raise
                        except Exception as e:
                            raise MigrationError(
                                f"Migration to version {migration.version} failed: {e}"
                            ) from e

            finally:
                cur.execute("SELECT pg_advisory_unlock(2718281828)")

        conn.commit()
