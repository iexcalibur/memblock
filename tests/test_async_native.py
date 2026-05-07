"""Tests for the native asyncpg path of `AsyncMemBlock`.

These tests cover the new code introduced in v0.10.0 — the
`postgresql+asyncpg://` URL scheme that triggers `AsyncMemBlock` to
use `AsyncPostgreSQLAdapter`, `AsyncQueryEngine`, and
`AsyncContextBuilder` directly (no `asyncio.to_thread` wrapping
storage I/O).

URL-detection tests run anywhere. Live-DB tests skip cleanly when
`MEMBLOCK_TEST_DB_URL` is unset — see `tests/conftest.py`.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from memblock import AsyncMemBlock, BlockType, EdgeRelation
from memblock.async_memblock import _is_native_async_url
from memblock.hooks import EventType


# ─── Pure-Python URL detection ───────────────────────────────────────


class TestNativeURLDetection:
    """`_is_native_async_url` decides which adapter `AsyncMemBlock`
    uses. Get this wrong and we silently fall back to legacy mode."""

    def test_postgresql_asyncpg_is_native(self):
        assert _is_native_async_url(
            "postgresql+asyncpg://user:pass@host:5432/db"
        )

    def test_plain_postgresql_is_legacy(self):
        assert not _is_native_async_url("postgresql://user:pass@host:5432/db")

    def test_sqlite_is_legacy(self):
        assert not _is_native_async_url("sqlite:///./memory.db")

    def test_none_is_legacy(self):
        assert not _is_native_async_url(None)

    def test_empty_string_is_legacy(self):
        assert not _is_native_async_url("")

    def test_uppercase_scheme_does_not_match(self):
        # PEP-style URL schemes are case-insensitive in some parsers,
        # but our regex is case-sensitive. Document the contract.
        assert not _is_native_async_url(
            "POSTGRESQL+ASYNCPG://user@host/db"
        )

    def test_bare_postgresql_asyncpg_prefix(self):
        # No host portion — still classified by scheme.
        assert _is_native_async_url("postgresql+asyncpg://")


# ─── Live-DB CRUD ────────────────────────────────────────────────────


class TestNativeCRUD:
    """Native-async store / get / update / delete roundtrips."""

    async def test_store_returns_block_with_id(self, mb_native: AsyncMemBlock):
        block = await mb_native.store("CRUD test 1", type=BlockType.FACT)
        assert block is not None
        assert block.id.startswith("blk_")
        assert block.content == "CRUD test 1"
        assert block.type == BlockType.FACT

    async def test_get_roundtrips(self, mb_native: AsyncMemBlock):
        original = await mb_native.store(
            "roundtrip content", type=BlockType.FACT, tags=["round"],
        )
        fetched = await mb_native.get(original.id)
        assert fetched is not None
        assert fetched.id == original.id
        assert fetched.content == "roundtrip content"
        assert "round" in fetched.tags

    async def test_get_missing_returns_none(self, mb_native: AsyncMemBlock):
        assert await mb_native.get("blk_doesnotexist") is None

    async def test_update_changes_content(self, mb_native: AsyncMemBlock):
        block = await mb_native.store("v1", type=BlockType.FACT)
        updated = await mb_native.update(block.id, content="v2")
        assert updated is not None
        assert updated.content == "v2"
        # And persists.
        refetched = await mb_native.get(block.id)
        assert refetched.content == "v2"

    async def test_delete_marks_block_deleted(self, mb_native: AsyncMemBlock):
        block = await mb_native.store("to be deleted", type=BlockType.FACT)
        ok = await mb_native.delete(block.id)
        assert ok is True
        # `get` typically returns None for deleted blocks (depending on
        # adapter). Either way, querying must not surface it.
        hits = await mb_native.query(text_search="to be deleted")
        assert all(b.id != block.id for b in hits)

    async def test_metadata_fields_persist(self, mb_native: AsyncMemBlock):
        block = await mb_native.store(
            "metadata test",
            type=BlockType.FACT,
            confidence=0.42,
            tags=["a", "b"],
            session_id="sess_xyz",
            org_id="org_xyz",
        )
        fetched = await mb_native.get(block.id)
        assert fetched.metadata.confidence == pytest.approx(0.42)
        assert set(fetched.tags) == {"a", "b"}
        assert fetched.metadata.session_id == "sess_xyz"
        assert fetched.metadata.org_id == "org_xyz"


# ─── Live-DB Query / FTS ─────────────────────────────────────────────


class TestNativeQuery:
    """The trigger-scope bug we fixed during schema-isolation testing
    silently broke FTS in custom schemas. These tests pin the
    behaviour so it can never regress."""

    async def test_fts_finds_stored_block(self, mb_native: AsyncMemBlock):
        await mb_native.store(
            "purple elephants march on Tuesday", type=BlockType.FACT,
        )
        hits = await mb_native.query(text_search="purple elephants", limit=5)
        assert any("purple elephants" in b.content for b in hits)

    async def test_filter_by_type(self, mb_native: AsyncMemBlock):
        await mb_native.store("a fact", type=BlockType.FACT)
        await mb_native.store("a preference", type=BlockType.PREFERENCE)
        facts = await mb_native.query(type=BlockType.FACT, limit=10)
        assert all(b.type == BlockType.FACT for b in facts)
        assert any(b.content == "a fact" for b in facts)

    async def test_filter_by_tag(self, mb_native: AsyncMemBlock):
        await mb_native.store("tagged-1", tags=["alpha"])
        await mb_native.store("tagged-2", tags=["beta"])
        alpha = await mb_native.query(tags=["alpha"], limit=10)
        contents = {b.content for b in alpha}
        assert "tagged-1" in contents
        assert "tagged-2" not in contents

    async def test_filter_by_session_id(self, mb_native: AsyncMemBlock):
        sess = f"sess_{uuid.uuid4().hex[:6]}"
        await mb_native.store("scoped", session_id=sess)
        await mb_native.store("unscoped")
        scoped = await mb_native.query(session_id=sess, limit=10)
        assert {b.content for b in scoped} == {"scoped"}


# ─── Live-DB Edges ───────────────────────────────────────────────────


class TestNativeEdges:
    async def test_link_creates_neighbor(self, mb_native: AsyncMemBlock):
        a = await mb_native.store("A node", type=BlockType.FACT)
        b = await mb_native.store("B node", type=BlockType.FACT)

        await mb_native.link(a.id, b.id, relation=EdgeRelation.SUPPORTS)
        nbrs = await mb_native.neighbors(a.id)
        assert any(n.id == b.id for n in nbrs)

    async def test_unlink_removes_neighbor(self, mb_native: AsyncMemBlock):
        a = await mb_native.store("X", type=BlockType.FACT)
        b = await mb_native.store("Y", type=BlockType.FACT)
        await mb_native.link(a.id, b.id, relation=EdgeRelation.SUPPORTS)
        await mb_native.unlink(a.id, b.id)
        nbrs = await mb_native.neighbors(a.id)
        assert all(n.id != b.id for n in nbrs)

    async def test_traverse_walks_multi_hop(self, mb_native: AsyncMemBlock):
        a = await mb_native.store("A", type=BlockType.FACT)
        b = await mb_native.store("B", type=BlockType.FACT)
        c = await mb_native.store("C", type=BlockType.FACT)
        await mb_native.link(a.id, b.id, relation=EdgeRelation.SUPPORTS)
        await mb_native.link(b.id, c.id, relation=EdgeRelation.SUPPORTS)
        reached = await mb_native.traverse(a.id, max_depth=2)
        ids = {b.id for b in reached}
        assert b.id in ids
        assert c.id in ids


# ─── Live-DB Op-log / verify ─────────────────────────────────────────


class TestNativeOpLog:
    async def test_verify_clean_chain_after_writes(
        self, mb_native: AsyncMemBlock,
    ):
        await mb_native.store("op-log entry 1", type=BlockType.FACT)
        await mb_native.store("op-log entry 2", type=BlockType.FACT)
        report = await mb_native.verify()
        assert report.valid, f"chain invalid: {report.message}"
        assert report.total_ops >= 2

    async def test_verify_clean_chain_after_update(
        self, mb_native: AsyncMemBlock,
    ):
        block = await mb_native.store("before", type=BlockType.FACT)
        await mb_native.update(block.id, content="after")
        report = await mb_native.verify()
        assert report.valid
        assert report.total_ops >= 2

    async def test_verify_clean_chain_after_delete(
        self, mb_native: AsyncMemBlock,
    ):
        block = await mb_native.store("doomed", type=BlockType.FACT)
        await mb_native.delete(block.id)
        report = await mb_native.verify()
        assert report.valid


# ─── Schema isolation ────────────────────────────────────────────────


class TestSchemaIsolation:
    """Constructing `AsyncMemBlock(schema='custom')` should:
       (1) bootstrap the schema on first use
       (2) keep all writes inside that schema
       (3) leave the public schema untouched"""

    async def test_custom_schema_bootstraps_and_isolates(
        self, postgres_async_url: str, postgres_sync_url: str,
    ):
        import asyncpg

        schema = f"mb_iso_{uuid.uuid4().hex[:10]}"
        mb = AsyncMemBlock(
            storage=postgres_async_url,
            embeddings=False,
            extract=False,
            schema=schema,
        )

        try:
            block = await mb.store(
                "schema isolation pin", type=BlockType.FACT,
            )

            raw = await asyncpg.connect(postgres_sync_url)
            try:
                # All 10 SDK tables should exist in the custom schema.
                tables = await raw.fetch(
                    """
                    SELECT tablename FROM pg_tables
                     WHERE schemaname = $1 AND tablename LIKE 'memblock_%'
                    """,
                    schema,
                )
                names = {r["tablename"] for r in tables}
                expected = {
                    "memblock_blocks",
                    "memblock_metadata",
                    "memblock_edges",
                    "memblock_operations",
                    "memblock_embeddings",
                    "memblock_embeddings_vec",
                    "memblock_schema_version",
                    "memblock_org_questions",
                    "memblock_org_question_users",
                    "memblock_org_question_events",
                }
                assert names == expected, (
                    f"missing tables in custom schema: {expected - names}"
                )

                # Block row should be in custom schema, not public
                # (or at least not bound to *this* run's id).
                count_custom = await raw.fetchval(
                    f'SELECT COUNT(*) FROM "{schema}".memblock_blocks '
                    f"WHERE id = $1",
                    block.id,
                )
                assert count_custom == 1
            finally:
                await raw.close()
        finally:
            # Best-effort cleanup.
            try:
                await mb.close()
            except Exception:
                pass
            try:
                raw = await asyncpg.connect(postgres_sync_url)
                await raw.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
                await raw.close()
            except Exception:
                pass

    async def test_drop_schema_does_not_touch_public(
        self, postgres_async_url: str, postgres_sync_url: str,
    ):
        import asyncpg

        schema = f"mb_iso_{uuid.uuid4().hex[:10]}"
        mb = AsyncMemBlock(
            storage=postgres_async_url,
            embeddings=False,
            extract=False,
            schema=schema,
        )
        await mb.store("touch", type=BlockType.FACT)

        raw = await asyncpg.connect(postgres_sync_url)
        try:
            before = {
                r["tablename"]
                for r in await raw.fetch(
                    "SELECT tablename FROM pg_tables "
                    "WHERE schemaname = 'public' "
                    "AND tablename LIKE 'memblock_%'"
                )
            }
            await raw.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            after = {
                r["tablename"]
                for r in await raw.fetch(
                    "SELECT tablename FROM pg_tables "
                    "WHERE schemaname = 'public' "
                    "AND tablename LIKE 'memblock_%'"
                )
            }
            assert before == after, (
                f"public mutated by custom-schema drop: "
                f"+{after - before} -{before - after}"
            )
        finally:
            await raw.close()
            try:
                await mb.close()
            except Exception:
                pass


# ─── Hooks emission ──────────────────────────────────────────────────


class TestNativeHooks:
    """Phase 2.13a wired hook emission into the native CRUD path. If
    these tests ever fail it means a refactor silently dropped an
    `_hooks.emit(...)` call."""

    async def test_on_add_fires_for_native_store(
        self, mb_native: AsyncMemBlock,
    ):
        seen: list[dict] = []
        mb_native.on(EventType.ON_ADD, lambda d: seen.append(d))

        block = await mb_native.store("hook-test add", type=BlockType.FACT)

        assert len(seen) == 1
        assert seen[0]["block_id"] == block.id

    async def test_on_update_fires_for_native_update(
        self, mb_native: AsyncMemBlock,
    ):
        seen: list[dict] = []
        mb_native.on(EventType.ON_UPDATE, lambda d: seen.append(d))

        block = await mb_native.store("hook-test before", type=BlockType.FACT)
        await mb_native.update(block.id, content="hook-test after")

        assert len(seen) == 1
        assert seen[0]["block_id"] == block.id

    async def test_on_delete_fires_for_native_delete(
        self, mb_native: AsyncMemBlock,
    ):
        seen: list[dict] = []
        mb_native.on(EventType.ON_DELETE, lambda d: seen.append(d))

        block = await mb_native.store("hook-test doomed", type=BlockType.FACT)
        await mb_native.delete(block.id)

        assert len(seen) == 1
        assert seen[0]["block_id"] == block.id

    async def test_on_query_fires_for_native_query(
        self, mb_native: AsyncMemBlock,
    ):
        seen: list[dict] = []
        mb_native.on(EventType.ON_QUERY, lambda d: seen.append(d))

        await mb_native.store("hook-test query target", type=BlockType.FACT)
        await mb_native.query(text_search="hook-test query target", limit=5)

        assert len(seen) == 1
        assert seen[0]["query_text"] == "hook-test query target"

    async def test_async_hook_callback_runs(self, mb_native: AsyncMemBlock):
        """Coroutine callbacks should fire too — `register_async`
        path."""
        results: list[str] = []

        async def on_add(data: dict) -> None:
            results.append(data["block_id"])

        mb_native.on(EventType.ON_ADD, on_add)
        block = await mb_native.store("async hook", type=BlockType.FACT)

        # Async callbacks dispatch as tasks; give the loop a turn.
        await asyncio.sleep(0)
        # And one more tick — callback may take a turn to schedule.
        await asyncio.sleep(0)

        assert block.id in results
