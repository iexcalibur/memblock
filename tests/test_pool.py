"""Tests for MemBlockPool and AsyncMemBlockPool."""

import asyncio
import pytest
from memblock import MemBlockPool, AsyncMemBlockPool, BlockType


# ─── MemBlockPool ─────────────────────────────────────────────────────────


class TestMemBlockPool:

    def test_get_creates_instance(self):
        with MemBlockPool(storage="sqlite:///:memory:") as pool:
            mem = pool.get("user_1")
            assert mem is not None
            assert len(pool) == 1

    def test_get_returns_cached_instance(self):
        with MemBlockPool(storage="sqlite:///:memory:") as pool:
            mem1 = pool.get("user_1")
            mem2 = pool.get("user_1")
            assert mem1 is mem2
            assert len(pool) == 1

    def test_get_different_users(self):
        with MemBlockPool(storage="sqlite:///:memory:") as pool:
            mem1 = pool.get("user_1")
            mem2 = pool.get("user_2")
            assert mem1 is not mem2
            assert len(pool) == 2

    def test_lru_eviction(self):
        with MemBlockPool(storage="sqlite:///:memory:", max_instances=2) as pool:
            pool.get("user_1")
            pool.get("user_2")
            pool.get("user_3")  # should evict user_1
            assert len(pool) == 2
            assert "user_1" not in pool
            assert "user_2" in pool
            assert "user_3" in pool

    def test_lru_access_refreshes(self):
        with MemBlockPool(storage="sqlite:///:memory:", max_instances=2) as pool:
            pool.get("user_1")
            pool.get("user_2")
            pool.get("user_1")  # refresh user_1
            pool.get("user_3")  # should evict user_2, not user_1
            assert "user_1" in pool
            assert "user_2" not in pool
            assert "user_3" in pool

    def test_remove(self):
        with MemBlockPool(storage="sqlite:///:memory:") as pool:
            pool.get("user_1")
            pool.get("user_2")
            pool.remove("user_1")
            assert "user_1" not in pool
            assert len(pool) == 1

    def test_remove_nonexistent(self):
        with MemBlockPool(storage="sqlite:///:memory:") as pool:
            pool.remove("nonexistent")  # should not raise
            assert len(pool) == 0

    def test_close_all(self):
        pool = MemBlockPool(storage="sqlite:///:memory:")
        pool.get("user_1")
        pool.get("user_2")
        pool.close_all()
        assert len(pool) == 0

    def test_context_manager(self):
        with MemBlockPool(storage="sqlite:///:memory:") as pool:
            pool.get("user_1")
            assert len(pool) == 1
        # after exit, pool should be cleaned up
        assert len(pool) == 0

    def test_contains(self):
        with MemBlockPool(storage="sqlite:///:memory:") as pool:
            assert "user_1" not in pool
            pool.get("user_1")
            assert "user_1" in pool

    def test_size_property(self):
        with MemBlockPool(storage="sqlite:///:memory:") as pool:
            assert pool.size == 0
            pool.get("user_1")
            assert pool.size == 1

    def test_instances_are_functional(self):
        with MemBlockPool(storage="sqlite:///:memory:") as pool:
            mem = pool.get("user_1")
            block = mem.store("Test content", type=BlockType.FACT)
            assert block is not None
            assert block.content == "Test content"

    def test_shared_kwargs_passed(self):
        with MemBlockPool(
            storage="sqlite:///:memory:",
            author="test_author",
        ) as pool:
            mem = pool.get("user_1")
            block = mem.store("Test", type=BlockType.FACT)
            # The author is used in the operation log
            ops = mem._storage.get_operations(block.id)
            assert ops[0].author == "test_author"


# ─── AsyncMemBlockPool ───────────────────────────────────────────────────


class TestAsyncMemBlockPool:

    def test_async_get_creates_instance(self):
        async def run():
            async with AsyncMemBlockPool(storage="sqlite:///:memory:") as pool:
                mem = await pool.get("user_1")
                assert mem is not None
                assert len(pool) == 1
        asyncio.run(run())

    def test_async_get_cached(self):
        async def run():
            async with AsyncMemBlockPool(storage="sqlite:///:memory:") as pool:
                mem1 = await pool.get("user_1")
                mem2 = await pool.get("user_1")
                # Both should wrap the same underlying MemBlock
                assert mem1._mem is mem2._mem
                assert len(pool) == 1
        asyncio.run(run())

    def test_async_lru_eviction(self):
        async def run():
            async with AsyncMemBlockPool(
                storage="sqlite:///:memory:", max_instances=2
            ) as pool:
                await pool.get("user_1")
                await pool.get("user_2")
                await pool.get("user_3")
                assert len(pool) == 2
                assert "user_1" not in pool
        asyncio.run(run())

    def test_async_remove(self):
        async def run():
            async with AsyncMemBlockPool(storage="sqlite:///:memory:") as pool:
                await pool.get("user_1")
                await pool.remove("user_1")
                assert "user_1" not in pool
        asyncio.run(run())

    def test_async_functional(self):
        async def run():
            async with AsyncMemBlockPool(storage="sqlite:///:memory:") as pool:
                mem = await pool.get("user_1")
                block = await mem.store("Async test", type=BlockType.FACT)
                assert block.content == "Async test"
        asyncio.run(run())

    def test_async_close_all(self):
        async def run():
            pool = AsyncMemBlockPool(storage="sqlite:///:memory:")
            await pool.get("user_1")
            await pool.get("user_2")
            await pool.close_all()
            assert len(pool) == 0
        asyncio.run(run())
