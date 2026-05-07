"""MemBlockPool — LRU-cached MemBlock instance factory for multi-user deployments.

Avoids repeated schema initialization and connection creation by caching
MemBlock instances per user_id with automatic LRU eviction.

Usage:
    # Sync
    with MemBlockPool(storage="postgresql://...", max_instances=100) as pool:
        mem = pool.get("user_123")
        mem.store("User prefers dark mode", type=BlockType.PREFERENCE)

    # Async
    async with AsyncMemBlockPool(storage="postgresql://...") as pool:
        mem = await pool.get("user_123")
        await mem.store("User prefers dark mode", type=BlockType.PREFERENCE)
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any

from memblock.memblock import MemBlock


class MemBlockPool:
    """
    LRU-cached factory for MemBlock instances, keyed by user_id.

    All instances share the same storage URI and configuration.
    When the cache exceeds max_instances, the least-recently-used
    instance is closed and evicted.
    """

    def __init__(
        self,
        storage: str,
        max_instances: int = 100,
        **shared_kwargs: Any,
    ) -> None:
        self._storage = storage
        self._shared_kwargs = shared_kwargs
        self._max = max_instances
        self._instances: OrderedDict[str, MemBlock] = OrderedDict()

    def get(self, user_id: str, **overrides: Any) -> MemBlock:
        """Get or create a MemBlock instance for the given user_id."""
        if user_id in self._instances:
            self._instances.move_to_end(user_id)
            return self._instances[user_id]

        inst = MemBlock(
            storage=self._storage,
            user_id=user_id,
            **self._shared_kwargs,
            **overrides,
        )
        self._instances[user_id] = inst

        # LRU eviction
        if len(self._instances) > self._max:
            _, old = self._instances.popitem(last=False)
            old.close()

        return inst

    def remove(self, user_id: str) -> None:
        """Close and remove a specific user's instance."""
        if user_id in self._instances:
            self._instances.pop(user_id).close()

    @property
    def size(self) -> int:
        """Number of cached instances."""
        return len(self._instances)

    def close_all(self) -> None:
        """Close all cached instances."""
        for inst in self._instances.values():
            inst.close()
        self._instances.clear()

    def __enter__(self) -> MemBlockPool:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close_all()

    def __contains__(self, user_id: str) -> bool:
        return user_id in self._instances

    def __len__(self) -> int:
        return len(self._instances)


class AsyncMemBlockPool:
    """
    Async wrapper around MemBlockPool.

    Returns AsyncMemBlock instances that share the underlying
    sync MemBlock from the pool's LRU cache.
    """

    def __init__(
        self,
        storage: str,
        max_instances: int = 100,
        **shared_kwargs: Any,
    ) -> None:
        self._pool = MemBlockPool(
            storage=storage,
            max_instances=max_instances,
            **shared_kwargs,
        )

    async def get(self, user_id: str, **overrides: Any) -> Any:
        """Get or create an AsyncMemBlock instance for the given user_id."""
        from memblock.async_memblock import AsyncMemBlock

        sync_inst = await asyncio.to_thread(self._pool.get, user_id, **overrides)
        wrapper = AsyncMemBlock.__new__(AsyncMemBlock)
        wrapper._mem = sync_inst
        # `__new__` bypasses `__init__`, so `_native_async` (added in
        # v0.10.0) wouldn't otherwise be set — every async method
        # checks it. Pool always wraps a sync `MemBlock`, so the
        # wrapper is always in legacy mode.
        wrapper._native_async = False
        return wrapper

    async def remove(self, user_id: str) -> None:
        """Close and remove a specific user's instance."""
        await asyncio.to_thread(self._pool.remove, user_id)

    @property
    def size(self) -> int:
        """Number of cached instances."""
        return self._pool.size

    async def close_all(self) -> None:
        """Close all cached instances."""
        await asyncio.to_thread(self._pool.close_all)

    async def __aenter__(self) -> AsyncMemBlockPool:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close_all()

    def __contains__(self, user_id: str) -> bool:
        return user_id in self._pool

    def __len__(self) -> int:
        return len(self._pool)
