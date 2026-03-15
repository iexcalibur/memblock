"""Event hooks — register callbacks for memory lifecycle events."""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any, Callable


class EventType(str, Enum):
    """Memory lifecycle events."""

    ON_ADD = "on_add"
    ON_UPDATE = "on_update"
    ON_DELETE = "on_delete"
    ON_QUERY = "on_query"


class HookManager:
    """
    Manages event hooks for memory operations.

    Usage:
        hooks = HookManager()
        hooks.register(EventType.ON_ADD, lambda data: print(f"Added: {data['block_id']}"))
        hooks.emit(EventType.ON_ADD, {"block_id": "blk_123", "content": "..."})
    """

    def __init__(self) -> None:
        self._hooks: dict[EventType, list[Callable]] = {e: [] for e in EventType}
        self._async_hooks: dict[EventType, list[Callable]] = {e: [] for e in EventType}

    def register(self, event: EventType, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register a synchronous callback for an event."""
        if isinstance(event, str):
            event = EventType(event)
        self._hooks[event].append(callback)

    def register_async(self, event: EventType, callback: Callable) -> None:
        """Register an async callback for an event."""
        if isinstance(event, str):
            event = EventType(event)
        self._async_hooks[event].append(callback)

    def unregister(self, event: EventType, callback: Callable) -> None:
        """Remove a callback from an event."""
        if isinstance(event, str):
            event = EventType(event)
        if callback in self._hooks[event]:
            self._hooks[event].remove(callback)
        if callback in self._async_hooks[event]:
            self._async_hooks[event].remove(callback)

    def emit(self, event: EventType, data: dict[str, Any]) -> None:
        """
        Fire all registered callbacks for an event.

        Sync callbacks are called directly. Async callbacks are scheduled
        if an event loop is running, otherwise they are skipped.
        Errors in callbacks are silently caught — they should never break
        the main operation.
        """
        if isinstance(event, str):
            event = EventType(event)

        # Fire sync hooks
        for callback in self._hooks[event]:
            try:
                callback(data)
            except Exception:
                pass  # hooks must not break the main operation

        # Fire async hooks if event loop is running
        if self._async_hooks[event]:
            try:
                loop = asyncio.get_running_loop()
                for callback in self._async_hooks[event]:
                    loop.create_task(self._safe_async_call(callback, data))
            except RuntimeError:
                pass  # no event loop running — skip async hooks

    @staticmethod
    async def _safe_async_call(callback: Callable, data: dict[str, Any]) -> None:
        """Call an async hook safely."""
        try:
            await callback(data)
        except Exception:
            pass

    def clear(self, event: EventType | None = None) -> None:
        """Clear all hooks, or hooks for a specific event."""
        if event is not None:
            if isinstance(event, str):
                event = EventType(event)
            self._hooks[event] = []
            self._async_hooks[event] = []
        else:
            for e in EventType:
                self._hooks[e] = []
                self._async_hooks[e] = []

    @property
    def has_hooks(self) -> bool:
        """Whether any hooks are registered."""
        return any(
            len(self._hooks[e]) > 0 or len(self._async_hooks[e]) > 0
            for e in EventType
        )
