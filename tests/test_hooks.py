"""Tests for event hooks — HookManager and MemBlock.on() integration."""

import pytest

from memblock import MemBlock, BlockType, EventType
from memblock.hooks import HookManager


class TestHookManager:
    def test_register_and_emit(self):
        hooks = HookManager()
        received = []
        hooks.register(EventType.ON_ADD, lambda data: received.append(data))
        hooks.emit(EventType.ON_ADD, {"block_id": "blk_1"})
        assert len(received) == 1
        assert received[0]["block_id"] == "blk_1"

    def test_multiple_hooks(self):
        hooks = HookManager()
        calls = []
        hooks.register(EventType.ON_ADD, lambda d: calls.append("hook1"))
        hooks.register(EventType.ON_ADD, lambda d: calls.append("hook2"))
        hooks.emit(EventType.ON_ADD, {})
        assert calls == ["hook1", "hook2"]

    def test_different_events(self):
        hooks = HookManager()
        add_calls = []
        delete_calls = []
        hooks.register(EventType.ON_ADD, lambda d: add_calls.append(d))
        hooks.register(EventType.ON_DELETE, lambda d: delete_calls.append(d))
        hooks.emit(EventType.ON_ADD, {"id": 1})
        hooks.emit(EventType.ON_DELETE, {"id": 2})
        assert len(add_calls) == 1
        assert len(delete_calls) == 1

    def test_unregister(self):
        hooks = HookManager()
        calls = []
        cb = lambda d: calls.append(d)
        hooks.register(EventType.ON_ADD, cb)
        hooks.emit(EventType.ON_ADD, {"x": 1})
        assert len(calls) == 1

        hooks.unregister(EventType.ON_ADD, cb)
        hooks.emit(EventType.ON_ADD, {"x": 2})
        assert len(calls) == 1  # No new call

    def test_clear_specific_event(self):
        hooks = HookManager()
        calls = []
        hooks.register(EventType.ON_ADD, lambda d: calls.append("add"))
        hooks.register(EventType.ON_DELETE, lambda d: calls.append("delete"))
        hooks.clear(EventType.ON_ADD)
        hooks.emit(EventType.ON_ADD, {})
        hooks.emit(EventType.ON_DELETE, {})
        assert calls == ["delete"]

    def test_clear_all(self):
        hooks = HookManager()
        calls = []
        hooks.register(EventType.ON_ADD, lambda d: calls.append("add"))
        hooks.register(EventType.ON_DELETE, lambda d: calls.append("delete"))
        hooks.clear()
        hooks.emit(EventType.ON_ADD, {})
        hooks.emit(EventType.ON_DELETE, {})
        assert calls == []

    def test_has_hooks(self):
        hooks = HookManager()
        assert hooks.has_hooks is False
        hooks.register(EventType.ON_ADD, lambda d: None)
        assert hooks.has_hooks is True

    def test_error_in_hook_does_not_propagate(self):
        hooks = HookManager()
        after_calls = []

        def bad_hook(d):
            raise ValueError("Hook error!")

        hooks.register(EventType.ON_ADD, bad_hook)
        hooks.register(EventType.ON_ADD, lambda d: after_calls.append("ok"))

        # Should not raise
        hooks.emit(EventType.ON_ADD, {})
        assert after_calls == ["ok"]

    def test_string_event_names(self):
        hooks = HookManager()
        calls = []
        hooks.register("on_add", lambda d: calls.append(d))
        hooks.emit("on_add", {"test": True})
        assert len(calls) == 1


class TestMemBlockHooksIntegration:
    def test_on_add_hook(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        events = []
        mem.on("on_add", lambda d: events.append(d))
        block = mem.store("Test content", type=BlockType.FACT)
        assert len(events) == 1
        assert events[0]["block_id"] == block.id
        assert events[0]["content"] == "Test content"
        assert events[0]["type"] == "fact"
        mem.close()

    def test_on_update_hook(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        events = []
        mem.on("on_update", lambda d: events.append(d))
        block = mem.store("Original", type=BlockType.FACT)
        mem.update(block.id, content="Updated")
        assert len(events) == 1
        assert events[0]["block_id"] == block.id
        mem.close()

    def test_on_delete_hook(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        events = []
        mem.on("on_delete", lambda d: events.append(d))
        block = mem.store("To delete", type=BlockType.FACT)
        mem.delete(block.id)
        assert len(events) == 1
        assert events[0]["block_id"] == block.id
        mem.close()

    def test_hooks_via_constructor(self):
        events = []
        mem = MemBlock(
            storage="sqlite:///:memory:",
            hooks={"on_add": [lambda d: events.append(d)]},
        )
        mem.store("Hook via constructor", type=BlockType.FACT)
        assert len(events) == 1
        mem.close()

    def test_multiple_hooks_on_same_event(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        calls = []
        mem.on("on_add", lambda d: calls.append("first"))
        mem.on("on_add", lambda d: calls.append("second"))
        mem.store("Test", type=BlockType.FACT)
        assert calls == ["first", "second"]
        mem.close()

    def test_hook_error_does_not_break_store(self):
        mem = MemBlock(storage="sqlite:///:memory:")

        def bad_hook(d):
            raise RuntimeError("Hook failed!")

        mem.on("on_add", bad_hook)
        # store should still succeed
        block = mem.store("Should work", type=BlockType.FACT)
        assert block is not None
        assert block.content == "Should work"
        mem.close()

    def test_hook_receives_block_object(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        events = []
        mem.on("on_add", lambda d: events.append(d))
        block = mem.store("With block", type=BlockType.PREFERENCE)
        assert events[0]["block"].id == block.id
        assert events[0]["block"].type == BlockType.PREFERENCE
        mem.close()

    def test_delete_hook_has_cascade_info(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        events = []
        mem.on("on_delete", lambda d: events.append(d))
        block = mem.store("Test", type=BlockType.FACT)
        mem.delete(block.id, cascade=True)
        assert events[0]["cascade"] is True
        mem.close()
