"""Tests for conflict resolution — ConflictResolver and action parsing."""

import json
import pytest

from memblock.block import Block
from memblock.conflict import (
    ConflictAction,
    ConflictActionType,
    ConflictResolver,
    ConflictResult,
)
from memblock.types import BlockMetadata, BlockType, SourceType


def _make_block(content: str, block_id: str = "blk_test") -> Block:
    """Helper to create a Block for testing."""
    return Block(
        id=block_id,
        content=content,
        type=BlockType.FACT,
        tags=[],
        metadata=BlockMetadata(confidence=0.9, source=SourceType.EXPLICIT),
    )


class FakeLLMProvider:
    """Fake LLM provider that returns pre-configured responses."""

    def __init__(self, response: str = "[]"):
        self._response = response

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._response


class FailingProvider:
    """LLM provider that always raises."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("LLM call failed!")


class TestConflictActionType:
    def test_enum_values(self):
        assert ConflictActionType.ADD == "ADD"
        assert ConflictActionType.UPDATE == "UPDATE"
        assert ConflictActionType.DELETE == "DELETE"
        assert ConflictActionType.NONE == "NONE"


class TestConflictResolver:
    def test_no_existing_blocks_returns_add(self):
        resolver = ConflictResolver(provider=FakeLLMProvider())
        result = resolver.resolve("New info", [])
        assert len(result.actions) == 1
        assert result.actions[0].action == ConflictActionType.ADD
        assert result.actions[0].new_content == "New info"

    def test_add_action(self):
        response = json.dumps([{
            "action": "ADD",
            "block_id": None,
            "new_content": "Brand new info",
            "reason": "No conflict",
        }])
        resolver = ConflictResolver(provider=FakeLLMProvider(response))
        existing = [_make_block("Old info", "blk_1")]
        result = resolver.resolve("Brand new info", existing)
        assert len(result.actions) == 1
        assert result.actions[0].action == ConflictActionType.ADD

    def test_update_action(self):
        response = json.dumps([{
            "action": "UPDATE",
            "block_id": "blk_1",
            "new_content": "Updated info",
            "reason": "Supersedes old",
        }])
        resolver = ConflictResolver(provider=FakeLLMProvider(response))
        existing = [_make_block("Old info", "blk_1")]
        result = resolver.resolve("New info", existing)
        assert len(result.actions) == 1
        assert result.actions[0].action == ConflictActionType.UPDATE
        assert result.actions[0].block_id == "blk_1"
        assert result.actions[0].new_content == "Updated info"

    def test_delete_action(self):
        response = json.dumps([{
            "action": "DELETE",
            "block_id": "blk_1",
            "new_content": None,
            "reason": "Contradicted",
        }])
        resolver = ConflictResolver(provider=FakeLLMProvider(response))
        existing = [_make_block("Wrong info", "blk_1")]
        result = resolver.resolve("Correction", existing)
        assert len(result.actions) == 1
        assert result.actions[0].action == ConflictActionType.DELETE
        assert result.actions[0].block_id == "blk_1"

    def test_none_action(self):
        response = json.dumps([{
            "action": "NONE",
            "block_id": None,
            "new_content": None,
            "reason": "Already known",
        }])
        resolver = ConflictResolver(provider=FakeLLMProvider(response))
        existing = [_make_block("Known info", "blk_1")]
        result = resolver.resolve("Known info", existing)
        assert len(result.actions) == 1
        assert result.actions[0].action == ConflictActionType.NONE

    def test_multiple_actions(self):
        response = json.dumps([
            {"action": "UPDATE", "block_id": "blk_1", "new_content": "Updated", "reason": "Refine"},
            {"action": "DELETE", "block_id": "blk_2", "new_content": None, "reason": "Contradicted"},
        ])
        resolver = ConflictResolver(provider=FakeLLMProvider(response))
        existing = [
            _make_block("Info 1", "blk_1"),
            _make_block("Info 2", "blk_2"),
        ]
        result = resolver.resolve("New info", existing)
        assert len(result.actions) == 2

    def test_invalid_block_id_skipped(self):
        """UPDATE/DELETE with non-existent block_id should be skipped."""
        response = json.dumps([{
            "action": "UPDATE",
            "block_id": "blk_nonexistent",
            "new_content": "Won't happen",
            "reason": "Bad ref",
        }])
        resolver = ConflictResolver(provider=FakeLLMProvider(response))
        existing = [_make_block("Info", "blk_1")]
        result = resolver.resolve("New", existing)
        assert len(result.actions) == 0  # Skipped invalid reference

    def test_llm_failure_falls_back_to_add(self):
        resolver = ConflictResolver(provider=FailingProvider())
        existing = [_make_block("Info", "blk_1")]
        result = resolver.resolve("New content", existing)
        assert len(result.actions) == 1
        assert result.actions[0].action == ConflictActionType.ADD
        assert "Fallback" in result.actions[0].reason
        assert len(result.errors) > 0

    def test_unparseable_response_falls_back_to_add(self):
        resolver = ConflictResolver(provider=FakeLLMProvider("not json at all"))
        existing = [_make_block("Info", "blk_1")]
        result = resolver.resolve("New", existing)
        assert len(result.actions) == 1
        assert result.actions[0].action == ConflictActionType.ADD
        assert len(result.errors) > 0

    def test_code_fenced_json_response(self):
        """LLM responses often wrap JSON in code fences."""
        inner = json.dumps([{"action": "ADD", "block_id": None, "new_content": "Test", "reason": "New"}])
        response = f"```json\n{inner}\n```"
        resolver = ConflictResolver(provider=FakeLLMProvider(response))
        existing = [_make_block("Old", "blk_1")]
        result = resolver.resolve("Test", existing)
        assert len(result.actions) == 1
        assert result.actions[0].action == ConflictActionType.ADD

    def test_dict_response_with_actions_key(self):
        """Some LLMs return {"actions": [...]} instead of a plain array."""
        response = json.dumps({
            "actions": [{"action": "NONE", "block_id": None, "new_content": None, "reason": "Known"}]
        })
        resolver = ConflictResolver(provider=FakeLLMProvider(response))
        existing = [_make_block("Known", "blk_1")]
        result = resolver.resolve("Known", existing)
        assert len(result.actions) == 1
        assert result.actions[0].action == ConflictActionType.NONE

    def test_custom_system_prompt(self):
        resolver = ConflictResolver(
            provider=FakeLLMProvider("[]"),
            system_prompt="Custom prompt",
        )
        assert resolver.system_prompt == "Custom prompt"

    def test_raw_response_stored(self):
        response = json.dumps([{"action": "ADD", "block_id": None, "new_content": "X", "reason": "New"}])
        resolver = ConflictResolver(provider=FakeLLMProvider(response))
        existing = [_make_block("Y", "blk_1")]
        result = resolver.resolve("X", existing)
        assert result.raw_response == response
