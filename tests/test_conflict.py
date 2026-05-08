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

    def test_cross_type_update_dropped_when_new_block_type_set(self):
        """v0.10.2 type-scoped guard: when the resolver picks a target
        block of a *different* type than the new write, drop the
        action. Prevents the FACT-content-overwrites-ENTITY bug."""
        # The LLM is told to UPDATE blk_entity (type=ENTITY) with
        # content from a new FACT write — should be dropped.
        response = json.dumps([{
            "action": "UPDATE",
            "block_id": "blk_entity_xyz",
            "new_content": "User likes that fund for tax-efficiency",
            "reason": "Looks similar enough",
        }])
        resolver = ConflictResolver(provider=FakeLLMProvider(response))
        # Existing block is an ENTITY (e.g., a fund anchor).
        entity_block = Block(
            id="blk_entity_xyz",
            content="Parag Parikh Flexi Cap",
            type=BlockType.ENTITY,
            tags=[],
            metadata=BlockMetadata(
                confidence=1.0, source=SourceType.EXPLICIT,
            ),
        )
        # We're storing a FACT — the resolver's UPDATE pointing at
        # the ENTITY should be dropped.
        result = resolver.resolve(
            "User likes that fund for tax-efficiency",
            [entity_block],
            new_block_type=BlockType.FACT,
        )
        assert len(result.actions) == 0, (
            f"cross-type UPDATE should be dropped; got {result.actions}"
        )

    def test_same_type_update_still_allowed_with_guard(self):
        """The type-scoped guard must NOT drop legitimate same-type
        UPDATE actions — that would break normal conflict resolution."""
        response = json.dumps([{
            "action": "UPDATE",
            "block_id": "blk_fact_old",
            "new_content": "User actually prefers active management",
            "reason": "Supersedes prior fact",
        }])
        resolver = ConflictResolver(provider=FakeLLMProvider(response))
        existing_fact = _make_block(
            "User prefers passive index funds", "blk_fact_old",
        )
        result = resolver.resolve(
            "User actually prefers active management",
            [existing_fact],
            new_block_type=BlockType.FACT,
        )
        assert len(result.actions) == 1
        assert result.actions[0].action == ConflictActionType.UPDATE
        assert result.actions[0].block_id == "blk_fact_old"

    def test_no_type_arg_keeps_old_behavior(self):
        """Backward-compat: when `new_block_type` isn't passed (legacy
        callers), the type guard is off and all valid block_ids
        pass through."""
        response = json.dumps([{
            "action": "UPDATE",
            "block_id": "blk_entity_zzz",
            "new_content": "Will be applied",
            "reason": "Caller didn't opt into the type guard",
        }])
        resolver = ConflictResolver(provider=FakeLLMProvider(response))
        entity_block = Block(
            id="blk_entity_zzz", content="Some entity",
            type=BlockType.ENTITY, tags=[],
            metadata=BlockMetadata(
                confidence=1.0, source=SourceType.EXPLICIT,
            ),
        )
        # No new_block_type → guard is off → action survives.
        result = resolver.resolve("New text", [entity_block])
        assert len(result.actions) == 1

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
