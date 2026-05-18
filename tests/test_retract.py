"""Tests for the RETRACT conflict action (G3, v0.12.0).

RETRACT is the user-initiated closure of a prior fact — "I sold my
Vanguard position", "we cancelled the SIP", etc. Distinct from
DELETE (which is for memories that were factually wrong).

These tests cover the pure-enum + edge-write side of the contract.
End-to-end LLM-driven resolver tests live in test_conflict.py and
require an LLM provider.
"""

import pytest

from memblock import MemBlock, BlockType
from memblock.conflict import ConflictActionType
from memblock.types import EdgeRelation


def test_retract_is_a_member_of_conflict_action_type():
    assert ConflictActionType.RETRACT.value == "RETRACT"
    assert ConflictActionType("RETRACT") is ConflictActionType.RETRACT


def test_retract_distinct_from_delete():
    assert ConflictActionType.RETRACT is not ConflictActionType.DELETE


def test_conflict_system_prompt_documents_retract():
    """The LLM resolver prompt must include RETRACT in its action
    enumeration — otherwise the LLM never emits it."""
    from memblock.conflict import CONFLICT_SYSTEM_PROMPT
    assert "RETRACT" in CONFLICT_SYSTEM_PROMPT


class TestRetractEdgeWriting:
    """When a retraction lands via store() with `metadata._retracts:
    [block_id, ...]`, the auto-link layer must materialise a
    CONTRADICTS edge from the new block to each retracted block."""

    @pytest.fixture
    def mem(self):
        return MemBlock(storage="sqlite:///:memory:")

    def test_retracts_marker_writes_contradicts_edge(self, mem):
        # Store the original "open position" fact
        original = mem.store(
            "Holds VTSAX 150 shares",
            type=BlockType.ENTITY,
        )
        # Store the retraction with the marker
        retraction = mem.store(
            "Sold VTSAX position",
            type=BlockType.EVENT,
            metadata={"_retracts": [original.id]},
            happened_at=None,  # avoid colliding via temporal hash
        )

        # The retraction should have a CONTRADICTS edge pointing at
        # the original block. `get_edges_between(src, tgt)` returns
        # all outgoing edges from src to tgt — we filter for the
        # CONTRADICTS relation.
        edges = mem._graph.get_edges_between(retraction.id, original.id)
        relations = [e.relation for e in edges]
        assert EdgeRelation.CONTRADICTS in relations

    def test_no_marker_no_contradicts_edge(self, mem):
        original = mem.store("foo", type=BlockType.FACT)
        unrelated = mem.store("bar", type=BlockType.FACT)
        # No metadata._retracts — no CONTRADICTS edge from unrelated → original.
        edges = mem._graph.get_edges_between(unrelated.id, original.id)
        relations = [e.relation for e in edges]
        assert EdgeRelation.CONTRADICTS not in relations
