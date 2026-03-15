"""Tests for hierarchical scoping — org → project → user → agent → session."""

import pytest

from memblock import MemBlock, BlockType


class TestHierarchicalScoping:
    """Test the 5-level hierarchical scoping system."""

    @pytest.fixture
    def mem(self):
        m = MemBlock(storage="sqlite:///:memory:")
        yield m
        m.close()

    def test_store_with_org_id(self, mem):
        block = mem.store("Org fact", type=BlockType.FACT, org_id="org_acme")
        assert block.metadata.org_id == "org_acme"

    def test_store_with_project_id(self, mem):
        block = mem.store("Project fact", type=BlockType.FACT, project_id="proj_alpha")
        assert block.metadata.project_id == "proj_alpha"

    def test_store_with_agent_id(self, mem):
        block = mem.store("Agent fact", type=BlockType.FACT, agent_id="agent_bot1")
        assert block.metadata.agent_id == "agent_bot1"

    def test_store_with_session_id(self, mem):
        block = mem.store("Session fact", type=BlockType.FACT, session_id="sess_001")
        assert block.metadata.session_id == "sess_001"

    def test_store_with_all_levels(self, mem):
        block = mem.store(
            "Full hierarchy",
            type=BlockType.FACT,
            org_id="org_1",
            project_id="proj_1",
            agent_id="agent_1",
            session_id="sess_1",
        )
        assert block.metadata.org_id == "org_1"
        assert block.metadata.project_id == "proj_1"
        assert block.metadata.agent_id == "agent_1"
        assert block.metadata.session_id == "sess_1"

    def test_query_scoped_by_org_id(self, mem):
        mem.store("Acme fact", type=BlockType.FACT, org_id="acme")
        mem.store("Beta fact", type=BlockType.FACT, org_id="beta")
        mem.store("No org", type=BlockType.FACT)

        results = mem.query(org_id="acme")
        assert len(results) == 1
        assert results[0].content == "Acme fact"

    def test_query_scoped_by_project_id(self, mem):
        mem.store("Alpha task", type=BlockType.FACT, project_id="alpha")
        mem.store("Beta task", type=BlockType.FACT, project_id="beta")

        results = mem.query(project_id="alpha")
        assert len(results) == 1
        assert results[0].content == "Alpha task"

    def test_query_scoped_by_agent_id(self, mem):
        mem.store("Bot1 memory", type=BlockType.FACT, agent_id="bot1")
        mem.store("Bot2 memory", type=BlockType.FACT, agent_id="bot2")

        results = mem.query(agent_id="bot1")
        assert len(results) == 1
        assert results[0].content == "Bot1 memory"

    def test_query_scoped_by_session(self, mem):
        mem.store("Chat 1 msg", type=BlockType.FACT, session_id="chat_1")
        mem.store("Chat 2 msg", type=BlockType.FACT, session_id="chat_2")

        results = mem.query(session_id="chat_1")
        assert len(results) == 1
        assert results[0].content == "Chat 1 msg"

    def test_combined_scoping(self, mem):
        """Multiple scope levels filter together (AND logic)."""
        mem.store("Match", type=BlockType.FACT, org_id="acme", project_id="alpha")
        mem.store("Wrong project", type=BlockType.FACT, org_id="acme", project_id="beta")
        mem.store("Wrong org", type=BlockType.FACT, org_id="other", project_id="alpha")

        results = mem.query(org_id="acme", project_id="alpha")
        assert len(results) == 1
        assert results[0].content == "Match"

    def test_query_without_scope_returns_all(self, mem):
        """Queries without scope filters return all blocks."""
        mem.store("Org block", type=BlockType.FACT, org_id="acme")
        mem.store("Plain block", type=BlockType.FACT)
        mem.store("Session block", type=BlockType.FACT, session_id="s1")

        results = mem.query()
        assert len(results) == 3

    def test_default_scoping_from_constructor(self):
        """Default scope set at constructor level applies to all operations."""
        mem = MemBlock(
            storage="sqlite:///:memory:",
            org_id="default_org",
            project_id="default_proj",
        )
        mem.store("Auto-scoped block", type=BlockType.FACT)
        block = mem.query()[0]
        assert block.metadata.org_id == "default_org"
        assert block.metadata.project_id == "default_proj"
        mem.close()

    def test_per_call_override_of_defaults(self):
        """Per-call scope overrides constructor defaults."""
        mem = MemBlock(
            storage="sqlite:///:memory:",
            org_id="default_org",
        )
        mem.store("Default org", type=BlockType.FACT)
        mem.store("Override org", type=BlockType.FACT, org_id="other_org")

        default_results = mem.query(org_id="default_org")
        override_results = mem.query(org_id="other_org")
        assert len(default_results) == 1
        assert len(override_results) == 1
        mem.close()

    def test_build_context_scoped(self, mem):
        """build_context respects scope filters."""
        mem.store("Acme knowledge", type=BlockType.FACT, org_id="acme")
        mem.store("Beta knowledge", type=BlockType.FACT, org_id="beta")

        ctx = mem.build_context(query="knowledge", org_id="acme")
        assert "Acme" in ctx
        assert "Beta" not in ctx

    def test_full_hierarchy_query(self, mem):
        """All 5 levels combined."""
        mem.store(
            "Exact match",
            type=BlockType.FACT,
            org_id="org",
            project_id="proj",
            agent_id="agent",
            session_id="sess",
        )
        mem.store(
            "Partial match",
            type=BlockType.FACT,
            org_id="org",
            project_id="proj",
        )

        results = mem.query(
            org_id="org",
            project_id="proj",
            agent_id="agent",
            session_id="sess",
        )
        assert len(results) == 1
        assert results[0].content == "Exact match"
