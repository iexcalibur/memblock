"""End-to-end tests for the MemBlock SDK public API."""

import pytest

from memblock import (
    MemBlock,
    BlockType,
    EdgeRelation,
    EncryptionLevel,
    SourceType,
)


@pytest.fixture
def mem():
    m = MemBlock(storage="sqlite:///:memory:")
    yield m
    m.close()


@pytest.fixture
def mem_encrypted():
    m = MemBlock(storage="sqlite:///:memory:", encryption_key="test-secret-key")
    yield m
    m.close()


class TestMemBlockBasicFlow:
    def test_store_and_get(self, mem: MemBlock):
        block = mem.store(content="User likes Python", type=BlockType.PREFERENCE)
        retrieved = mem.get(block.id)

        assert retrieved is not None
        assert retrieved.content == "User likes Python"
        assert retrieved.type == BlockType.PREFERENCE

    def test_store_with_all_options(self, mem: MemBlock):
        block = mem.store(
            content="Deployed v2.0 on March 10",
            type=BlockType.EVENT,
            confidence=0.95,
            source=SourceType.OBSERVED,
            tags=["deployment", "v2"],
            decay_rate=0.05,
            ttl=86400,
        )
        assert block.metadata.confidence == 0.95
        assert block.tags == ["deployment", "v2"]

    def test_update_block(self, mem: MemBlock):
        block = mem.store(content="Original")
        updated = mem.update(block.id, content="Modified", confidence=0.8)
        assert updated.content == "Modified"
        assert updated.metadata.confidence == 0.8

    def test_delete_block(self, mem: MemBlock):
        block = mem.store(content="Delete me")
        assert mem.delete(block.id) is True
        assert mem.get(block.id) is None


class TestMemBlockGraph:
    def test_link_and_neighbors(self, mem: MemBlock):
        python = mem.store(content="Python", type=BlockType.ENTITY)
        pref = mem.store(content="User likes Python", type=BlockType.PREFERENCE)

        mem.link(pref.id, python.id, relation=EdgeRelation.ABOUT)

        neighbors = mem.neighbors(python.id)
        assert len(neighbors) == 1
        assert neighbors[0].id == pref.id

    def test_unlink(self, mem: MemBlock):
        a = mem.store(content="A", type=BlockType.FACT)
        b = mem.store(content="B", type=BlockType.FACT)

        mem.link(a.id, b.id, relation=EdgeRelation.SUPPORTS)
        assert mem.unlink(a.id, b.id) == 1
        assert len(mem.neighbors(a.id)) == 0

    def test_link_with_string_relation(self, mem: MemBlock):
        a = mem.store(content="A", type=BlockType.FACT)
        b = mem.store(content="B", type=BlockType.FACT)

        mem.link(a.id, b.id, relation="supports")
        neighbors = mem.neighbors(a.id)
        assert len(neighbors) == 1

    def test_traverse(self, mem: MemBlock):
        a = mem.store(content="A", type=BlockType.FACT)
        b = mem.store(content="B", type=BlockType.FACT)
        c = mem.store(content="C", type=BlockType.FACT)

        mem.link(a.id, b.id, relation=EdgeRelation.RELATED_TO)
        mem.link(b.id, c.id, relation=EdgeRelation.RELATED_TO)

        traversed = mem.traverse(a.id, max_depth=5)
        assert len(traversed) == 2


class TestMemBlockQuery:
    def test_query_by_type(self, mem: MemBlock):
        mem.store(content="Fact 1", type=BlockType.FACT)
        mem.store(content="Fact 2", type=BlockType.FACT)
        mem.store(content="Pref 1", type=BlockType.PREFERENCE)

        results = mem.query(type=BlockType.FACT)
        assert len(results) == 2

    def test_query_by_text(self, mem: MemBlock):
        mem.store(content="User likes Python for backend")
        mem.store(content="User deploys to AWS")

        results = mem.query(text_search="Python")
        assert len(results) == 1

    def test_query_by_confidence(self, mem: MemBlock):
        mem.store(content="High", confidence=0.95)
        mem.store(content="Low", confidence=0.2)

        results = mem.query(min_confidence=0.5)
        assert len(results) == 1


class TestMemBlockContext:
    def test_build_context(self, mem: MemBlock):
        mem.store(content="User likes TypeScript", type=BlockType.PREFERENCE)
        mem.store(content="User uses AWS", type=BlockType.FACT)

        context = mem.build_context(token_budget=4000)
        assert "## Memory Context" in context
        assert "TypeScript" in context

    def test_build_context_respects_budget(self, mem: MemBlock):
        for i in range(50):
            mem.store(content=f"Memory block {i} with extra content for tokens")

        context = mem.build_context(token_budget=200)
        # Rough check: 200 tokens * 4 chars = 800 chars max
        assert len(context) < 1000

    def test_build_context_graph_walk(self, mem: MemBlock):
        entity = mem.store(content="React framework", type=BlockType.ENTITY)
        fact = mem.store(content="React uses virtual DOM", type=BlockType.FACT)
        mem.link(fact.id, entity.id, relation=EdgeRelation.ABOUT)

        context = mem.build_context(query="React", strategy="graph_walk")
        assert "React" in context

    def test_build_context_type_grouped(self, mem: MemBlock):
        mem.store(content="A fact", type=BlockType.FACT)
        mem.store(content="A preference", type=BlockType.PREFERENCE)

        context = mem.build_context(strategy="type_grouped")
        assert "## Memory Context" in context


class TestMemBlockEncryption:
    def test_store_encrypted(self, mem_encrypted: MemBlock):
        block = mem_encrypted.store(
            content="My secret API key",
            encryption_level=EncryptionLevel.STANDARD,
        )
        assert block.encrypted is True

        # Retrieved content should be decrypted
        retrieved = mem_encrypted.get(block.id, decrypt=True)
        assert retrieved.content == "My secret API key"

    def test_store_sensitive(self, mem_encrypted: MemBlock):
        block = mem_encrypted.store(
            content="SSN: 123-45-6789",
            encryption_level=EncryptionLevel.SENSITIVE,
        )
        retrieved = mem_encrypted.get(block.id, decrypt=True)
        assert retrieved.content == "SSN: 123-45-6789"

    def test_no_encryption_when_level_none(self, mem_encrypted: MemBlock):
        block = mem_encrypted.store(
            content="Public info",
            encryption_level=EncryptionLevel.NONE,
        )
        assert block.encrypted is False


class TestMemBlockIntegrity:
    def test_verify_clean(self, mem: MemBlock):
        mem.store(content="Block 1")
        mem.store(content="Block 2")

        report = mem.verify()
        assert report.valid is True
        assert report.total_ops >= 2

    def test_verify_detects_tampering(self, mem: MemBlock):
        mem.store(content="Block 1")
        mem.store(content="Block 2")

        # Tamper directly with the database
        mem._storage.conn.execute(
            "UPDATE operations SET hash = 'sha256:tampered' WHERE clock = 0"
        )
        mem._storage.conn.commit()

        report = mem.verify()
        assert report.valid is False


class TestMemBlockDecay:
    def test_strongest_and_weakest(self, mem: MemBlock):
        for i in range(5):
            mem.store(content=f"Block {i}", confidence=(i + 1) * 0.2)

        strongest = mem.strongest(limit=2)
        assert len(strongest) == 2
        assert strongest[0][1] >= strongest[1][1]  # sorted by strength

        weakest = mem.weakest(limit=2)
        assert len(weakest) == 2
        assert weakest[0][1] <= weakest[1][1]


class TestMemBlockExport:
    def test_export_markdown(self, mem: MemBlock):
        mem.store(content="User likes Python", type=BlockType.PREFERENCE, tags=["lang"])
        mem.store(content="Uses AWS", type=BlockType.FACT)

        md = mem.export_markdown()
        assert "# MemBlock Export" in md
        assert "PREFERENCE" in md
        assert "FACT" in md
        assert "Python" in md


class TestMemBlockStats:
    def test_stats(self, mem: MemBlock):
        mem.store(content="Fact 1", type=BlockType.FACT)
        mem.store(content="Fact 2", type=BlockType.FACT)
        mem.store(content="Pref 1", type=BlockType.PREFERENCE)

        stats = mem.stats()
        assert stats["total_blocks"] == 3
        assert stats["blocks_by_type"]["fact"] == 2
        assert stats["blocks_by_type"]["preference"] == 1


class TestMemBlockContextManager:
    def test_with_statement(self):
        with MemBlock(storage="sqlite:///:memory:") as mem:
            block = mem.store(content="Test")
            assert block.id.startswith("blk_")


class TestMemBlockRepr:
    def test_repr(self, mem: MemBlock):
        mem.store(content="Test")
        r = repr(mem)
        assert "MemBlock(" in r
        assert "blocks=1" in r


class TestFullWorkflow:
    """Simulates a real-world usage scenario."""

    def test_developer_memory_workflow(self, mem: MemBlock):
        # 1. Store various memories about a user
        python = mem.store(content="Python", type=BlockType.ENTITY, tags=["language"])
        typescript = mem.store(content="TypeScript", type=BlockType.ENTITY, tags=["language"])
        aws = mem.store(content="AWS", type=BlockType.ENTITY, tags=["cloud"])

        pref1 = mem.store(
            content="User prefers Python for backend",
            type=BlockType.PREFERENCE,
            confidence=0.9,
            tags=["backend"],
        )
        pref2 = mem.store(
            content="User uses TypeScript for frontend",
            type=BlockType.PREFERENCE,
            confidence=0.85,
            tags=["frontend"],
        )
        fact1 = mem.store(
            content="User deployed app to AWS on March 10",
            type=BlockType.EVENT,
            confidence=1.0,
            tags=["deployment"],
        )

        # 2. Create relationships
        mem.link(pref1.id, python.id, relation=EdgeRelation.ABOUT)
        mem.link(pref2.id, typescript.id, relation=EdgeRelation.ABOUT)
        mem.link(fact1.id, aws.id, relation=EdgeRelation.ABOUT)

        # 3. Query memories
        preferences = mem.query(type=BlockType.PREFERENCE)
        assert len(preferences) == 2

        python_related = mem.query(related_to=python.id)
        assert any("Python" in b.content for b in python_related)

        # 4. Build context for LLM
        context = mem.build_context(
            query="What stack does the user prefer?",
            token_budget=2000,
            strategy="relevance",
        )
        assert len(context) > 0
        assert "## Memory Context" in context

        # 5. Verify integrity
        report = mem.verify()
        assert report.valid is True

        # 6. Check stats
        stats = mem.stats()
        assert stats["total_blocks"] == 6
        assert stats["total_edges"] == 3

        # 7. Export
        md = mem.export_markdown()
        assert "Python" in md
        assert "TypeScript" in md
