"""Tests for AsyncMemBlock — async wrapper around MemBlock."""

import asyncio
import pytest

from memblock import AsyncMemBlock, BlockType, SourceType, EdgeRelation


class TestAsyncStore:
    def test_store_and_get(self):
        async def _test():
            async with AsyncMemBlock(storage="sqlite:///:memory:") as mem:
                block = await mem.store("Async test content", type=BlockType.FACT)
                assert block is not None
                assert block.content == "Async test content"

                retrieved = await mem.get(block.id)
                assert retrieved is not None
                assert retrieved.content == "Async test content"

        asyncio.run(_test())

    def test_store_with_params(self):
        async def _test():
            async with AsyncMemBlock(storage="sqlite:///:memory:") as mem:
                block = await mem.store(
                    "Async preference",
                    type=BlockType.PREFERENCE,
                    confidence=0.9,
                    source=SourceType.INFERRED,
                    tags=["async", "test"],
                    session_id="sess_1",
                    org_id="org_1",
                    project_id="proj_1",
                    agent_id="agent_1",
                    metadata={"env": "test"},
                )
                assert block.type == BlockType.PREFERENCE
                assert block.metadata.confidence == 0.9
                assert "async" in block.tags
                assert block.metadata.session_id == "sess_1"
                assert block.metadata.org_id == "org_1"

        asyncio.run(_test())

    def test_update(self):
        async def _test():
            async with AsyncMemBlock(storage="sqlite:///:memory:") as mem:
                block = await mem.store("Original", type=BlockType.FACT)
                updated = await mem.update(block.id, content="Updated")
                assert updated is not None
                assert updated.content == "Updated"

        asyncio.run(_test())

    def test_delete(self):
        async def _test():
            async with AsyncMemBlock(storage="sqlite:///:memory:") as mem:
                block = await mem.store("To delete", type=BlockType.FACT)
                result = await mem.delete(block.id)
                assert result is True
                retrieved = await mem.get(block.id)
                assert retrieved is None

        asyncio.run(_test())


class TestAsyncGraph:
    def test_link_and_neighbors(self):
        async def _test():
            async with AsyncMemBlock(storage="sqlite:///:memory:") as mem:
                b1 = await mem.store("Node A", type=BlockType.FACT)
                b2 = await mem.store("Node B", type=BlockType.FACT)
                await mem.link(b1.id, b2.id, relation=EdgeRelation.RELATED_TO)
                neighbors = await mem.neighbors(b1.id)
                assert len(neighbors) >= 1
                assert any(n.id == b2.id for n in neighbors)

        asyncio.run(_test())

    def test_unlink(self):
        async def _test():
            async with AsyncMemBlock(storage="sqlite:///:memory:") as mem:
                b1 = await mem.store("Node A", type=BlockType.FACT)
                b2 = await mem.store("Node B", type=BlockType.FACT)
                await mem.link(b1.id, b2.id)
                removed = await mem.unlink(b1.id, b2.id)
                assert removed >= 1

        asyncio.run(_test())

    def test_traverse(self):
        async def _test():
            async with AsyncMemBlock(storage="sqlite:///:memory:") as mem:
                b1 = await mem.store("Root", type=BlockType.FACT)
                b2 = await mem.store("Child", type=BlockType.FACT)
                b3 = await mem.store("Grandchild", type=BlockType.FACT)
                await mem.link(b1.id, b2.id)
                await mem.link(b2.id, b3.id)
                nodes = await mem.traverse(b1.id, max_depth=3)
                assert len(nodes) >= 2

        asyncio.run(_test())


class TestAsyncQuery:
    def test_query_by_type(self):
        async def _test():
            async with AsyncMemBlock(storage="sqlite:///:memory:") as mem:
                await mem.store("Fact 1", type=BlockType.FACT)
                await mem.store("Pref 1", type=BlockType.PREFERENCE)
                results = await mem.query(type=BlockType.FACT)
                assert all(b.type == BlockType.FACT for b in results)

        asyncio.run(_test())

    def test_query_by_tags(self):
        async def _test():
            async with AsyncMemBlock(storage="sqlite:///:memory:") as mem:
                await mem.store("Tagged", type=BlockType.FACT, tags=["important"])
                await mem.store("Not tagged", type=BlockType.FACT)
                results = await mem.query(tags=["important"])
                assert len(results) >= 1
                assert all("important" in b.tags for b in results)

        asyncio.run(_test())

    def test_query_text_search(self):
        async def _test():
            async with AsyncMemBlock(storage="sqlite:///:memory:") as mem:
                await mem.store("Python is great for AI", type=BlockType.FACT)
                await mem.store("JavaScript is for web", type=BlockType.FACT)
                results = await mem.query(text_search="Python AI")
                assert len(results) >= 1

        asyncio.run(_test())


class TestAsyncContext:
    def test_build_context(self):
        async def _test():
            async with AsyncMemBlock(storage="sqlite:///:memory:") as mem:
                await mem.store("User prefers dark mode", type=BlockType.PREFERENCE)
                await mem.store("User works at Acme Corp", type=BlockType.FACT)
                ctx = await mem.build_context(query="user preferences", token_budget=2000)
                assert isinstance(ctx, str)
                assert len(ctx) > 0

        asyncio.run(_test())


class TestAsyncSession:
    def test_sessions(self):
        async def _test():
            async with AsyncMemBlock(storage="sqlite:///:memory:") as mem:
                await mem.store("Msg 1", type=BlockType.EVENT, session_id="chat_1")
                await mem.store("Msg 2", type=BlockType.EVENT, session_id="chat_2")
                sessions = await mem.get_sessions()
                assert "chat_1" in sessions
                assert "chat_2" in sessions

        asyncio.run(_test())

    def test_session_history(self):
        async def _test():
            async with AsyncMemBlock(storage="sqlite:///:memory:") as mem:
                await mem.store("First", type=BlockType.EVENT, session_id="s1")
                await mem.store("Second", type=BlockType.EVENT, session_id="s1")
                history = await mem.get_session_history("s1")
                assert len(history) == 2

        asyncio.run(_test())


class TestAsyncMisc:
    def test_verify(self):
        async def _test():
            async with AsyncMemBlock(storage="sqlite:///:memory:") as mem:
                report = await mem.verify()
                assert report.valid is True

        asyncio.run(_test())

    def test_stats(self):
        async def _test():
            async with AsyncMemBlock(storage="sqlite:///:memory:") as mem:
                await mem.store("Test", type=BlockType.FACT)
                stats = await mem.stats()
                assert stats["total_blocks"] == 1

        asyncio.run(_test())

    def test_export_markdown(self):
        async def _test():
            async with AsyncMemBlock(storage="sqlite:///:memory:") as mem:
                await mem.store("Export test", type=BlockType.FACT)
                md = await mem.export_markdown()
                assert "Export test" in md

        asyncio.run(_test())

    def test_context_manager(self):
        async def _test():
            async with AsyncMemBlock(storage="sqlite:///:memory:") as mem:
                block = await mem.store("CM test", type=BlockType.FACT)
                assert block is not None

        asyncio.run(_test())

    def test_sync_property(self):
        from memblock import MemBlock
        mem = AsyncMemBlock(storage="sqlite:///:memory:")
        assert isinstance(mem.sync, MemBlock)
        mem.sync.close()

    def test_has_embeddings(self):
        mem = AsyncMemBlock(storage="sqlite:///:memory:")
        assert mem.has_embeddings is False
        mem.sync.close()

    def test_repr(self):
        mem = AsyncMemBlock(storage="sqlite:///:memory:")
        r = repr(mem)
        assert "Async" in r
        mem.sync.close()

    def test_prune(self):
        async def _test():
            async with AsyncMemBlock(storage="sqlite:///:memory:") as mem:
                pruned = await mem.prune(min_strength=0.1)
                assert isinstance(pruned, list)

        asyncio.run(_test())

    def test_strongest_weakest(self):
        async def _test():
            async with AsyncMemBlock(storage="sqlite:///:memory:") as mem:
                await mem.store("Strong", type=BlockType.FACT, confidence=1.0)
                strongest = await mem.strongest(limit=5)
                weakest = await mem.weakest(limit=5)
                assert isinstance(strongest, list)
                assert isinstance(weakest, list)

        asyncio.run(_test())

    def test_on_hook(self):
        async def _test():
            async with AsyncMemBlock(storage="sqlite:///:memory:") as mem:
                events = []
                mem.on("on_add", lambda d: events.append(d))
                await mem.store("Hook test", type=BlockType.FACT)
                assert len(events) == 1

        asyncio.run(_test())
