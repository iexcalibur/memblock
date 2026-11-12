"""MemorySession: read memory, call the model, commit the turn, drop it from the prompt."""

from __future__ import annotations

import pytest

from memblock import (
    BlockType,
    EdgeRelation,
    MemBlock,
    MemorySession,
    MemorySessionError,
    ModelReply,
)
from memblock.session import (
    MEMORY_HEADER,
    content_words,
    normalize_reply,
    render_message,
    shares_content_word,
    split_document,
)
from memblock.transactions import is_internal, is_staged


class Recorder:
    """A model that records every prompt and answers from a script (or a fixed reply)."""

    def __init__(self, replies=None, default="Noted."):
        self.replies = list(replies or [])
        self.default = default
        self.prompts = []

    def __call__(self, messages, tools=None):
        self.prompts.append(messages)
        return self.replies.pop(0) if self.replies else self.default

    @property
    def last(self):
        return self.prompts[-1]

    def system(self, index=-1):
        first = self.prompts[index][0]
        return first["content"] if first["role"] == "system" else ""


def tool_call(call_id="c1", name="search", arguments=None):
    return {"content": "", "tool_calls": [{"id": call_id, "name": name, "arguments": arguments or {"q": "x"}}]}


@pytest.fixture
def db(tmp_path):
    return f"sqlite:///{tmp_path / 'mem.db'}"


@pytest.fixture
def mem(db):
    m = MemBlock(storage=db)
    yield m
    m.close()


def episodes(mem, session_id=None):
    blocks = mem._storage.query_blocks({"type": BlockType.EPISODE})
    if session_id:
        blocks = [b for b in blocks if b.metadata.session_id == session_id]
    blocks.sort(key=lambda b: (b.metadata.custom_metadata["_turn"] or 0, b.metadata.custom_metadata["_seq"]))
    return blocks


# ─── The core loop ───────────────────────────────────────────────────────────


class TestCommitThenClear:
    def test_turn_is_saved_word_for_word_and_leaves_the_prompt(self, mem):
        model = Recorder(default="Lovely name.")
        session = MemorySession(mem, model, recent_turns=0)
        said = "I have a three-month-old kitten named Memmy   who loves small shrimp.\nShe is grey."
        assert session.chat(said, enable_memory=True) == "Lovely name."

        stored = episodes(mem)
        assert [b.content for b in stored] == [said, "Lovely name."]  # exact text, spacing included
        assert [b.metadata.custom_metadata["_role"] for b in stored] == ["user", "assistant"]
        assert all(b.metadata.session_id == session.id for b in stored)
        assert all(b.metadata.decay_rate == 0.0 for b in stored)
        assert session.status()["uncommitted_turns"] == 0 and session.messages == []

        session.chat("What should I feed a kitten?")
        assert [m["role"] for m in model.last] == ["system", "user"]  # the old turn is not in the prompt

    def test_earlier_fact_comes_back_from_memory(self, mem):
        model = Recorder()
        session = MemorySession(mem, model, recent_turns=1)
        session.chat("My kitten Memmy is three months old and loves shrimp.", enable_memory=True)
        for i in range(8):
            session.chat(f"The train to platform {i} was late today.", enable_memory=True)
        session.chat("How old is my kitten Memmy?")
        system = model.system()
        assert system.startswith(MEMORY_HEADER)
        assert "My kitten Memmy is three months old and loves shrimp." in system
        assert session.last_prompt["recalled"]

    def test_a_turn_still_in_the_prompt_is_not_repeated_as_memory(self, mem):
        model = Recorder()
        session = MemorySession(mem, model, recent_turns=2)
        session.chat("My kitten Memmy loves shrimp.", enable_memory=True)
        session.chat("Tell me about my kitten Memmy.")
        prompt_text = "\n".join(m["content"] for m in model.last)
        assert prompt_text.count("My kitten Memmy loves shrimp.") == 1
        assert [m["role"] for m in model.last] == ["user", "assistant", "user"]

    def test_prompt_stays_flat_over_a_long_conversation(self, mem):
        model = Recorder(default="Understood, I have noted that down for you.")
        session = MemorySession(mem, model)
        sizes, transcript = [], 0
        for i in range(80):
            text = f"Entry {i}: the meeting about project {i % 7} moved to room {i} on floor {i % 5}."
            session.chat(text, enable_memory=True)
            sizes.append(session.last_prompt["tokens"])
            transcript += len(text) + len(model.default)
        full_transcript_tokens = transcript // 4
        assert max(sizes[40:]) < 700  # bounded by recall_budget + the short tail
        assert max(sizes[40:]) < full_transcript_tokens / 3
        assert max(sizes[60:]) <= max(sizes[20:40]) * 1.5  # no growth trend
        assert len(episodes(mem)) == 160

    def test_without_enable_memory_it_is_an_ordinary_chat(self, mem):
        model = Recorder()
        session = MemorySession(mem, model)
        session.chat("first")
        session.chat("second")
        assert [m["content"] for m in model.last] == ["first", "Noted.", "second"]
        assert episodes(mem) == [] and session.status()["uncommitted_turns"] == 2
        assert session.commit_history() == 2
        assert len(episodes(mem)) == 4

    def test_reading_and_saving_are_separate_switches(self, mem):
        model = Recorder()
        session = MemorySession(mem, model, recent_turns=0)
        session.chat("My kitten Memmy loves shrimp.", enable_memory=True)
        session.set_memory_read(False)
        session.chat("What does my kitten Memmy love?", enable_memory=True)  # still saved
        assert [m["role"] for m in model.last] == ["user"]  # nothing read
        assert len(episodes(mem)) == 4
        session.set_memory_read(True)
        session.chat("What does my kitten Memmy love?")
        assert "shrimp" in model.system()

    def test_system_prompt_is_kept_and_memory_added_after_it(self, mem):
        model = Recorder()
        session = MemorySession(mem, model, system_prompt="You are a vet's assistant.", recent_turns=0)
        session.chat("My kitten Memmy loves shrimp.", enable_memory=True)
        session.chat("What does my kitten Memmy love?")
        system = model.system()
        assert system.startswith("You are a vet's assistant.\n\n" + MEMORY_HEADER)

    def test_resuming_a_session_continues_turn_numbers(self, mem):
        first = MemorySession(mem, Recorder(), session_id="trip")
        first.chat("one", enable_memory=True)
        first.chat("two", enable_memory=True)
        later = MemorySession(mem, Recorder(), session_id="trip")
        assert later.status()["next_turn"] == 3
        later.chat("three", enable_memory=True)
        assert [b.metadata.custom_metadata["_turn"] for b in episodes(mem, "trip")] == [1, 1, 2, 2, 3, 3]

    def test_scope(self, mem):
        MemorySession(mem, Recorder(), session_id="a").chat("My kitten Memmy loves shrimp.", enable_memory=True)
        own = Recorder()
        MemorySession(mem, own, session_id="b", scope="session").chat("What does my kitten Memmy love?")
        assert own.system() == ""
        everyone = Recorder()
        MemorySession(mem, everyone, session_id="c", scope="user").chat("What does my kitten Memmy love?")
        assert "shrimp" in everyone.system()

    def test_other_block_types_are_recalled_too(self, mem):
        mem.store("User is allergic to peanuts", type=BlockType.FACT)
        model = Recorder()
        MemorySession(mem, model).chat("Is the user allergic to peanuts?")
        assert "- (fact) User is allergic to peanuts" in model.system()


class TestRelevanceFloor:
    def test_helpers(self):
        assert content_words("What is the name of my kitten?") == {"name", "kitten"}
        assert shares_content_word({"running"}, "she likes to run")
        assert shares_content_word({"kitten"}, "two kittens")
        assert not shares_content_word({"kitten"}, "the trains were late and it was cold")

    def test_filler_only_matches_are_left_out(self, mem):
        seed = MemorySession(mem, Recorder(), recent_turns=0)
        seed.chat("The trains were late and it was cold.", enable_memory=True)
        seed.chat("My kitten Memmy loves shrimp.", enable_memory=True)

        strict = Recorder()
        MemorySession(mem, strict, recent_turns=0).chat("What is the name of my kitten and what does it love?")
        assert "Memmy" in strict.system() and "trains" not in strict.system()

        loose = Recorder()
        MemorySession(mem, loose, recent_turns=0, relevance_floor=False).chat(
            "What is the name of my kitten and what does it love?"
        )
        assert "trains" in loose.system()  # matched on "and"/"the"/"it" alone


# ─── Write boundaries ────────────────────────────────────────────────────────


class TestWriteBoundaries:
    def test_tool_chain_is_one_turn_saved_after_the_final_answer(self, mem):
        model = Recorder([tool_call("c1"), tool_call("c2", arguments={"q": "y"}), "Found it: 42."])
        session = MemorySession(mem, model, tools=[{"type": "function", "function": {"name": "search"}}])

        first = session.chat("Look up the answer", enable_memory=True)
        assert first["requires_action"] and not first["turn_completed"]
        assert first["tool_calls"][0]["name"] == "search"
        assert episodes(mem) == []
        with pytest.raises(ValueError, match="pending tool results"):
            session.chat("another question")

        second = session.submit_tool_results([{"tool_call_id": "c1", "content": "nothing yet"}])
        assert second["requires_action"] and episodes(mem) == []

        final = session.submit_tool_results([{"tool_call_id": "c2", "content": "the answer is 42"}])
        assert final["answer"] == "Found it: 42." and final["memory_updated"]
        assert final["committed_turns"] == 1

        stored = episodes(mem)
        assert [b.metadata.custom_metadata["_role"] for b in stored] == [
            "user", "assistant", "tool", "assistant", "tool", "assistant",
        ]
        assert {b.metadata.custom_metadata["_turn"] for b in stored} == {1}
        assert stored[1].content == '[tool call c1] search({"q": "x"})'
        assert stored[2].content == "nothing yet"
        # every round reached the model as one growing turn
        assert [m["role"] for m in model.prompts[2]] == ["user", "assistant", "tool", "assistant", "tool"]

    def test_tool_result_validation(self, mem):
        model = Recorder([tool_call("c1")])
        session = MemorySession(mem, model, tools=[{"type": "function", "function": {"name": "search"}}])
        with pytest.raises(ValueError, match="No tool calls"):
            session.submit_tool_results([])
        session.chat("go")
        with pytest.raises(ValueError, match="exactly one result"):
            session.submit_tool_results([])
        with pytest.raises(ValueError, match="exactly one result"):
            session.submit_tool_results([{"tool_call_id": "other", "content": "x"}])
        with pytest.raises(ValueError, match="tool_call_id"):
            session.submit_tool_results([{"content": "x"}])
        with pytest.raises(ValueError, match="pending tool results"):
            session.save_state("unused.json")

    def test_cut_off_reply_is_not_saved_until_a_turn_completes(self, mem):
        model = Recorder([
            {"content": "This reply was cut", "finish_reason": "max_tokens"},
            {"content": "Complete now.", "finish_reason": "end_turn"},
        ])
        session = MemorySession(mem, model)
        cut = session.chat("Explain everything", enable_memory=True, return_dict=True)
        assert cut["finish_reason"] == "length" and not cut["memory_updated"]
        assert episodes(mem) == [] and session.status()["uncommitted_turns"] == 1
        # it is still in the prompt for the next call
        done = session.chat("Please continue", enable_memory=True, return_dict=True)
        assert [m["content"] for m in model.last] == ["Explain everything", "This reply was cut", "Please continue"]
        assert done["committed_turns"] == 2
        assert [b.content for b in episodes(mem)] == [
            "Explain everything", "This reply was cut", "Please continue", "Complete now.",
        ]

    def test_commit_history_saves_a_cut_off_turn(self, mem):
        session = MemorySession(mem, Recorder([{"content": "partial", "finish_reason": "length"}]))
        session.chat("go", enable_memory=True)
        assert session.commit_history() == 1
        assert [b.content for b in episodes(mem)] == ["go", "partial"]
        assert session.commit_history() == 0


# ─── Failures ────────────────────────────────────────────────────────────────


class TestFailures:
    def test_model_failure_saves_nothing_and_the_session_carries_on(self, mem):
        def broken(messages):
            raise ConnectionError("model is down")

        session = MemorySession(mem, broken)
        with pytest.raises(ConnectionError):
            session.chat("hello", enable_memory=True)
        assert episodes(mem) == [] and session.status()["uncommitted_turns"] == 0
        assert not session.uncertain and session.status()["next_turn"] == 1

    def test_failed_commit_saves_nothing_and_can_be_retried(self, mem, monkeypatch):
        session = MemorySession(mem, Recorder(default="reply"))
        real = mem._store.create
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("disk full")
            return real(*args, **kwargs)

        monkeypatch.setattr(mem._store, "create", flaky)
        with pytest.raises(MemorySessionError) as failure:
            session.chat("remember this", enable_memory=True)
        assert failure.value.stage == "commit" and failure.value.answer == "reply"
        assert not failure.value.uncertain and not session.uncertain  # receipts show nothing was saved
        assert episodes(mem) == []
        leftovers = [b for b in mem._storage.get_all_blocks(include_deleted=True) if is_internal(b) or is_staged(b)]
        assert leftovers == []
        assert session.status()["uncommitted_turns"] == 1  # the turn is still here

        assert session.commit_history() == 1
        assert [b.content for b in episodes(mem)] == ["remember this", "reply"]
        assert mem.verify().valid

    def test_commit_left_in_doubt_blocks_writes_until_recover(self, mem):
        """The commit reached its commit point but the process failed before
        finishing: the session cannot tell, so it stops and asks for recover()."""
        session = MemorySession(mem, Recorder(default="reply"))

        def fail_once(point):
            if point == "apply:0":
                mem._txn_fault_hook = None
                raise OSError("connection lost")

        mem._txn_fault_hook = fail_once
        with pytest.raises(MemorySessionError) as failure:
            session.chat("remember this", enable_memory=True)
        assert failure.value.uncertain and session.uncertain
        with pytest.raises(MemorySessionError, match="recover"):
            session.chat("next")
        with pytest.raises(MemorySessionError, match="recover"):
            session.commit_history()

        assert session.recover() == "applied"  # recovery completed the published commit
        assert not session.uncertain and session.status()["uncommitted_turns"] == 0
        assert [b.content for b in episodes(mem)] == ["remember this", "reply"]
        session.chat("and we carry on", enable_memory=True)
        assert len(episodes(mem)) == 4  # nothing stored twice

    def test_closed_session(self, mem):
        with MemorySession(mem, Recorder()) as session:
            session.chat("hi")
        with pytest.raises(MemorySessionError, match="closed"):
            session.chat("again")
        assert mem.query(limit=1) == []  # the MemBlock itself is still open

    def test_argument_checks(self, mem):
        session = MemorySession(mem, Recorder())
        with pytest.raises(ValueError):
            session.chat("   ")
        with pytest.raises(ValueError):
            session.chat("hi", enable_memory="yes")
        with pytest.raises(ValueError):
            MemorySession(mem, Recorder(), scope="team")
        with pytest.raises(ValueError):
            MemorySession(mem, Recorder(), experience="always")
        with pytest.raises(TypeError):
            MemorySession(mem, model=42)


# ─── Models ──────────────────────────────────────────────────────────────────


class TestModels:
    def test_reply_shapes(self):
        assert normalize_reply("hi").text == "hi"
        assert normalize_reply(ModelReply(text="hi")).finish_reason == "stop"
        assert normalize_reply({"content": "hi", "stop_reason": "max_tokens"}).finish_reason == "length"
        nested = normalize_reply({
            "content": None,
            "tool_calls": [{"id": "a", "function": {"name": "search", "arguments": '{"q": 1}'}}],
        })
        assert nested.finish_reason == "tool_calls"
        assert nested.tool_calls == [{"id": "a", "name": "search", "arguments": '{"q": 1}'}]
        with pytest.raises(TypeError):
            normalize_reply(42)

    def test_provider_with_complete(self, mem):
        class Provider:
            def __init__(self):
                self.seen = []

            def complete(self, system_prompt, user_prompt):
                self.seen.append((system_prompt, user_prompt))
                return "from provider"

        provider = Provider()
        session = MemorySession(mem, provider, system_prompt="Be brief.", recent_turns=0)
        assert session.chat("My kitten Memmy loves shrimp.", enable_memory=True) == "from provider"
        assert provider.seen[0] == ("Be brief.", "My kitten Memmy loves shrimp.")
        session.chat("What does my kitten Memmy love?")
        system, user = provider.seen[1]
        assert "shrimp" in system and user == "What does my kitten Memmy love?"

    def test_provider_flattens_a_multi_turn_prompt(self, mem):
        class Provider:
            def complete(self, system_prompt, user_prompt):
                self.user = user_prompt
                return "ok"

        provider = Provider()
        session = MemorySession(mem, provider)
        session.chat("first")
        session.chat("second")
        assert provider.user == "User: first\nAssistant: ok\nUser: second\nAssistant:"

    def test_tools_need_a_model_that_accepts_them(self, mem):
        session = MemorySession(mem, lambda messages: "hi", tools=[{"type": "function", "function": {"name": "t"}}])
        with pytest.raises(MemorySessionError, match="tools"):
            session.chat("go")

    def test_render_message(self):
        assert render_message({"role": "assistant", "content": ""}) == "(empty)"
        assert render_message({"role": "assistant", "content": "ok", "tool_calls": [
            {"id": "c1", "name": "f", "arguments": {"b": 1, "a": 2}},
        ]}) == 'ok\n[tool call c1] f({"a": 2, "b": 1})'


# ─── Documents, state, extraction, branching ─────────────────────────────────


class TestDocumentsAndState:
    def test_split_document(self):
        text = "\n\n".join(f"Paragraph {i} " + "x" * 300 for i in range(10))
        chunks = split_document(text, 1000)
        assert all(len(c) <= 1000 for c in chunks) and len(chunks) > 3
        assert "\n\n".join(chunks) == text
        assert split_document("y" * 2500, 1000) == ["y" * 1000, "y" * 1000, "y" * 500]

    def test_ingest_stores_chunks_and_commits_pending_turns(self, mem):
        model = Recorder()
        session = MemorySession(mem, model, recent_turns=0)
        session.chat("an unsaved turn")
        document = "\n\n".join(
            [f"Section {i}. Filler text about scheduling and rooms. " * 6 for i in range(6)]
            + ["The warranty for the Zephyr 3000 lasts twenty-four months."]
        )
        count = session.ingest(document, title="Manual", chunk_chars=400)
        docs = [b for b in mem._storage.query_blocks({"type": BlockType.EPISODE})
                if b.metadata.custom_metadata["_role"] == "document"]
        assert count == len(docs) > 3
        assert all(b.metadata.custom_metadata["_title"] == "Manual" for b in docs)
        assert session.status()["uncommitted_turns"] == 0  # the pending turn went in with it
        session.chat("How long is the Zephyr 3000 warranty?")
        assert "twenty-four months" in model.system()

    def test_save_and_load_state(self, mem, tmp_path):
        session = MemorySession(mem, Recorder(), scope="session", session_id="trip")
        session.chat("My kitten Memmy loves shrimp.", enable_memory=True)
        session.chat("an unsaved turn")
        path = str(tmp_path / "trip.json")
        stamp = session.save_state(path)  # commits the unsaved turn first
        assert stamp["counts"]["blocks"] == 4 and stamp["scope"] == {"session_id": "trip"}

        other = MemBlock(storage=f"sqlite:///{tmp_path / 'other.db'}")
        model = Recorder()
        restored = MemorySession(other, model, scope="session", session_id="trip", state=path)
        assert restored.messages == []  # memory comes back, the transcript does not
        restored.chat("What does my kitten Memmy love?")
        assert "shrimp" in model.system()
        other.close()

    def test_extractor_facts_link_back_to_the_turn(self, mem):
        class Extractor:
            def extract_from_messages(self, messages, memblock=None, auto_store=True):
                assert [m["role"] for m in messages] == ["user", "assistant"]
                block = memblock.store("User's kitten is named Memmy", type=BlockType.FACT)
                assert block.deleted  # written inside the turn's transaction

                class Result:
                    block_ids = [block.id]
                return Result()

        session = MemorySession(mem, Recorder(), extractor=Extractor())
        session.chat("My kitten is called Memmy.", enable_memory=True)
        fact = mem.query(type=BlockType.FACT, limit=1)[0]
        first_episode = episodes(mem)[0]
        edges = mem._storage.get_edges(fact.id, direction="outgoing")
        assert [(e.target_id, e.relation) for e in edges] == [(first_episode.id, EdgeRelation.DERIVED_FROM)]

    def test_extractor_failure_saves_neither_the_turn_nor_its_facts(self, mem):
        class Broken:
            def extract_from_messages(self, messages, memblock=None, auto_store=True):
                memblock.store("half a fact", type=BlockType.FACT)
                raise RuntimeError("extraction failed")

        session = MemorySession(mem, Recorder(), extractor=Broken())
        with pytest.raises(MemorySessionError):
            session.chat("My kitten is called Memmy.", enable_memory=True)
        assert mem.query(limit=10) == [] and not session.uncertain

    def test_fork_branches_the_conversation(self, mem):
        session = MemorySession(mem, Recorder(default="a"))
        session.chat("shared start", enable_memory=True)
        session.chat("unsaved")
        branch = session.fork("branch")
        assert branch.id == "branch" and len(branch.messages) == len(session.messages) == 4
        branch.chat("only in the branch", enable_memory=True)
        session.chat("only in the original", enable_memory=True)
        texts = [b.content for b in mem._storage.query_blocks({"type": BlockType.EPISODE})]
        assert texts.count("unsaved") == 2  # each branch saved its own copy, under its own receipts
        assert texts.count("shared start") == 1

    def test_reset_keeps_memory(self, mem):
        session = MemorySession(mem, Recorder())
        session.chat("saved", enable_memory=True)
        session.chat("not saved")
        session.reset()
        assert session.messages == [] and len(episodes(mem)) == 2


# ─── Episodes are never pruned ───────────────────────────────────────────────


class TestEpisodesNeverFade:
    def test_prune_and_dedup_and_classifier_leave_episodes_alone(self, db):
        class DropEverything:
            def classify(self, content):
                return {"skip": True}

        mem = MemBlock(storage=db, on_duplicate="skip", classifier=DropEverything())
        assert mem.store("a fact the classifier drops") is None
        a = mem.store_episode("ok", role="user", session_id="s", turn=1, seq=0)
        b = mem.store_episode("ok", role="user", session_id="s", turn=2, seq=0)
        assert a.id != b.id and a.type == BlockType.EPISODE
        assert mem._decay.calculate_strength(a) == 1.0
        assert mem.prune(min_strength=0.99) == []
        assert len(mem.query(type=BlockType.EPISODE, limit=10)) == 2
        with pytest.raises(Exception):
            mem.store_episode("   ")
        mem.close()
