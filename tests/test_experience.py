"""Experience memory and recall on trouble (memblock/experience.py)."""

from __future__ import annotations

import pytest

from memblock import (
    BlockType,
    EdgeRelation,
    Experience,
    ExperienceBank,
    ExtractionError,
    MemBlock,
    MemorySession,
    TroubleDetector,
    ValidationError,
)
from memblock.decay import default_decay_rate_for
from memblock.session import EXPERIENCE_HEADER


@pytest.fixture
def mem(tmp_path):
    m = MemBlock(storage=f"sqlite:///{tmp_path / 'mem.db'}")
    yield m
    m.close()


@pytest.fixture
def bank(mem):
    b = ExperienceBank(mem)
    b.add(
        when="pytest fails with ModuleNotFoundError in a src-layout repository",
        do="Install the package in editable mode with pip install -e . before running tests",
        verify="Re-run the failing test and confirm the import error is gone",
        tags=["python", "testing"],
    )
    b.add(
        when="A database migration fails because a column already exists",
        do="Check the migration history table and mark the migration as applied",
        verify="Run the migration command again; it reports nothing to do",
    )
    return b


class FakeProvider:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return self.reply


TOOLS = [{"type": "function", "function": {"name": "run"}}]


def call(call_id, command):
    return {"content": "", "tool_calls": [{"id": call_id, "name": "run", "arguments": {"cmd": command}}]}


# ─── The bank ────────────────────────────────────────────────────────────────


class TestExperienceBank:
    def test_add_and_retrieve(self, mem, bank):
        found = bank.retrieve("tests fail: ModuleNotFoundError: No module named mypkg", limit=1)
        assert len(found) == 1
        experience = found[0]
        assert experience.when.startswith("pytest fails with ModuleNotFoundError")
        assert "pip install -e ." in experience.do and experience.tags == ["python", "testing"]
        assert experience.render() == (
            "WHEN: pytest fails with ModuleNotFoundError in a src-layout repository\n"
            "DO: Install the package in editable mode with pip install -e . before running tests\n"
            "VERIFY: Re-run the failing test and confirm the import error is gone"
        )
        block = mem.get(experience.id)
        assert block.type == BlockType.EXPERIENCE
        assert block.metadata.decay_rate == default_decay_rate_for(BlockType.EXPERIENCE) == 0.002
        assert len(bank.all()) == 2

    def test_retrieval_picks_the_matching_one(self, bank):
        assert "migration history" in bank.retrieve("migration fails: column already exists", limit=1)[0].do
        assert bank.retrieve("   ") == [] and bank.retrieve("anything", limit=0) == []
        assert bank.retrieve("quantum chromodynamics lattice") == []

    def test_needs_when_and_do(self, mem):
        with pytest.raises(ValidationError):
            ExperienceBank(mem).add(when="only a condition", do="")
        with pytest.raises(ValidationError):
            ExperienceBank(mem).add(Experience(when="", do="something"))

    def test_links_back_to_the_run_it_came_from(self, mem):
        episode = mem.store_episode("ran pytest; it failed with ModuleNotFoundError", role="tool")
        block = ExperienceBank(mem).add(
            when="pytest import error", do="pip install -e .", derived_from=[episode.id],
        )
        edges = mem._storage.get_edges(block.id, direction="outgoing")
        assert [(e.target_id, e.relation) for e in edges] == [(episode.id, EdgeRelation.DERIVED_FROM)]

    def test_block_written_by_hand_is_read_back(self, mem):
        block = mem.store("WHEN: the build is slow\nDO: enable the cache", type=BlockType.EXPERIENCE)
        experience = Experience.from_block(block)
        assert (experience.when, experience.do, experience.verify) == ("the build is slow", "enable the cache", "")

    def test_scope(self, mem):
        ExperienceBank(mem, project_id="alpha").add(when="alpha build breaks", do="clear alpha cache")
        assert ExperienceBank(mem, project_id="beta").retrieve("build breaks") == []
        assert len(ExperienceBank(mem, project_id="alpha").retrieve("build breaks")) == 1


class TestLearn:
    def test_learn_from_a_run(self, mem):
        provider = FakeProvider(
            '```json\n[{"when": "npm install fails with EACCES", "do": "Fix the cache owner, '
            'do not use sudo", "verify": "npm install completes"},'
            ' {"when": "", "do": "ignored: no condition"}, {"when": "a", "do": "b"},'
            ' {"when": "c", "do": "d"}]\n```'
        )
        episode = mem.store_episode("npm install -> EACCES", role="tool")
        run = [{"role": "user", "content": "install deps"}, {"role": "tool", "content": "EACCES"}]
        stored = ExperienceBank(mem).learn(
            run, provider, outcome="success", derived_from=[episode.id], max_items=3, tags=["node"],
        )
        assert [b.type for b in stored] == [BlockType.EXPERIENCE, BlockType.EXPERIENCE]
        assert all(b.metadata.confidence == 0.8 and b.tags == ["node"] for b in stored)
        system, user = provider.calls[0]
        assert "JSON array" in system and "Outcome of the run: success" in user and "Tool: EACCES" in user
        assert ExperienceBank(mem).retrieve("npm install EACCES")[0].do.startswith("Fix the cache owner")

    def test_empty_and_invalid_answers(self, mem):
        assert ExperienceBank(mem).learn("a run", FakeProvider("[]")) == []
        with pytest.raises(ExtractionError):
            ExperienceBank(mem).learn("a run", FakeProvider("I could not find anything."))
        with pytest.raises(ExtractionError):
            ExperienceBank(mem).learn("a run", FakeProvider("[{broken json]"))


# ─── Trouble detection ───────────────────────────────────────────────────────


class TestTroubleDetector:
    def test_failures_in_a_row(self):
        detector = TroubleDetector(max_failures=2)
        assert detector.observe("run", {"cmd": "a"}, "Traceback (most recent call last): ...") is None
        assert detector.observe("run", {"cmd": "b"}, "exit code 1") == "2 tool calls in a row failed"
        # a success in between resets the count
        assert detector.observe("run", {"cmd": "c"}, "command not found: foo") is None
        assert detector.observe("run", {"cmd": "d"}, "all 12 tests passed") is None
        assert detector.observe("run", {"cmd": "e"}, "Permission denied") is None

    def test_empty_results_in_a_row(self):
        detector = TroubleDetector(max_empty=2)
        assert detector.observe("search", {"q": "a"}, "") is None
        assert detector.observe("search", {"q": "b"}, " [] ") == "2 tool calls in a row returned nothing"

    def test_same_call_repeated(self):
        detector = TroubleDetector(max_failures=9, max_repeats=3)
        assert detector.observe("read", {"path": "a", "n": 1}, "contents") is None
        assert detector.observe("read", {"n": 1, "path": "a"}, "contents") is None  # same call, keys reordered
        assert detector.observe("read", {"path": "a", "n": 1}, "contents") == \
            "the same call to read was made 3 times"

    def test_explicit_flag_wins_over_the_text(self):
        detector = TroubleDetector(max_failures=1)
        assert detector.observe("run", {}, "the word error appears in a passing log", is_error=False) is None
        assert detector.observe("run", {}, "looks fine", is_error=True) == "1 tool calls in a row failed"

    def test_reset(self):
        detector = TroubleDetector(max_failures=2)
        detector.observe("run", {}, "failed")
        detector.reset()
        assert detector.observe("run", {}, "failed") is None
        with pytest.raises(ValueError):
            TroubleDetector(max_failures=0)


# ─── In a session ────────────────────────────────────────────────────────────


class Scripted:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, messages, tools=None):
        self.prompts.append(messages)
        return self.replies.pop(0)

    def system(self, index):
        first = self.prompts[index][0]
        return first["content"] if first["role"] == "system" else ""


class TestSessionExperience:
    def test_added_before_the_task_starts(self, mem, bank):
        model = Scripted(["done"])
        session = MemorySession(mem, model, experience="start", experience_bank=bank)
        session.chat("The tests fail with ModuleNotFoundError, please fix")
        assert EXPERIENCE_HEADER in model.system(0) and "pip install -e ." in model.system(0)
        assert session.experience_log[0]["reason"] == "start"

    def test_added_only_when_the_agent_gets_stuck_and_only_once(self, mem, bank):
        model = Scripted([
            call("c1", "pytest"), call("c2", "pytest -x"), call("c3", "pytest -q"),
            call("c4", "pytest -k a"), "fixed", "next task done",
        ])
        session = MemorySession(mem, model, tools=TOOLS, experience="on_trouble", experience_bank=bank)
        session.chat("Make the test suite pass")
        assert EXPERIENCE_HEADER not in model.system(0)

        session.submit_tool_results([{"tool_call_id": "c1", "content": "ModuleNotFoundError: No module named mypkg"}])
        assert EXPERIENCE_HEADER not in model.system(1)  # one failure is not trouble yet

        session.submit_tool_results([{"tool_call_id": "c2", "content": "ModuleNotFoundError: No module named mypkg"}])
        assert "pip install -e ." in model.system(2)  # second failure in a row
        assert session.experience_log == [{
            "turn": 1,
            "reason": "2 tool calls in a row failed",
            "experience_ids": [session.experience_log[0]["experience_ids"][0]],
        }]

        session.submit_tool_results([{"tool_call_id": "c3", "content": "error: still failing"}])
        session.submit_tool_results([{"tool_call_id": "c4", "content": "error: still failing"}])
        assert len(session.experience_log) == 1  # at most one per task
        assert "pip install -e ." in model.system(4)  # and it stays for the rest of the task

        session.chat("A different task")
        assert EXPERIENCE_HEADER not in model.system(5)  # a new task starts clean

    def test_no_matching_experience_adds_nothing(self, mem):
        model = Scripted([call("c1", "x"), call("c2", "y"), "gave up"])
        session = MemorySession(mem, model, tools=TOOLS, experience="on_trouble")
        session.chat("Do the thing")
        session.submit_tool_results([{"tool_call_id": "c1", "content": "failed"}])
        session.submit_tool_results([{"tool_call_id": "c2", "content": "failed"}])
        assert session.experience_log == [] and EXPERIENCE_HEADER not in model.system(2)

    def test_experience_is_kept_out_of_ordinary_memory_recall(self, mem, bank):
        model = Scripted(["ok"])
        MemorySession(mem, model, experience="off").chat("pytest fails with ModuleNotFoundError")
        assert model.system(0) == ""  # experience is added only through the experience path

    def test_stuck_agent_recovers_once_it_is_shown_the_experience(self, mem, bank):
        """Mechanism check with a scripted agent: it repeats a failing command
        until guidance containing the fix appears in its prompt."""

        class Agent:
            def __init__(self):
                self.step = 0

            def __call__(self, messages, tools=None):
                self.step += 1
                system = messages[0]["content"] if messages[0]["role"] == "system" else ""
                last = messages[-1]
                if last["role"] == "tool" and "passed" in last["content"]:
                    return "All tests pass now."
                command = "pip install -e . && pytest" if "pip install -e ." in system else "pytest"
                return call(f"c{self.step}", command)

        def run_tool(tool_call):
            command = tool_call["arguments"]["cmd"]
            return "12 passed" if "pip install -e ." in command else "ModuleNotFoundError: No module named mypkg"

        def run(mode):
            session = MemorySession(mem, Agent(), tools=TOOLS, experience=mode, experience_bank=bank)
            result = session.chat("Make the test suite pass")
            steps = 0
            while result["requires_action"] and steps < 8:
                steps += 1
                result = session.submit_tool_results([
                    {"tool_call_id": c["id"], "content": run_tool(c)} for c in result["tool_calls"]
                ])
            return result, steps

        stuck, stuck_steps = run("off")
        assert stuck["requires_action"] and stuck_steps == 8  # never gets out

        helped, helped_steps = run("on_trouble")
        assert helped["answer"] == "All tests pass now." and helped_steps == 3
