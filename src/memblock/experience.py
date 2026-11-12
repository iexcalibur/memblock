"""Experience memory — reusable guidance learned from past agent runs.

An experience is a short procedural record with three parts:

    WHEN   the conditions it applies to
    DO     the recommended procedure
    VERIFY how to check that it worked

`ExperienceBank` stores them as EXPERIENCE blocks and retrieves the ones that
match a new task. `ExperienceBank.learn()` asks an LLM to distil them from a
finished run. `TroubleDetector` watches tool results for signs that an agent is
stuck (failures, empty results, the same call repeated); `MemorySession` uses
it to add the most relevant experience to the prompt at that moment
(`experience="on_trouble"`), or before the task starts (`experience="start"`).

    bank = ExperienceBank(mem)
    bank.add(
        when="pytest fails with ModuleNotFoundError in a src-layout repo",
        do="Install the package in editable mode: pip install -e .",
        verify="Re-run the failing test; the import error is gone",
    )
    for experience in bank.retrieve("tests fail: ModuleNotFoundError: mypkg"):
        print(experience.render())
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from memblock.block import Block
from memblock.errors import ExtractionError, ValidationError
from memblock.types import BlockType, EdgeRelation, SourceType


EXPERIENCE_KEY = "_experience"

LEARN_SYSTEM_PROMPT = """You turn one finished agent run into reusable experience for future runs.
Return ONLY a JSON array (possibly empty) of objects:
[{"when": "...", "do": "...", "verify": "..."}]
when: the conditions under which this applies, stated so it can be recognised in a new task.
do: the recommended procedure, as concrete steps.
verify: how to check that it worked.
Keep each field to one or two sentences. Include only guidance that would help on a different,
similar task. Leave out anything specific to this one run (file names that will not recur,
one-off values). If the run teaches nothing reusable, return []."""

LEARN_USER_TEMPLATE = "Outcome of the run: {outcome}\n\nThe run:\n{trajectory}\n\nJSON:"


@dataclass
class Experience:
    """One piece of procedural guidance."""

    when: str
    do: str
    verify: str = ""
    tags: list[str] = field(default_factory=list)
    id: str | None = None

    def render(self) -> str:
        lines = [f"WHEN: {self.when}", f"DO: {self.do}"]
        if self.verify:
            lines.append(f"VERIFY: {self.verify}")
        return "\n".join(lines)

    @classmethod
    def from_block(cls, block: Block, content: str | None = None) -> "Experience":
        data = (block.metadata.custom_metadata or {}).get(EXPERIENCE_KEY)
        if isinstance(data, dict) and data.get("when") and data.get("do"):
            return cls(
                when=str(data["when"]), do=str(data["do"]), verify=str(data.get("verify") or ""),
                tags=list(block.tags), id=block.id,
            )
        # A block stored by hand: read the parts back from its text.
        parts = {"when": "", "do": "", "verify": ""}
        for line in (content if content is not None else block.content).splitlines():
            head, _, rest = line.partition(":")
            if head.strip().lower() in parts:
                parts[head.strip().lower()] = rest.strip()
        return cls(
            when=parts["when"] or (content or block.content), do=parts["do"],
            verify=parts["verify"], tags=list(block.tags), id=block.id,
        )


class ExperienceBank:
    """Stores and retrieves experiences in a `MemBlock`."""

    def __init__(
        self,
        memory: Any,
        *,
        org_id: str | None = None,
        project_id: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        self.memory = memory
        self._scope = {
            key: value
            for key, value in (("org_id", org_id), ("project_id", project_id), ("agent_id", agent_id))
            if value is not None
        }

    def add(
        self,
        experience: Experience | None = None,
        *,
        when: str | None = None,
        do: str | None = None,
        verify: str = "",
        tags: list[str] | None = None,
        derived_from: list[str] | None = None,
        confidence: float = 1.0,
        receipt: str | None = None,
    ) -> Block:
        """
        Store one experience. Pass an `Experience`, or `when` and `do`.

        `derived_from` lists the blocks (usually the run's episodes) it was
        learned from; each gets a DERIVED_FROM link.
        """
        if experience is None:
            experience = Experience(when=when or "", do=do or "", verify=verify, tags=tags or [])
        if not experience.when.strip() or not experience.do.strip():
            raise ValidationError("An experience needs both 'when' and 'do'")
        block = self.memory.store(
            experience.render(),
            type=BlockType.EXPERIENCE,
            source=SourceType.INFERRED,
            confidence=confidence,
            tags=list(experience.tags) or None,
            metadata={
                EXPERIENCE_KEY: {
                    "when": experience.when, "do": experience.do, "verify": experience.verify,
                }
            },
            receipt=receipt,
            **self._scope,
        )
        if block is not None:
            for source_id in derived_from or []:
                self.memory.link(block.id, source_id, relation=EdgeRelation.DERIVED_FROM)
            experience.id = block.id
        return block

    def retrieve(self, task: str, limit: int = 3) -> list[Experience]:
        """The experiences that best match a task description or an error message."""
        if not isinstance(task, str) or not task.strip() or limit < 1:
            return []
        blocks = self.memory.query(
            text_search=task, type=BlockType.EXPERIENCE, limit=limit, **self._scope,
        )
        found = []
        for block in blocks[:limit]:
            text = self.memory._plaintext(block)
            if text is not None:
                found.append(Experience.from_block(block, text))
        return found

    def all(self) -> list[Experience]:
        blocks = self.memory.query(type=BlockType.EXPERIENCE, limit=10_000, **self._scope)
        return [Experience.from_block(b, self.memory._plaintext(b) or b.content) for b in blocks]

    def learn(
        self,
        trajectory: str | list[dict[str, Any]],
        provider: Any,
        *,
        outcome: str = "unknown",
        derived_from: list[str] | None = None,
        max_items: int = 3,
        tags: list[str] | None = None,
    ) -> list[Block]:
        """
        Distil a finished run into experiences and store them.

        Args:
            trajectory: The run, as text or as a message list.
            provider: An LLM with `complete(system_prompt, user_prompt)` — any
                `memblock.extraction` provider, or your own.
            outcome: How the run ended ("success", "failure", or a sentence).
            derived_from: Blocks to link the new experiences back to.
            max_items: Most experiences taken from one run.
        """
        if not isinstance(trajectory, str):
            trajectory = "\n".join(
                f"{m.get('role', 'user').title()}: {m.get('content') or ''}" for m in trajectory
            )
        raw = provider.complete(
            LEARN_SYSTEM_PROMPT,
            LEARN_USER_TEMPLATE.format(outcome=outcome, trajectory=trajectory),
        )
        items = _parse_json_array(raw)
        stored = []
        for item in items[:max_items]:
            when, do = str(item.get("when") or "").strip(), str(item.get("do") or "").strip()
            if not when or not do:
                continue
            block = self.add(
                when=when, do=do, verify=str(item.get("verify") or "").strip(),
                tags=tags, derived_from=derived_from, confidence=0.8,
            )
            if block is not None:
                stored.append(block)
        return stored


def _parse_json_array(raw: str) -> list[dict[str, Any]]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        raise ExtractionError("The model did not return a JSON array of experiences")
    try:
        data = json.loads(text[start:end + 1])
    except ValueError as error:
        raise ExtractionError(f"The model returned invalid JSON: {error}") from error
    return [item for item in data if isinstance(item, dict)]


# ─── Trouble detection ───────────────────────────────────────────────────────


# "error" and "exception" are matched at the end of a word, so class names such
# as ModuleNotFoundError and NullPointerException count; "0 errors" does not.
_FAILURE = re.compile(
    r"(?i)(error\b|exception\b|\btraceback\b|\bfail(ed|ure|s)?\b|\bfatal\b|\bpanic\b"
    r"|\bnot found\b|\bno such file\b|\bpermission denied\b|\btimed out\b"
    r"|\bexit (code|status) [1-9]\d*\b)"
)
_EMPTY = {"", "[]", "{}", "null", "none", "no results", "no results found", "no matches", "no output"}


class TroubleDetector:
    """
    Spots an agent that is stuck, from its tool results.

    `observe()` returns a reason the first time one of these happens, else None:
      - `max_failures` results in a row that look like failures,
      - `max_empty` results in a row that are empty,
      - the same tool call (same name and arguments) made `max_repeats` times.

    Only what the run itself shows is used: nothing about the right answer.
    Call `reset()` at the start of each task.
    """

    def __init__(self, max_failures: int = 2, max_empty: int = 2, max_repeats: int = 3) -> None:
        if min(max_failures, max_empty, max_repeats) < 1:
            raise ValueError("thresholds must be at least 1")
        self.max_failures = max_failures
        self.max_empty = max_empty
        self.max_repeats = max_repeats
        self.reset()

    def reset(self) -> None:
        self._failures = 0
        self._empty = 0
        self._calls: dict[str, int] = {}

    @staticmethod
    def looks_failed(content: str) -> bool:
        return bool(_FAILURE.search(content or ""))

    @staticmethod
    def looks_empty(content: str) -> bool:
        return (content or "").strip().lower() in _EMPTY

    def observe(
        self,
        name: str,
        arguments: Any,
        content: str,
        is_error: bool | None = None,
    ) -> str | None:
        failed = bool(is_error) if is_error is not None else self.looks_failed(content)
        empty = not failed and self.looks_empty(content)
        self._failures = self._failures + 1 if failed else 0
        self._empty = self._empty + 1 if empty else 0
        try:
            key = f"{name}:{json.dumps(arguments, sort_keys=True, default=str)}"
        except (TypeError, ValueError):
            key = f"{name}:{arguments!r}"
        self._calls[key] = self._calls.get(key, 0) + 1

        if self._failures >= self.max_failures:
            self._failures = 0
            return f"{self.max_failures} tool calls in a row failed"
        if self._empty >= self.max_empty:
            self._empty = 0
            return f"{self.max_empty} tool calls in a row returned nothing"
        if self._calls[key] >= self.max_repeats:
            self._calls[key] = 0
            return f"the same call to {name or 'a tool'} was made {self.max_repeats} times"
        return None
