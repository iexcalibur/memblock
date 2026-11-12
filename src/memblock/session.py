"""MemorySession — a memory layer between your app and any chat model.

The app talks to the session; the session talks to the model. Each turn it

1. reads memory for the new message and puts what it finds in the prompt,
2. calls the model,
3. commits the finished turn to memory, word for word, and drops it from the
   prompt.

So the prompt stays about the same size however long the conversation runs, and
nothing that was said is lost: committed turns live in MemBlock as EPISODE
blocks and come back through search when a later message needs them.

    from memblock import MemBlock, MemorySession

    def my_model(messages):            # any model: return the reply text
        return client.chat(messages)

    session = MemorySession(MemBlock("sqlite:///memory.db"), my_model)
    session.chat("My kitten Memmy is three months old.", enable_memory=True)
    session.chat("How old is my kitten?")          # answered from memory

The model
---------
`model` is any callable taking the message list (and, if it accepts one, a
`tools` keyword) and returning the reply: a string, a `ModelReply`, or a dict
with `text`/`content`, optional `tool_calls` and `finish_reason`. An object
with `complete(system_prompt, user_prompt)` (MemBlock's own LLM providers) also
works, for text-only chat. Messages use the common
`{"role": ..., "content": ...}` shape.

Write boundaries
----------------
Only a completed turn is committed. A chain of tool calls and their results is
one turn and is committed after the final answer. A reply cut off by the length
limit is kept in the prompt and committed later, with the next completed turn,
`commit_history()` or `save_state()`.

Commits are all-or-nothing (`MemBlock.transaction`) and every message is stored
under a receipt, so a commit that is retried after a failure cannot store a
turn twice.
"""

from __future__ import annotations

import inspect
import json
import re
import threading
import uuid
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from memblock.block import Block
from memblock.context import estimate_tokens
from memblock.errors import MemorySessionError
from memblock.transactions import ReceiptStatus, is_internal
from memblock.types import BlockType, EdgeRelation, now_utc


MEMORY_HEADER = (
    "Relevant memory from earlier conversations "
    "(quoted records and saved facts; use it when it helps):"
)
EXPERIENCE_HEADER = "Experience from earlier runs that may apply here:"


# ─── Model adapter ───────────────────────────────────────────────────────────


@dataclass
class ModelReply:
    """What a model returned for one call."""

    text: str = ""
    #: Each: {"id": ..., "name": ..., "arguments": ...}
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    #: "stop" (finished), "length" (cut off) or "tool_calls" (wants tool results)
    finish_reason: str = "stop"
    raw: Any = None


_FINISH = {
    "stop": "stop", "end_turn": "stop", "stop_sequence": "stop", "eos": "stop",
    "completed": "stop", "length": "length", "max_tokens": "length",
    "tool_calls": "tool_calls", "tool_use": "tool_calls", "function_call": "tool_calls",
}


def normalize_reply(value: Any) -> ModelReply:
    """Turn whatever a model callable returned into a `ModelReply`."""
    if isinstance(value, ModelReply):
        reply = value
    elif isinstance(value, str):
        reply = ModelReply(text=value)
    elif isinstance(value, dict):
        text = value.get("text")
        if text is None:
            text = value.get("content")
        if text is None:
            text = value.get("answer")
        calls = []
        for index, call in enumerate(value.get("tool_calls") or []):
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            calls.append({
                "id": str(call.get("id") or f"call_{index}"),
                "name": call.get("name") or function.get("name") or "",
                "arguments": call.get("arguments", function.get("arguments")),
            })
        finish = value.get("finish_reason") or value.get("stop_reason") or "stop"
        reply = ModelReply(
            text=text if isinstance(text, str) else "",
            tool_calls=calls,
            finish_reason=_FINISH.get(str(finish), "stop"),
            raw=value,
        )
    else:
        raise TypeError(
            "A model must return a string, a ModelReply or a dict, "
            f"not {type(value).__name__}"
        )
    if reply.tool_calls:
        reply.finish_reason = "tool_calls"
    return reply


def _flatten(messages: list[dict[str, Any]]) -> tuple[str, str]:
    """Message list → (system, user) for providers that take two strings."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system" and m.get("content"))
    rest = [m for m in messages if m["role"] != "system"]
    if len(rest) == 1:
        return system, rest[0].get("content") or ""
    lines = [f"{m['role'].title()}: {m.get('content') or ''}" for m in rest]
    lines.append("Assistant:")
    return system, "\n".join(lines)


def adapt_model(model: Any) -> Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], ModelReply]:
    """Wrap any supported model as `call(messages, tools) -> ModelReply`."""
    if not callable(model) and hasattr(model, "complete"):
        def call_provider(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> ModelReply:
            if tools:
                raise MemorySessionError(
                    "This model takes plain text only; pass a callable to use tools",
                    stage="model",
                )
            system, user = _flatten(messages)
            return normalize_reply(model.complete(system, user))
        return call_provider

    if not callable(model):
        raise TypeError("model must be a callable or have a complete(system, user) method")

    try:
        parameters = inspect.signature(model).parameters
        takes_tools = "tools" in parameters or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        )
    except (TypeError, ValueError):
        takes_tools = False

    def call(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> ModelReply:
        if takes_tools:
            return normalize_reply(model(messages, tools=tools))
        if tools:
            raise MemorySessionError(
                "The session has tools, but the model callable has no `tools` parameter",
                stage="model",
            )
        return normalize_reply(model(messages))

    return call


# ─── Helpers ─────────────────────────────────────────────────────────────────

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a about after again all also am an and any are as at be because been before being both "
    "but by can could did do does doing don down during each few for from further had has have "
    "having he her here hers him his how i if in into is it its just ll me more most my no nor "
    "not now of off on once only or other our out over own re s same she should so some such t "
    "than that the their them then there these they this those through to too under until up ve "
    "very was we were what when where which while who whom why will with would you your yours".split()
)


def content_words(text: str) -> set[str]:
    """The words of a text that carry meaning (no 'the', 'and', 'what', ...)."""
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1}


def _same_word(a: str, b: str) -> bool:
    """Loose match that survives the store's stemming ('running' ~ 'run')."""
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) <= 3:
        return long.startswith(short)
    return short[:4] == long[:4]


def shares_content_word(query_words: set[str], text: str) -> bool:
    words = content_words(text)
    return any(_same_word(q, w) for q in query_words for w in words)



def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def render_message(message: dict[str, Any]) -> str:
    """The exact text stored for one message. Tool calls are kept as JSON lines."""
    parts = []
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        parts.append(content)
    for call in message.get("tool_calls") or []:
        arguments = call.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, sort_keys=True, default=str)
        parts.append(f"[tool call {call.get('id', '')}] {call.get('name', '')}({arguments})")
    return "\n".join(parts) if parts else "(empty)"


def split_document(document: str, chunk_chars: int) -> list[str]:
    """Split on paragraph boundaries into chunks of at most `chunk_chars`."""
    chunks: list[str] = []
    current = ""
    for paragraph in document.split("\n\n"):
        while len(paragraph) > chunk_chars:  # one oversized paragraph
            if current.strip():
                chunks.append(current)
                current = ""
            chunks.append(paragraph[:chunk_chars])
            paragraph = paragraph[chunk_chars:]
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) > chunk_chars and current.strip():
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current.strip():
        chunks.append(current)
    return chunks


@dataclass
class _Turn:
    index: int
    messages: list[dict[str, Any]]
    receipt: str
    at: Any
    enable_memory: bool = False
    completed: bool = False
    pending_calls: list[dict[str, Any]] = field(default_factory=list)
    block_ids: list[str] = field(default_factory=list)

    def receipts(self) -> list[str]:
        return [f"{self.receipt}:{seq}" for seq in range(len(self.messages))]


# ─── Session ─────────────────────────────────────────────────────────────────


class MemorySession:
    """One conversation with memory. See the module docstring."""

    def __init__(
        self,
        memory: Any,
        model: Any,
        *,
        session_id: str | None = None,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        memory_read: bool = True,
        recall_limit: int = 8,
        recall_budget: int = 1500,
        max_item_chars: int = 1200,
        recent_turns: int = 2,
        scope: str = "user",
        relevance_floor: bool = True,
        extractor: Any = None,
        experience: str = "off",
        experience_bank: Any = None,
        trouble: Any = None,
        max_experience_injections: int = 1,
        state: str | None = None,
        clock: Callable[[], Any] | None = None,
    ) -> None:
        """
        Args:
            memory: A `MemBlock`.
            model: The chat model (see the module docstring).
            session_id: Conversation ID; pass the same one to continue numbering
                the turns of an earlier session.
            system_prompt: Your own system prompt. Memory is appended to it.
            tools: Tool schemas, passed through to the model. The app runs the
                tools and returns results with `submit_tool_results()`.
            memory_read: Read memory into the prompt (default on).
            recall_limit: Most memory items put in one prompt.
            recall_budget: Most tokens (estimated) of memory in one prompt.
            max_item_chars: Longer items are shortened in the prompt only; the
                store keeps the full text.
            recent_turns: Committed turns kept word for word at the end of the
                prompt, so "and the second one?" still has its context. 0 keeps
                none.
            scope: "user" recalls from everything in the store; "session" only
                from this conversation.
            relevance_floor: Leave out keyword matches that share only filler
                words ("the", "and", "what") with the message. Applies when the
                store has no embeddings; with embeddings, matches by meaning
                are kept even when no word is shared.
            extractor: Optional `memblock.extraction.LLMExtractor`. When given,
                each committed turn is also distilled into facts that link back
                to the turn with DERIVED_FROM. Without it, turns are kept word
                for word only.
            experience: "off", "start" (add relevant experience before the model
                starts a task) or "on_trouble" (add it when tool results show
                the model is stuck). See `memblock.experience`.
            experience_bank: An `ExperienceBank`; one is created when needed.
            trouble: A `TroubleDetector`; one with defaults is created when needed.
            max_experience_injections: Most times experience is added per task.
            state: A snapshot file to load before the first turn.
            clock: Returns the time a turn happened (a timezone-aware
                datetime). Defaults to now; set it when replaying a past
                conversation so each turn keeps its original date.
        """
        if scope not in ("user", "session"):
            raise ValueError("scope must be 'user' or 'session'")
        if experience not in ("off", "start", "on_trouble"):
            raise ValueError("experience must be 'off', 'start' or 'on_trouble'")
        if recall_limit < 1 or recall_budget < 1 or recent_turns < 0:
            raise ValueError("recall_limit and recall_budget must be positive; recent_turns >= 0")
        if tools is not None and not isinstance(tools, list):
            raise ValueError("tools must be a list of tool schemas")

        self.memory = memory
        self._model = adapt_model(model)
        self._raw_model = model
        self.id = session_id or f"ses_{uuid.uuid4().hex[:12]}"
        self.system_prompt = system_prompt
        self.tools = deepcopy(tools or [])
        self.memory_read = bool(memory_read)
        self.recall_limit = recall_limit
        self.recall_budget = recall_budget
        self.max_item_chars = max_item_chars
        self.recent_turns = recent_turns
        self.scope = scope
        self.relevance_floor = relevance_floor
        self.extractor = extractor
        self.clock = clock or now_utc

        self.experience = experience
        self.max_experience_injections = max_experience_injections
        self._bank = experience_bank
        self._trouble = trouble
        if experience != "off":
            from memblock.experience import ExperienceBank, TroubleDetector
            if self._bank is None:
                self._bank = ExperienceBank(memory)
            if self._trouble is None:
                self._trouble = TroubleDetector()
        self._pinned_experience: str | None = None
        self._injections = 0
        #: One entry per time experience was added: {"turn", "reason", "experience_ids"}
        self.experience_log: list[dict[str, Any]] = []

        self._turns: list[_Turn] = []
        self._recent: deque[_Turn] = deque(maxlen=recent_turns) if recent_turns else deque(maxlen=0)
        self._tool_turn: _Turn | None = None
        self._pending: list[_Turn] | None = None
        self._pending_receipts: list[str] = []
        self._next_turn = self._resume_turn_number() if session_id else 1
        self._lock = threading.RLock()
        self.uncertain = False
        self.closed = False
        #: One record per model call.
        self.responses: list[dict[str, Any]] = []
        #: Size of the most recent prompt: {"messages", "chars", "tokens", "recalled"}
        self.last_prompt: dict[str, Any] = {}

        if state is not None:
            self.load_state(state)

    # ── Guards ───────────────────────────────────────────────────────────

    def _resume_turn_number(self) -> int:
        try:
            blocks = self.memory._storage.query_blocks(
                {"session_id": self.id, "type": BlockType.EPISODE}
            )
        except Exception:
            return 1
        turns = [
            (b.metadata.custom_metadata or {}).get("_turn") for b in blocks
        ]
        return 1 + max((t for t in turns if isinstance(t, int)), default=0)

    @contextmanager
    def _operation(self) -> Iterator[None]:
        with self._lock:
            if self.closed:
                raise MemorySessionError("Session is closed; create a new session")
            if self.uncertain:
                raise MemorySessionError(
                    "A commit did not finish cleanly; call recover() before going on",
                    uncertain=True,
                )
            yield

    def _check_no_tools(self) -> None:
        if self._tool_turn is not None:
            raise ValueError("Submit the pending tool results before this operation")

    # ── Reading memory ───────────────────────────────────────────────────

    def _format(self, block: Block, text: str) -> str:
        if len(text) > self.max_item_chars:
            text = text[: self.max_item_chars].rstrip() + " …"
        text = text.replace("\n", " ")
        if block.type == BlockType.EPISODE:
            custom = block.metadata.custom_metadata or {}
            when = block.metadata.happened_at or block.metadata.created_at
            return f"- [{when.strftime('%Y-%m-%d')}] {custom.get('_role', 'user')}: {text}"
        return f"- ({block.type.value}) {text}"

    def recall(self, query: str, exclude: set[str] | None = None) -> tuple[str, list[str]]:
        """What the session would put in the prompt for this message: the
        memory text, and the IDs of the blocks it came from."""
        exclude = exclude or set()
        scope = {"session_id": self.id} if self.scope == "session" else {}
        blocks = self.memory.query(text_search=query, limit=self.recall_limit * 3, **scope)
        wanted = content_words(query)
        floor = self.relevance_floor and bool(wanted) and not self.memory.has_embeddings
        lines: list[str] = []
        ids: list[str] = []
        used = estimate_tokens(MEMORY_HEADER)
        for block in blocks:
            if len(lines) >= self.recall_limit:
                break
            if block.id in exclude or is_internal(block) or block.type == BlockType.EXPERIENCE:
                continue
            text = self.memory._plaintext(block)
            if text is None:
                continue
            if floor and not shares_content_word(wanted, text):
                continue
            line = self._format(block, text)
            cost = estimate_tokens(line)
            if used + cost > self.recall_budget:
                continue
            lines.append(line)
            ids.append(block.id)
            used += cost
        if not lines:
            return "", []
        return MEMORY_HEADER + "\n" + "\n".join(lines), ids

    def _build_messages(self, turn: _Turn) -> tuple[list[dict[str, Any]], list[str]]:
        query = next((m["content"] for m in turn.messages if m["role"] == "user"), "")
        in_prompt = {b for t in [*self._recent, *self._turns] for b in t.block_ids}
        memory_text, recalled = self.recall(query, in_prompt) if self.memory_read else ("", [])
        parts = [p for p in (self.system_prompt, memory_text, self._pinned_experience) if p]
        messages: list[dict[str, Any]] = []
        if parts:
            messages.append({"role": "system", "content": "\n\n".join(parts)})
        for earlier in [*self._recent, *self._turns]:
            messages.extend(deepcopy(earlier.messages))
        messages.extend(deepcopy(turn.messages))
        return messages, recalled

    # ── Experience ───────────────────────────────────────────────────────

    def _pin_experience(self, turn: _Turn, query: str, reason: str, limit: int) -> None:
        if self._bank is None or self._injections >= self.max_experience_injections:
            return
        found = self._bank.retrieve(query, limit=limit)
        if not found:
            return
        self._pinned_experience = EXPERIENCE_HEADER + "\n" + "\n\n".join(e.render() for e in found)
        self._injections += 1
        self.experience_log.append({
            "turn": turn.index,
            "reason": reason,
            "experience_ids": [e.id for e in found],
        })

    # ── One model call ───────────────────────────────────────────────────

    def _generate(self, turn: _Turn) -> ModelReply:
        messages, recalled = self._build_messages(turn)
        chars = sum(len(m.get("content") or "") for m in messages)
        self.last_prompt = {
            "messages": len(messages),
            "chars": chars,
            "tokens": estimate_tokens("x" * chars),
            "recalled": recalled,
        }
        reply = self._model(messages, self.tools or None)
        assistant: dict[str, Any] = {"role": "assistant", "content": reply.text}
        if reply.tool_calls:
            assistant["tool_calls"] = deepcopy(reply.tool_calls)
        turn.messages.append(assistant)
        turn.pending_calls = deepcopy(reply.tool_calls)
        return reply

    def _finish(self, turn: _Turn, reply: ModelReply, return_dict: bool) -> Any:
        result: dict[str, Any] = {
            "answer": reply.text,
            "turn": turn.index,
            "finish_reason": reply.finish_reason,
            "generation_completed": reply.finish_reason == "stop",
            "turn_completed": reply.finish_reason == "stop",
            "requires_action": bool(reply.tool_calls),
            "tool_calls": deepcopy(reply.tool_calls),
            "memory_updated": False,
            "committed_turns": 0,
            "prompt": dict(self.last_prompt),
        }
        if reply.tool_calls:
            self._tool_turn = turn
            result["turn_completed"] = False
            self.responses.append(result)
            return result
        self._tool_turn = None
        turn.completed = reply.finish_reason == "stop"
        self._turns.append(turn)
        if turn.enable_memory and turn.completed:
            try:
                result["committed_turns"] = self._commit(answer=reply.text)
            except MemorySessionError:
                self.responses.append(result)
                raise
            result["memory_updated"] = True
        self.responses.append(result)
        return result if (return_dict or self.tools) else reply.text

    # ── Writing memory ───────────────────────────────────────────────────

    def _write_turn(self, turn: _Turn) -> None:
        ids = []
        for seq, message in enumerate(turn.messages):
            block = self.memory.store_episode(
                render_message(message),
                role=message["role"],
                session_id=self.id,
                turn=turn.index,
                seq=seq,
                happened_at=turn.at,
                receipt=f"{turn.receipt}:{seq}",
            )
            ids.append(block.id)
        turn.block_ids = ids
        if self.extractor is not None:
            spoken = [
                {"role": m["role"], "content": m["content"]}
                for m in turn.messages
                if m["role"] in ("user", "assistant") and m.get("content")
            ]
            extracted = self.extractor.extract_from_messages(spoken, memblock=self.memory)
            for derived_id in getattr(extracted, "block_ids", None) or []:
                self.memory.link(derived_id, ids[0], relation=EdgeRelation.DERIVED_FROM)

    def _outcome(self, turns: list[_Turn], more: list[str] | None = None) -> str:
        """Whether these writes reached memory: "applied", "absent" or "unknown"."""
        receipts = [r for turn in turns for r in turn.receipts()] + list(more or [])
        if not receipts:
            return "absent"
        try:
            statuses = {self.memory.receipt_status(receipt) for receipt in receipts}
        except Exception:
            return "unknown"
        if statuses == {ReceiptStatus.APPLIED}:
            return "applied"
        if statuses <= {ReceiptStatus.UNKNOWN}:
            return "absent"
        return "unknown"

    def _settle(self, turns: list[_Turn]) -> None:
        """Committed turns leave the prompt, except the short tail kept for context."""
        for turn in turns:
            self._recent.append(turn)
        self._turns = [t for t in self._turns if t not in turns]

    def _commit(
        self,
        answer: Any = None,
        extra: Callable[[], None] | None = None,
        extra_receipts: list[str] | None = None,
    ) -> int:
        turns = list(self._turns)
        if not turns and extra is None:
            return 0
        try:
            with self.memory.transaction():
                for turn in turns:
                    self._write_turn(turn)
                if extra is not None:
                    extra()
        except Exception as error:
            # Receipts say what actually happened, so most failures are not
            # left in doubt: either everything was saved, or nothing was.
            outcome = self._outcome(turns, extra_receipts)
            if outcome != "applied":
                self.uncertain = outcome == "unknown"
                self._pending = turns if self.uncertain else None
                self._pending_receipts = list(extra_receipts or []) if self.uncertain else []
                raise MemorySessionError(
                    f"Could not save to memory: {error}",
                    stage="commit",
                    uncertain=self.uncertain,
                    answer=answer,
                ) from error
        self._settle(turns)
        return len(turns)

    # ── Public API ───────────────────────────────────────────────────────

    def chat(
        self,
        query: str,
        *,
        enable_memory: bool = False,
        return_dict: bool = False,
    ) -> Any:
        """
        Send one message and return the reply.

        With `enable_memory=True`, once the reply completes, every turn not yet
        saved (this one included) is committed to memory and leaves the prompt.

        Returns the reply text, or a dict (`answer`, `finish_reason`,
        `requires_action`, `tool_calls`, `memory_updated`, `committed_turns`,
        `prompt`) when `return_dict=True` or the session has tools.
        """
        query = _text(query, "query")
        if type(enable_memory) is not bool:
            raise ValueError("enable_memory must be a boolean")
        with self._operation():
            self._check_no_tools()
            turn = _Turn(
                index=self._next_turn,
                messages=[{"role": "user", "content": query}],
                receipt=f"{self.id}:t{self._next_turn}:{uuid.uuid4().hex[:8]}",
                at=self.clock(),
                enable_memory=enable_memory,
            )
            self._pinned_experience = None
            self._injections = 0
            if self._trouble is not None:
                self._trouble.reset()
            if self.experience == "start":
                self._pin_experience(turn, query, "start", limit=3)
            reply = self._generate(turn)
            self._next_turn += 1
            return self._finish(turn, reply, return_dict)

    def submit_tool_results(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Continue the waiting turn with one result per tool call:
        `{"tool_call_id": ..., "content": ..., "is_error": optional bool}`.

        All tool rounds and the final answer stay one turn. With
        `experience="on_trouble"`, results that show the model is stuck
        (failures, empty results, the same call repeated) add relevant
        experience to the prompt for the rest of the task.
        """
        with self._operation():
            turn = self._tool_turn
            if turn is None:
                raise ValueError("No tool calls are waiting for results")
            by_id = {}
            for result in results:
                if not isinstance(result, dict) or "tool_call_id" not in result:
                    raise ValueError("Each result needs a tool_call_id")
                by_id[str(result["tool_call_id"])] = result
            wanted = [str(call["id"]) for call in turn.pending_calls]
            if sorted(by_id) != sorted(wanted):
                raise ValueError("Give exactly one result for each pending tool call")
            query = next((m["content"] for m in turn.messages if m["role"] == "user"), "")
            for call in turn.pending_calls:
                result = by_id[str(call["id"])]
                content = result.get("content")
                content = content if isinstance(content, str) else json.dumps(content, default=str)
                turn.messages.append({
                    "role": "tool",
                    "tool_call_id": str(call["id"]),
                    "name": call.get("name", ""),
                    "content": content,
                })
                if self._trouble is None:
                    continue
                reason = self._trouble.observe(
                    call.get("name", ""), call.get("arguments"), content, result.get("is_error"),
                )
                if reason and self.experience == "on_trouble":
                    self._pin_experience(turn, f"{query}\n{content[:500]}", reason, limit=1)
            turn.pending_calls = []
            reply = self._generate(turn)
            return self._finish(turn, reply, True)

    def commit_history(self) -> int:
        """Commit every turn not yet saved, cut-off ones included. Returns how many."""
        with self._operation():
            self._check_no_tools()
            return self._commit()

    def ingest(self, document: str, *, title: str | None = None, chunk_chars: int = 6000) -> int:
        """
        Put a document into memory word for word, in chunks, as one grouped
        write. Turns not yet saved are committed first. Returns the chunk count.
        """
        document = _text(document, "document")
        if chunk_chars < 200:
            raise ValueError("chunk_chars must be at least 200")
        chunks = split_document(document, chunk_chars)
        document_id = f"doc_{uuid.uuid4().hex[:12]}"
        receipts = [f"{self.id}:{document_id}:{seq}" for seq in range(len(chunks))]

        def write() -> None:
            for seq, chunk in enumerate(chunks):
                metadata = {"_document": document_id}
                if title:
                    metadata["_title"] = title
                self.memory.store_episode(
                    chunk,
                    role="document",
                    session_id=self.id,
                    seq=seq,
                    metadata=metadata,
                    receipt=receipts[seq],
                )

        with self._operation():
            self._check_no_tools()
            self._commit(extra=write, extra_receipts=receipts)
            return len(chunks)

    def save_state(self, path: str, *, overwrite: bool = False) -> dict[str, Any]:
        """
        Commit what is not yet saved, then write a snapshot: this session's
        blocks when `scope="session"`, otherwise the whole store. Returns the
        snapshot's stamp, with `path`.
        """
        with self._operation():
            self._check_no_tools()
            self._commit()
            return self.memory.snapshot(
                path,
                session_id=self.id if self.scope == "session" else None,
                overwrite=overwrite,
            )

    def load_state(self, path: str, *, embeddings: str = "strict") -> dict[str, Any]:
        """
        Add a snapshot's memory to this store and start with an empty
        conversation. The transcript is not part of a snapshot.
        """
        with self._operation():
            self._check_no_tools()
            result = self.memory.restore(path, mode="merge", embeddings=embeddings)
            self._turns, self._tool_turn = [], None
            self._recent.clear()
            self.memory_read = True
            return result

    def set_memory_read(self, enabled: bool) -> None:
        """Turn reading memory into the prompt on or off. Saving is separate."""
        if type(enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        with self._operation():
            self._check_no_tools()
            self.memory_read = enabled

    def recover(self, force: bool = False) -> str:
        """
        Settle a commit that did not finish cleanly. Returns "applied" (it was
        saved after all), "absent" (nothing was saved; the turns are still here
        and will be committed next time) or raises if it is still in doubt.
        """
        with self._lock:
            if self.closed:
                raise MemorySessionError("Session is closed")
            self.memory.recover_transactions(force=force)
            pending = self._pending or []
            outcome = self._outcome(pending, self._pending_receipts)
            if outcome == "unknown":
                raise MemorySessionError(
                    "The commit is still in progress elsewhere; try again once "
                    "its lease has expired, or pass force=True",
                    uncertain=True,
                )
            if outcome == "applied":
                self._settle(pending)
            self._pending, self._pending_receipts, self.uncertain = None, [], False
            return outcome

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "session": self.id,
                "closed": self.closed,
                "uncertain": self.uncertain,
                "memory_read": self.memory_read,
                "uncommitted_turns": len(self._turns),
                "pending_tool_calls": len(self._tool_turn.pending_calls) if self._tool_turn else 0,
                "next_turn": self._next_turn,
            }

    @property
    def messages(self) -> list[dict[str, Any]]:
        """The conversation still held in the prompt (a detached copy)."""
        with self._lock:
            turns = [*self._recent, *self._turns]
            if self._tool_turn is not None:
                turns.append(self._tool_turn)
            return deepcopy([m for t in turns for m in t.messages])

    def fork(self, session_id: str | None = None) -> "MemorySession":
        """
        Branch the conversation: a new session with the same settings and a
        copy of the turns still in the prompt. Memory is shared with this
        session; use `MemBlock.fork()` for an independent copy of the memory.
        """
        with self._operation():
            self._check_no_tools()
            branch = MemorySession(
                self.memory, self._raw_model,
                session_id=session_id, system_prompt=self.system_prompt, tools=self.tools,
                memory_read=self.memory_read, recall_limit=self.recall_limit,
                recall_budget=self.recall_budget, max_item_chars=self.max_item_chars,
                recent_turns=self.recent_turns, scope=self.scope,
                relevance_floor=self.relevance_floor, extractor=self.extractor,
                experience=self.experience, experience_bank=self._bank, trouble=None,
                max_experience_injections=self.max_experience_injections, clock=self.clock,
            )
            for turn in self._recent:
                branch._recent.append(turn)  # already saved; shared, read-only
            for turn in self._turns:
                copy = deepcopy(turn)
                copy.receipt = f"{branch.id}:t{copy.index}:{uuid.uuid4().hex[:8]}"
                branch._turns.append(copy)
            branch._next_turn = max(branch._next_turn, self._next_turn)
            return branch

    def reset(self) -> None:
        """Forget the conversation held here. Memory in the store is kept."""
        with self._lock:
            self._turns, self._tool_turn, self._pending = [], None, None
            self._pending_receipts = []
            self._recent.clear()
            self._pinned_experience, self._injections = None, 0
            self.uncertain = False

    def close(self) -> None:
        """Close the session. The `MemBlock` stays open."""
        with self._lock:
            self.closed = True

    def __enter__(self) -> "MemorySession":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"MemorySession(id={self.id!r}, uncommitted_turns={len(self._turns)}, "
            f"memory_read={self.memory_read})"
        )
