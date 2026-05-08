"""Conflict resolution — LLM-powered ADD/UPDATE/DELETE/NONE decisions.

When new information arrives, the conflict resolver compares it against
existing memories and uses an LLM to decide:
  - ADD: Store as a new memory (no conflict)
  - UPDATE: Update an existing memory with new information
  - DELETE: Remove an existing memory that's been contradicted
  - NONE: Skip — the information is already known or not worth storing

This mirrors Mem0's core "inference mode" but gives you full control.

Usage:
    from memblock.conflict import ConflictResolver

    resolver = ConflictResolver(provider=OpenAIProvider(api_key="sk-..."))
    actions = resolver.resolve("User now prefers TypeScript", existing_blocks)
    # actions = [ConflictAction(action="UPDATE", block_id="blk_abc", ...)]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from memblock.block import Block


class ConflictActionType(str, Enum):
    """Possible conflict resolution actions."""

    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NONE = "NONE"


@dataclass
class ConflictAction:
    """A single conflict resolution decision."""

    action: ConflictActionType
    block_id: str | None = None  # Existing block to UPDATE or DELETE
    new_content: str | None = None  # New content for ADD or UPDATE
    reason: str = ""  # LLM's reasoning


@dataclass
class ConflictResult:
    """Result of conflict resolution."""

    actions: list[ConflictAction] = field(default_factory=list)
    raw_response: str = ""
    errors: list[str] = field(default_factory=list)


CONFLICT_SYSTEM_PROMPT = """You are a memory conflict resolver. Given NEW information and EXISTING memories, decide what to do.

For each piece of new information, output a JSON array of actions:
- "action": "ADD" | "UPDATE" | "DELETE" | "NONE"
- "block_id": ID of the existing block to update/delete (null for ADD/NONE)
- "new_content": The content to store (for ADD) or the updated text (for UPDATE). Null for DELETE/NONE.
- "reason": Brief explanation of why this action was chosen

Rules:
1. ADD: New info that doesn't conflict with or duplicate anything existing
2. UPDATE: New info that supersedes or refines an existing memory — provide the UPDATED content
3. DELETE: An existing memory that is now incorrect or contradicted by the new info
4. NONE: The new info is already captured or too trivial to store
5. Be conservative — prefer NONE over ADD if the info is already known
6. When updating, merge the old and new info into a single coherent statement
7. Only DELETE when the new info clearly contradicts an existing memory

Respond with ONLY the JSON array, no other text."""

CONFLICT_USER_TEMPLATE = """NEW INFORMATION:
{new_content}

EXISTING MEMORIES:
{existing_blocks}

Decide what actions to take. Return a JSON array:"""


class ConflictResolver:
    """
    Resolves conflicts between new and existing memories using an LLM.

    Requires an LLM provider (from memblock.extraction).
    """

    def __init__(
        self,
        provider: Any,  # LLMProvider — Any to avoid circular import
        system_prompt: str | None = None,
    ) -> None:
        self.provider = provider
        self.system_prompt = system_prompt or CONFLICT_SYSTEM_PROMPT

    def resolve(
        self,
        new_content: str,
        existing_blocks: list[Block],
        new_block_type: Any = None,
    ) -> ConflictResult:
        """
        Resolve conflicts between new content and existing blocks.

        Args:
            new_content: The new information to evaluate
            existing_blocks: Top-K semantically similar existing blocks
            new_block_type: Optional `BlockType` (or its string value)
                of the block being stored. When provided, UPDATE/DELETE
                actions targeting blocks of a *different* type are
                rejected — preventing the resolver from silently
                merging a FACT write into an ENTITY block (or vice
                versa) just because their content overlaps textually.

        Returns:
            ConflictResult with a list of ConflictActions.
        """
        result = ConflictResult()

        if not existing_blocks:
            # No existing blocks — just ADD
            result.actions.append(ConflictAction(
                action=ConflictActionType.ADD,
                new_content=new_content,
                reason="No existing memories to conflict with",
            ))
            return result

        # Format existing blocks
        existing_text = "\n".join(
            f"- [ID: {b.id}] {b.content} (type: {b.type.value}, confidence: {b.metadata.confidence:.2f})"
            for b in existing_blocks
        )

        user_prompt = CONFLICT_USER_TEMPLATE.format(
            new_content=new_content,
            existing_blocks=existing_text,
        )

        try:
            raw_response = self.provider.complete(self.system_prompt, user_prompt)
            result.raw_response = raw_response
        except Exception as e:
            result.errors.append(f"LLM call failed: {e}")
            # Fall back to ADD on LLM failure
            result.actions.append(ConflictAction(
                action=ConflictActionType.ADD,
                new_content=new_content,
                reason="Fallback: LLM conflict resolution failed",
            ))
            return result

        try:
            actions = self._parse_response(
                raw_response, existing_blocks,
                new_block_type=new_block_type,
            )
            result.actions = actions
        except Exception as e:
            result.errors.append(f"Parse failed: {e}")
            result.actions.append(ConflictAction(
                action=ConflictActionType.ADD,
                new_content=new_content,
                reason="Fallback: could not parse LLM response",
            ))

        return result

    def _parse_response(
        self,
        raw: str,
        existing_blocks: list[Block],
        new_block_type: Any = None,
    ) -> list[ConflictAction]:
        """Parse the LLM response into ConflictActions.

        When `new_block_type` is provided, UPDATE/DELETE actions
        targeting blocks of a different type are silently dropped —
        the resolver can only mutate same-type matches. This is the
        Phase-fix for the cross-type conflation bug.
        """
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            cleaned = "\n".join(lines)

        parsed = json.loads(cleaned)

        if isinstance(parsed, dict):
            for key in ("actions", "results", "decisions"):
                if key in parsed and isinstance(parsed[key], list):
                    parsed = parsed[key]
                    break
            else:
                parsed = [parsed]

        if not isinstance(parsed, list):
            return []

        # Build lookup tables: id -> block (for type checks) and the
        # set of valid ids the resolver is allowed to reference.
        by_id = {b.id: b for b in existing_blocks}
        valid_ids = set(by_id.keys())

        # Normalize new_block_type to a string (BlockType.value) for
        # comparison with the candidate block's type.
        target_type_str: str | None = None
        if new_block_type is not None:
            target_type_str = (
                new_block_type.value
                if hasattr(new_block_type, "value")
                else str(new_block_type)
            )

        actions: list[ConflictAction] = []

        for item in parsed:
            action_type = ConflictActionType(item.get("action", "NONE").upper())
            block_id = item.get("block_id")

            # Validate block_id exists for UPDATE/DELETE
            if action_type in (ConflictActionType.UPDATE, ConflictActionType.DELETE):
                if block_id not in valid_ids:
                    continue  # Skip invalid references

                # Type-scoped guard — reject cross-type UPDATE/DELETE.
                # The resolver's LLM gets candidates regardless of
                # type (semantic similarity is type-agnostic), so a
                # FACT write can be told to UPDATE an ENTITY block.
                # That's never correct; demote to ADD by skipping.
                if target_type_str is not None:
                    target_block = by_id.get(block_id)
                    target_type = (
                        target_block.type.value
                        if target_block is not None
                        and hasattr(target_block.type, "value")
                        else None
                    )
                    if target_type and target_type != target_type_str:
                        continue  # Cross-type — drop this action

            actions.append(ConflictAction(
                action=action_type,
                block_id=block_id,
                new_content=item.get("new_content"),
                reason=item.get("reason", ""),
            ))

        return actions


class AsyncConflictResolver(ConflictResolver):
    """Async equivalent of `ConflictResolver`.

    The provider's `complete()` method is sync (HTTP call); we wrap
    it with `asyncio.to_thread` so the calling event loop stays
    responsive while the LLM request is in flight. All parsing /
    formatting logic is reused via inheritance — only the I/O entry
    point (`resolve` -> `aresolve`) differs.

    Used by `AsyncMemBlock`'s native-async `store()` path when
    `conflict_resolution=True` is configured. Behaves identically to
    the sync version: same prompts, same fallback behaviour on LLM
    or parse failure.
    """

    async def aresolve(
        self,
        new_content: str,
        existing_blocks: list[Block],
        new_block_type: Any = None,
    ) -> ConflictResult:
        """Async equivalent of `resolve()`. Returns the same
        `ConflictResult` shape.

        `new_block_type` enables the type-scoped guard — see the
        sync `resolve()` docstring for details.

        Fallback semantics (LLM call fails OR response unparseable)
        match the sync resolver: produce a single `ADD` action so
        the caller's `store()` path falls through to a normal write.
        """
        import asyncio

        result = ConflictResult()

        if not existing_blocks:
            result.actions.append(ConflictAction(
                action=ConflictActionType.ADD,
                new_content=new_content,
                reason="No existing memories to conflict with",
            ))
            return result

        existing_text = "\n".join(
            f"- [ID: {b.id}] {b.content} (type: {b.type.value}, "
            f"confidence: {b.metadata.confidence:.2f})"
            for b in existing_blocks
        )

        user_prompt = CONFLICT_USER_TEMPLATE.format(
            new_content=new_content,
            existing_blocks=existing_text,
        )

        try:
            # Wrap the sync HTTP call so it runs off the event loop.
            raw_response = await asyncio.to_thread(
                self.provider.complete, self.system_prompt, user_prompt,
            )
            result.raw_response = raw_response
        except Exception as e:
            result.errors.append(f"LLM call failed: {e}")
            result.actions.append(ConflictAction(
                action=ConflictActionType.ADD,
                new_content=new_content,
                reason="Fallback: LLM conflict resolution failed",
            ))
            return result

        try:
            actions = self._parse_response(
                raw_response, existing_blocks,
                new_block_type=new_block_type,
            )
            result.actions = actions
        except Exception as e:
            result.errors.append(f"Parse failed: {e}")
            result.actions.append(ConflictAction(
                action=ConflictActionType.ADD,
                new_content=new_content,
                reason="Fallback: could not parse LLM response",
            ))

        return result
