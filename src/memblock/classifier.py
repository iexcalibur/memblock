"""Pluggable memory classifier — the contract MemBlock(classifier=...) calls on the write path.

A classifier decides the *fields* of a memory (type, source, confidence, encryption level,
decay rate, temporal precision) and, optionally, the conflict action against similar existing
blocks. It does not extract sentences from conversations; that stays with `LLMExtractor`.

The contract is duck-typed so implementations need not import memblock:

    class MyClassifier:
        def classify(self, content: str) -> dict | None:
            # {"fields": {"type": "preference", "confidence": 0.93, ...},
            #  "skip": False, "escalate": False, "record": {...audit...}}
            ...
        def resolve_conflict(self, new_content: str, existing_blocks: list[Block]) -> list[dict] | None:
            # [{"block_id": ..., "action": "ADD|UPDATE|RETRACT|DELETE|NONE",
            #   "confidence": 0.8, "escalate": False, "record": {...}}]
            ...

`fields` may contain any `store()` keyword. Values are applied only where the caller left the
argument unset, so explicit arguments always win. `escalate=True` asks MemBlock to confirm the
decision with the configured LLM provider (when one exists) before storing. `record` is written
into the block's custom metadata under `_classifier` so every decision is auditable and can
later be used as training data.

`jevblock.memblock_bridge.MemoryClassifier` is the reference implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


CLASSIFIER_METADATA_KEY = "_classifier"

# store() keywords a classifier may set.
CLASSIFIER_FIELDS = {"type", "source", "confidence", "encryption_level", "decay_rate", "temporal_precision", "tags"}


@runtime_checkable
class MemoryClassifier(Protocol):
    def classify(self, content: str) -> dict[str, Any] | None: ...


@dataclass
class Classification:
    fields: dict[str, Any] = field(default_factory=dict)
    skip: bool = False
    escalate: bool = False
    record: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_result(cls, result: dict[str, Any] | None) -> "Classification | None":
        if result is None:
            return None
        return cls(fields=dict(result.get("fields") or {}), skip=bool(result.get("skip", False)),
                   escalate=bool(result.get("escalate", False)), record=dict(result.get("record") or {}))


# ─── LLM escalation ─────────────────────────────────────────────────────────

ESCALATION_SYSTEM_PROMPT = """You classify one memory sentence for a memory store. Return ONLY a JSON object:
{"type": "fact|preference|event|entity|relation",
 "source": "explicit|inferred|observed",
 "confidence": 0.0-1.0,
 "sensitive": true|false}
type: fact = objective statement; preference = likes/dislikes/wants; event = something that happened at a time;
entity = a named thing (person, project, tool, company); relation = how two named things relate.
source: explicit = stated directly; inferred = derived from context; observed = noticed from behaviour.
sensitive: credentials, financial, medical, identity or other private data."""

ESCALATION_USER_TEMPLATE = "Memory: {content}\n\nJSON:"


def llm_classify(provider: Any, content: str) -> dict[str, Any]:
    """Ask an `LLMProvider` (memblock.extraction) to classify one sentence. Returns store() fields."""
    raw = provider.complete(ESCALATION_SYSTEM_PROMPT, ESCALATION_USER_TEMPLATE.format(content=content))
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    data = json.loads(text)
    out: dict[str, Any] = {}
    if data.get("type") in ("fact", "preference", "event", "entity", "relation"):
        out["type"] = data["type"]
    if data.get("source") in ("explicit", "inferred", "observed"):
        out["source"] = data["source"]
    try:
        c = float(data.get("confidence", 1.0))
        out["confidence"] = max(0.0, min(1.0, c))
    except (TypeError, ValueError):
        pass
    if data.get("sensitive") is True:
        out["encryption_level"] = "sensitive"
    return out
