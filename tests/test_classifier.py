"""Pluggable classifier hook on MemBlock.store() and the conflict gate."""

from __future__ import annotations

import json

from memblock import BlockType, EncryptionLevel, MemBlock, SourceType
from memblock.classifier import CLASSIFIER_METADATA_KEY, llm_classify


class FakeClassifier:
    name = "fake"

    def __init__(self, fields=None, skip=False, escalate=False, conflict=None):
        self.fields, self.skip, self.escalate, self.conflict = fields or {}, skip, escalate, conflict
        self.calls = []

    def classify(self, content):
        self.calls.append(content)
        return {"fields": dict(self.fields), "skip": self.skip, "escalate": self.escalate, "record": {"why": "test"}}

    def resolve_conflict(self, new_content, existing_blocks):
        if self.conflict is None:
            return None
        return [dict(self.conflict, block_id=b.id) for b in existing_blocks]


def test_classifier_fills_unset_fields_and_records_audit():
    clf = FakeClassifier(fields={"type": "preference", "source": "inferred", "confidence": 0.81,
                                 "encryption_level": "sensitive", "decay_rate": 0.02, "jevblock_sensitivity": "financial"})
    mem = MemBlock(classifier=clf)
    b = mem.store("User prefers Python")
    assert b.type == BlockType.PREFERENCE
    assert b.metadata.source == SourceType.INFERRED
    assert b.metadata.confidence == 0.81
    assert b.encryption_level == EncryptionLevel.SENSITIVE
    assert b.metadata.decay_rate == 0.02
    rec = b.metadata.custom_metadata[CLASSIFIER_METADATA_KEY]
    assert rec["classifier"] == "fake" and rec["why"] == "test"
    assert rec["applied"]["type"] == "preference"
    assert rec["extra"] == {"jevblock_sensitivity": "financial"}
    assert clf.calls == ["User prefers Python"]


def test_explicit_arguments_win_over_classifier():
    clf = FakeClassifier(fields={"type": "preference", "confidence": 0.2})
    mem = MemBlock(classifier=clf)
    b = mem.store("User works at Acme", type=BlockType.FACT, confidence=0.95)
    assert b.type == BlockType.FACT and b.metadata.confidence == 0.95
    assert "type" not in b.metadata.custom_metadata[CLASSIFIER_METADATA_KEY]["applied"]


def test_defaults_unchanged_without_classifier():
    mem = MemBlock()
    b = mem.store("Plain")
    assert b.type == BlockType.FACT and b.metadata.confidence == 1.0
    assert b.metadata.source == SourceType.EXPLICIT and b.encryption_level == EncryptionLevel.NONE
    assert b.metadata.custom_metadata is None


def test_classifier_skip_drops_the_memory():
    mem = MemBlock(classifier=FakeClassifier(skip=True))
    assert mem.store("hello thanks") is None
    assert mem.query(limit=10) == []


def test_classifier_failure_never_breaks_store():
    class Broken:
        def classify(self, content):
            raise RuntimeError("boom")
    b = MemBlock(classifier=Broken()).store("still stored")
    assert b is not None and b.type == BlockType.FACT


def test_escalation_uses_llm_provider_when_configured():
    seen = {}

    def provider(system_prompt, user_prompt):
        seen["prompt"] = user_prompt
        return json.dumps({"type": "event", "source": "explicit", "confidence": 0.99, "sensitive": True})

    clf = FakeClassifier(fields={"type": "fact", "confidence": 0.4}, escalate=True)
    mem = MemBlock(classifier=clf, extract_provider=provider)
    b = mem.store("Deployed v2 on March 10")
    assert b.type == BlockType.EVENT and b.metadata.confidence == 0.99
    assert b.encryption_level == EncryptionLevel.SENSITIVE
    assert "Deployed v2" in seen["prompt"]
    rec = b.metadata.custom_metadata[CLASSIFIER_METADATA_KEY]
    assert rec["escalated"] is True and rec["llm"]["type"] == "event"


def test_escalation_can_be_disabled():
    provider = lambda s, u: json.dumps({"type": "event"})
    clf = FakeClassifier(fields={"type": "fact"}, escalate=True)
    b = MemBlock(classifier=clf, extract_provider=provider, classifier_escalate=False).store("x y z")
    assert b.type == BlockType.FACT
    assert "llm" not in b.metadata.custom_metadata[CLASSIFIER_METADATA_KEY]


def test_llm_classify_parses_fenced_json():
    class P:
        def complete(self, s, u):
            return "```json\n{\"type\": \"relation\", \"confidence\": 2}\n```"
    out = llm_classify(P(), "Alice mentors Bob")
    assert out == {"type": "relation", "confidence": 1.0}


def test_conflict_gate_classifier_retracts_without_llm():
    clf = FakeClassifier(fields={"type": "event"},
                         conflict={"action": "RETRACT", "confidence": 0.9, "escalate": False})
    # No embeddings and no LLM provider: only the classifier path can resolve conflicts.
    mem = MemBlock(classifier=clf, conflict_resolution=True)
    old = mem.store("User holds a Vanguard position", type=BlockType.FACT)
    new = mem.store("User sold the Vanguard position")
    assert mem.get(old.id) is None or mem.get(old.id).deleted
    rec = new.metadata.custom_metadata[CLASSIFIER_METADATA_KEY]
    assert rec["conflict_decided_by"] == "classifier"
    assert rec["conflict"][0]["action"] == "RETRACT"


def test_conflict_gate_none_skips_store():
    clf = FakeClassifier(conflict={"action": "NONE", "confidence": 0.95, "escalate": False})
    mem = MemBlock(classifier=clf, conflict_resolution=True)
    mem.store("User lives in Pune")
    assert mem.store("User lives in Pune city") is None


def test_conflict_gate_escalated_without_llm_falls_through_to_add():
    clf = FakeClassifier(conflict={"action": "DELETE", "confidence": 0.3, "escalate": True})
    mem = MemBlock(classifier=clf, conflict_resolution=True)
    a = mem.store("User lives in Pune")
    b = mem.store("User lives in Mumbai")
    assert b is not None and mem.get(a.id) is not None   # nothing destructive happened
    assert "conflict_decided_by" not in b.metadata.custom_metadata[CLASSIFIER_METADATA_KEY]
