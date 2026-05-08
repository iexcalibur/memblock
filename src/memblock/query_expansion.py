"""Rule-based query expansion for chat-driven retrieval.

Many chat follow-up questions are referential — "tell me more",
"compare them", "what about that fund". On their own, they have
almost no semantic signal: the search query is just stop words plus
a pronoun. Memory retrieval misses the relevant evidence.

This module rewrites such referential queries using **the most recent
turn's content as context**, before the query hits memblock retrieval.
It's pure regex + last-turn substitution — no LLM call, no extra
cost, no extra latency.

Usage:
    from memblock.query_expansion import expand_query

    expanded = expand_query(
        query="tell me more",
        recent_messages=[
            {"role": "user", "content": "what's a balanced fund?"},
            {"role": "assistant", "content": "A balanced fund holds..."},
        ],
    )
    # → "tell me more about balanced fund"

The expansion is intentionally conservative: when no clear referent
is detected the original query is returned unchanged so we never
*degrade* a well-formed query.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping


# Patterns that indicate a referential query — the user is asking
# about something already in context, not a fresh topic.
#
# Each pattern is paired with a "stitch template" that produces an
# expanded query. The literal `{topic}` placeholder is filled with
# whatever topic phrase we extract from the recent turns.
_REFERENTIAL_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    # "tell me more" / "tell me more about it"
    (re.compile(r"^\s*tell me more(?:\s+about\s+(?:it|that|this|them))?\s*\??\s*$", re.I),
     "tell me more about {topic}"),
    # "more details" / "give me details"
    (re.compile(r"^\s*(?:give me\s+)?more details?\s*\??\s*$", re.I),
     "more details about {topic}"),
    # "what about it/that/this/them"
    (re.compile(r"^\s*what about (?:it|that|this|them)\s*\??\s*$", re.I),
     "what about {topic}"),
    # "compare them" / "compare those"
    (re.compile(r"^\s*compare (?:them|those|these)\s*\??\s*$", re.I),
     "compare {topic}"),
    # "explain"  /  "explain it"
    (re.compile(r"^\s*explain(?:\s+(?:it|that|this))?\s*\??\s*$", re.I),
     "explain {topic}"),
    # bare follow-ups: "and?" / "go on" / "continue"
    (re.compile(r"^\s*(?:and|go on|continue|so)\??\s*$", re.I),
     "more about {topic}"),
    # "how does it work"
    (re.compile(r"^\s*how does (?:it|that|this) work\s*\??\s*$", re.I),
     "how does {topic} work"),
    # "why" / "why is that" — open-ended but referential
    (re.compile(r"^\s*why(?:\s+(?:is\s+)?that)?\s*\??\s*$", re.I),
     "why {topic}"),
)


# Words we don't want as the extracted "topic" — too generic to anchor
# retrieval. If the only candidate noun phrase is one of these, fall
# back to "no expansion".
_TOPIC_STOPWORDS = frozenset({
    "the", "a", "an", "this", "that", "these", "those",
    "it", "they", "them", "i", "you", "we",
    "is", "are", "was", "were", "be", "been",
    "yes", "no", "ok", "okay", "sure", "thanks",
    "well", "so", "just", "really",
})


# Match capitalized noun phrases (likely entity names) — preferred.
_CAPITALIZED_NP_RE = re.compile(
    r"\b[A-Z][a-zA-Z']+(?:\s+[A-Z][a-zA-Z']+)*"
)
# Match longer lowercase noun phrases as fallback.
_NOUN_PHRASE_RE = re.compile(r"\b[a-zA-Z]{4,}(?:\s+[a-zA-Z]{4,}){0,3}\b")


def _is_referential(query: str) -> tuple[re.Pattern, str] | None:
    """Detect whether `query` matches any referential pattern.
    Returns (pattern, template) on match, else None."""
    for pat, template in _REFERENTIAL_PATTERNS:
        if pat.match(query or ""):
            return pat, template
    return None


def _extract_topic_from(text: str) -> str | None:
    """Pull the strongest topic candidate from `text`.

    Strategy:
      1. Prefer the longest capitalized noun phrase (entity-like).
      2. Fall back to the longest lowercase noun phrase.
      3. Reject candidates that are pure stopwords.
    """
    if not text:
        return None

    # 1. Capitalized phrases (proper nouns, likely entities)
    cap_matches = _CAPITALIZED_NP_RE.findall(text)
    cap_matches = [m for m in cap_matches if m.lower() not in _TOPIC_STOPWORDS]
    if cap_matches:
        return max(cap_matches, key=len).strip()

    # 2. Longest lowercase phrase (4+ chars per word)
    np_matches = _NOUN_PHRASE_RE.findall(text)
    np_matches = [
        m.strip() for m in np_matches
        if all(w.lower() not in _TOPIC_STOPWORDS for w in m.split())
    ]
    if np_matches:
        return max(np_matches, key=len)

    return None


def expand_query(
    query: str,
    recent_messages: Iterable[Mapping[str, str]],
    *,
    max_lookback: int = 3,
) -> str:
    """Return an expanded form of `query` using `recent_messages` as
    context, or the original query when no expansion fires.

    Args:
        query: The current user query.
        recent_messages: Iterable of message dicts each with
            `{"role", "content"}` keys, oldest-first or newest-first
            (we scan from the end). Typically the result of
            `MemBlock.get_session_history(...)` or a Qonfido-style
            short-term context list.
        max_lookback: How many recent turns to scan for a topic.

    Returns:
        Expanded query (or original if no expansion possible).
    """
    if not query or not query.strip():
        return query

    match = _is_referential(query)
    if match is None:
        return query  # not a referential pattern; return unchanged

    _, template = match

    # Scan recent_messages from newest to oldest. We deliberately
    # ignore the user's own current turn — we want CONTEXT, not the
    # query itself.
    msgs = list(recent_messages)
    # Most callers pass chronological (oldest-first); take the tail.
    tail = msgs[-max_lookback:] if max_lookback > 0 else msgs

    topic: str | None = None
    for msg in reversed(tail):
        role = msg.get("role") if isinstance(msg, Mapping) else None
        content = msg.get("content") if isinstance(msg, Mapping) else None
        if not content:
            continue
        # Prefer assistant turns (richer entity content) but fall
        # back to user turns if assistant says nothing helpful.
        topic = _extract_topic_from(str(content))
        if topic:
            break

    if not topic:
        return query  # no extractable topic → don't degrade the query

    return template.replace("{topic}", topic)
