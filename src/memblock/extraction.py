"""
LLM-powered auto-extraction — extract structured memory blocks from conversations.

Supports multiple LLM providers:
- OpenAI (GPT-4, GPT-3.5, etc.)
- Anthropic (Claude)
- Google Gemini (Gemini 2.0 Flash, etc.)
- Custom providers via callable

Usage:
    from memblock import MemBlock
    from memblock.extraction import LLMExtractor, OpenAIProvider, AnthropicProvider

    # OpenAI
    extractor = LLMExtractor(
        provider=OpenAIProvider(api_key="sk-...", model="gpt-4o-mini")
    )

    # Anthropic
    extractor = LLMExtractor(
        provider=AnthropicProvider(api_key="sk-ant-...", model="claude-sonnet-4-20250514")
    )

    # Attach to MemBlock
    mem = MemBlock(storage="sqlite:///./memory.db")
    blocks = extractor.extract(
        conversation="User: I love Python and deploy on AWS",
        memblock=mem,
    )
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from memblock.types import BlockType, SourceType


# ─── Extraction Prompt ────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction engine. Given a conversation between a user and an AI assistant, extract structured memory blocks.

For each piece of knowledge found, output a JSON array of objects with these fields:
- "content": the memory text (concise, factual, self-contained — include the subject so the memory is useful standalone)
- "type": one of "fact", "preference", "event", "entity", "relation"
- "confidence": float 0.0-1.0 (how certain is this information)
- "source": one of "explicit" (user directly stated), "inferred" (derived from context), "observed" (noticed from behavior)
- "tags": list of relevant tags (1-3 short words)
- "happened_at": ISO date/datetime when the event/fact occurred (if mentioned or inferable), null otherwise. For events, always try to extract the date. Use the conversation context to infer approximate dates. IMPORTANT: Always resolve relative dates ("last year", "last week", "yesterday", "two months ago") to absolute ISO dates using the conversation date as reference. Never store relative time references — always compute the actual date.
- "temporal_precision": one of "exact", "day", "week", "month", "year", "approximate" — how precise the temporal information is
- "subject": the main entity this memory is about (person name, place, organization, etc.), null if not applicable
- "relations": list of objects {"target_content": "...", "relation": "supports|contradicts|about|related_to"} (optional connections between extracted memories)

Extraction guidelines by type:
- "fact": Extract as subject-predicate-object triples when possible. E.g., "Alice works at Google as a senior engineer" not "works at Google".
- "preference": Include who holds the preference and what specifically they prefer. E.g., "User prefers Python over JavaScript for backend work".
- "event": ALWAYS include when it happened (happened_at). Include who was involved and what occurred. E.g., "User deployed v2.0 to production on March 10, 2024".
- "entity": Extract the entity's full name and key attributes. E.g., "Project Alpha — internal code name for the new authentication system".
- "relation": Describe the relationship between two specific entities. E.g., "Alice mentors Bob on the backend team".

Rules:
1. Extract ONLY factual, useful knowledge — not small talk or filler
2. Be concise but SELF-CONTAINED — each content should be understandable without the original conversation
3. Set confidence based on how explicitly the information was stated
4. Use "inferred" source for things derived from context, not directly said
5. If nothing worth remembering, return an empty array []
6. DO NOT extract information that is too vague or temporary
7. For events, ALWAYS attempt to extract or infer a date (happened_at)
8. Include the subject's name in fact and preference content so the memory is findable via search
9. Extract separate blocks for separate facts — don't combine unrelated information
10. When the content references a time, include the ABSOLUTE date in the content text itself (e.g., "Melanie painted a sunrise in 2022" not "Melanie painted a sunrise last year")

Respond with ONLY the JSON array, no other text."""

EXTRACTION_USER_TEMPLATE = """Extract memory blocks from this conversation:

---
{conversation}
---

Return a JSON array of memory blocks:"""


# ─── Provider Interface ───────────────────────────────────────────────────────


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt to the LLM and return the response text."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        ...


# ─── OpenAI Provider ─────────────────────────────────────────────────────────


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible provider (works with GPT-4, GPT-3.5, etc.)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        temperature: float = 0.1,
    ) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI provider requires the openai package. "
                "Install with: pip install memblock[llm]"
            )

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        self._client = openai.OpenAI(**kwargs)
        self._model = model
        self._temperature = temperature

    @property
    def name(self) -> str:
        return f"openai/{self._model}"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self._temperature,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or "[]"


# ─── Anthropic Provider ──────────────────────────────────────────────────────


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic provider requires the anthropic package. "
                "Install with: pip install memblock[llm]"
            )

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    @property
    def name(self) -> str:
        return f"anthropic/{self._model}"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
            temperature=self._temperature,
        )
        return response.content[0].text


# ─── Gemini Provider ────────────────────────────────────────────────────────


class GeminiProvider(LLMProvider):
    """Google Gemini provider (Gemini 2.0 Flash, etc.)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.1,
    ) -> None:
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "Gemini provider requires the google-genai package. "
                "Install with: pip install memblock[llm-gemini]"
            )

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._temperature = temperature

    @property
    def name(self) -> str:
        return f"gemini/{self._model}"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self._model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self._temperature,
                response_mime_type="application/json",
            ),
        )
        return response.text or "[]"


# ─── Custom Provider ─────────────────────────────────────────────────────────


class CallableProvider(LLMProvider):
    """
    Custom provider using any callable.

    Usage:
        def my_llm(system: str, user: str) -> str:
            # call your custom LLM
            return response_text

        provider = CallableProvider(fn=my_llm, name="my-llm")
    """

    def __init__(self, fn: Callable[[str, str], str], provider_name: str = "custom") -> None:
        self._fn = fn
        self._name = provider_name

    @property
    def name(self) -> str:
        return self._name

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._fn(system_prompt, user_prompt)


# ─── Extraction Result ───────────────────────────────────────────────────────


@dataclass
class ExtractionResult:
    """Result of an extraction operation."""

    blocks_created: int = 0
    block_ids: list[str] = field(default_factory=list)
    raw_response: str = ""
    provider: str = ""
    errors: list[str] = field(default_factory=list)


# ─── LLM Extractor ───────────────────────────────────────────────────────────


class LLMExtractor:
    """
    Extracts structured memory blocks from conversations using an LLM.

    The extractor sends the conversation to the configured LLM provider,
    parses the response into memory blocks, and stores them in MemBlock.
    """

    def __init__(
        self,
        provider: LLMProvider,
        system_prompt: str | None = None,
        min_confidence: float = 0.3,
    ) -> None:
        """
        Initialize the extractor.

        Args:
            provider: LLM provider (OpenAI, Anthropic, or custom)
            system_prompt: Custom extraction prompt (uses default if None)
            min_confidence: Minimum confidence to store a block (filters low-quality extractions)
        """
        self.provider = provider
        self.system_prompt = system_prompt or EXTRACTION_SYSTEM_PROMPT
        self.min_confidence = min_confidence

    def extract(
        self,
        conversation: str,
        memblock: Any = None,  # MemBlock instance — Any to avoid circular import
        auto_store: bool = True,
        auto_link: bool = True,
    ) -> ExtractionResult:
        """
        Extract memory blocks from a conversation.

        Args:
            conversation: The conversation text (user + assistant messages)
            memblock: MemBlock instance to store extracted blocks into
            auto_store: If True, automatically stores blocks in memblock
            auto_link: If True, automatically creates edges between related blocks

        Returns:
            ExtractionResult with created block IDs and metadata.
        """
        result = ExtractionResult(provider=self.provider.name)

        # Send to LLM
        user_prompt = EXTRACTION_USER_TEMPLATE.format(conversation=conversation)

        try:
            raw_response = self.provider.complete(self.system_prompt, user_prompt)
            result.raw_response = raw_response
        except Exception as e:
            result.errors.append(f"LLM call failed: {e}")
            return result

        # Parse response
        try:
            extracted = self._parse_response(raw_response)
        except Exception as e:
            result.errors.append(f"Parse failed: {e}")
            return result

        if not extracted or not memblock:
            return result

        # Store blocks
        if auto_store:
            block_map: dict[str, str] = {}  # content -> block_id

            for item in extracted:
                confidence = item.get("confidence", 0.5)
                if confidence < self.min_confidence:
                    continue

                try:
                    block_type = BlockType(item.get("type", "fact"))
                    source = SourceType(item.get("source", "inferred"))

                    # Parse temporal data from extraction
                    happened_at = None
                    happened_at_end = None
                    temporal_precision = item.get("temporal_precision", "exact")
                    raw_happened = item.get("happened_at")
                    if raw_happened:
                        try:
                            from datetime import datetime
                            happened_at = datetime.fromisoformat(raw_happened)
                        except (ValueError, TypeError):
                            pass  # skip invalid dates

                    store_kwargs: dict[str, Any] = dict(
                        content=item["content"],
                        type=block_type,
                        confidence=confidence,
                        source=source,
                        tags=item.get("tags", []),
                    )
                    if happened_at is not None:
                        store_kwargs["happened_at"] = happened_at
                        store_kwargs["temporal_precision"] = temporal_precision
                    if happened_at_end is not None:
                        store_kwargs["happened_at_end"] = happened_at_end

                    block = memblock.store(**store_kwargs)

                    result.block_ids.append(block.id)
                    result.blocks_created += 1
                    block_map[item["content"]] = block.id

                except Exception as e:
                    result.errors.append(f"Store failed for '{item.get('content', '?')}': {e}")

            # Create edges between related blocks
            if auto_link:
                for item in extracted:
                    source_id = block_map.get(item.get("content", ""))
                    if not source_id:
                        continue

                    for rel in item.get("relations", []):
                        target_id = block_map.get(rel.get("target_content", ""))
                        if target_id:
                            try:
                                relation = rel.get("relation", "related_to")
                                memblock.link(source_id, target_id, relation=relation)
                            except Exception:
                                pass  # edge validation may fail, that's ok

        return result

    def extract_from_messages(
        self,
        messages: list[dict[str, str]],
        memblock: Any = None,
        auto_store: bool = True,
    ) -> ExtractionResult:
        """
        Extract from a list of message dicts (like your conversations table).

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            memblock: MemBlock instance
            auto_store: Auto-store extracted blocks

        Returns:
            ExtractionResult
        """
        # Format messages into conversation text
        lines = []
        for msg in messages:
            role = msg.get("role", "user").title()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")

        conversation = "\n".join(lines)
        return self.extract(conversation, memblock, auto_store)

    def _parse_response(self, raw: str) -> list[dict]:
        """Parse LLM response into list of memory block dicts."""
        # Clean up response — handle markdown code blocks
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Remove markdown code fences
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        # Try to parse as JSON
        parsed = json.loads(cleaned)

        # Handle both array and object-with-array responses
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            # Some models wrap in {"blocks": [...]} or {"memories": [...]}
            for key in ("blocks", "memories", "results", "data"):
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
            return [parsed]  # single block

        return []
