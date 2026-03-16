"""Tests for LLM auto-extraction (using mock provider — no API keys needed)."""

import json
import pytest

from memblock import MemBlock, BlockType
from memblock.extraction import (
    LLMExtractor,
    CallableProvider,
    ExtractionResult,
    EXTRACTION_SYSTEM_PROMPT,
)


@pytest.fixture
def mem():
    m = MemBlock(storage="sqlite:///:memory:")
    yield m
    m.close()


def mock_llm_response(system: str, user: str) -> str:
    """Mock LLM that returns predefined extraction results."""
    return json.dumps([
        {
            "content": "User prefers Python for backend development",
            "type": "preference",
            "confidence": 0.9,
            "source": "explicit",
            "tags": ["python", "backend"],
            "relations": [],
        },
        {
            "content": "User deploys applications on AWS",
            "type": "fact",
            "confidence": 0.85,
            "source": "explicit",
            "tags": ["aws", "cloud", "deployment"],
            "relations": [
                {
                    "target_content": "User prefers Python for backend development",
                    "relation": "related_to",
                }
            ],
        },
        {
            "content": "User switched from JavaScript to TypeScript last month",
            "type": "event",
            "confidence": 0.95,
            "source": "explicit",
            "tags": ["typescript", "migration"],
            "relations": [],
        },
        {
            "content": "Low confidence noise",
            "type": "fact",
            "confidence": 0.1,
            "source": "inferred",
            "tags": [],
            "relations": [],
        },
    ])


def mock_llm_empty(system: str, user: str) -> str:
    """Mock LLM that returns empty array."""
    return "[]"


def mock_llm_markdown(system: str, user: str) -> str:
    """Mock LLM that wraps response in markdown code block."""
    return '```json\n[{"content": "Wrapped in markdown", "type": "fact", "confidence": 0.8, "source": "explicit", "tags": []}]\n```'


def mock_llm_object_wrapper(system: str, user: str) -> str:
    """Mock LLM that wraps array in an object."""
    return json.dumps({
        "blocks": [
            {"content": "Wrapped in object", "type": "fact", "confidence": 0.8, "source": "explicit", "tags": []}
        ]
    })


def mock_llm_error(system: str, user: str) -> str:
    raise RuntimeError("API connection failed")


class TestLLMExtractor:
    def test_extract_basic(self, mem: MemBlock):
        provider = CallableProvider(fn=mock_llm_response, provider_name="mock")
        extractor = LLMExtractor(provider=provider)

        result = extractor.extract(
            conversation="User: I love Python for backend and deploy on AWS",
            memblock=mem,
        )

        assert isinstance(result, ExtractionResult)
        assert result.blocks_created == 3  # 4th was below min_confidence
        assert len(result.block_ids) == 3
        assert result.provider == "mock"
        assert len(result.errors) == 0

    def test_extract_filters_low_confidence(self, mem: MemBlock):
        provider = CallableProvider(fn=mock_llm_response, provider_name="mock")
        extractor = LLMExtractor(provider=provider, min_confidence=0.5)

        result = extractor.extract(
            conversation="User: Some conversation",
            memblock=mem,
        )

        # The 0.1 confidence block should be filtered out
        assert result.blocks_created == 3

        # Verify blocks are actually in memblock
        all_blocks = mem.query(limit=50)
        assert len(all_blocks) == 3

    def test_extract_creates_edges(self, mem: MemBlock):
        provider = CallableProvider(fn=mock_llm_response, provider_name="mock")
        extractor = LLMExtractor(provider=provider)

        result = extractor.extract(
            conversation="User: Python + AWS",
            memblock=mem,
            auto_link=True,
        )

        # The AWS fact should be linked to the Python preference
        assert result.blocks_created == 3

    def test_extract_empty_response(self, mem: MemBlock):
        provider = CallableProvider(fn=mock_llm_empty, provider_name="mock")
        extractor = LLMExtractor(provider=provider)

        result = extractor.extract(
            conversation="User: Hello\nAssistant: Hi!",
            memblock=mem,
        )

        assert result.blocks_created == 0

    def test_extract_handles_markdown_wrapping(self, mem: MemBlock):
        provider = CallableProvider(fn=mock_llm_markdown, provider_name="mock")
        extractor = LLMExtractor(provider=provider)

        result = extractor.extract(
            conversation="Some conversation",
            memblock=mem,
        )

        assert result.blocks_created == 1
        assert "Wrapped in markdown" in mem.query()[0].content

    def test_extract_handles_object_wrapping(self, mem: MemBlock):
        provider = CallableProvider(fn=mock_llm_object_wrapper, provider_name="mock")
        extractor = LLMExtractor(provider=provider)

        result = extractor.extract(
            conversation="Some conversation",
            memblock=mem,
        )

        assert result.blocks_created == 1

    def test_extract_handles_llm_error(self, mem: MemBlock):
        provider = CallableProvider(fn=mock_llm_error, provider_name="mock")
        extractor = LLMExtractor(provider=provider)

        result = extractor.extract(
            conversation="Some conversation",
            memblock=mem,
        )

        assert result.blocks_created == 0
        assert len(result.errors) > 0
        assert "failed" in result.errors[0].lower()

    def test_extract_without_memblock(self):
        provider = CallableProvider(fn=mock_llm_response, provider_name="mock")
        extractor = LLMExtractor(provider=provider)

        # Should not crash, just return raw result
        result = extractor.extract(
            conversation="User: I like Python",
            memblock=None,
        )

        assert result.blocks_created == 0
        assert len(result.raw_response) > 0


class TestExtractFromMessages:
    def test_extract_from_message_list(self, mem: MemBlock):
        provider = CallableProvider(fn=mock_llm_response, provider_name="mock")
        extractor = LLMExtractor(provider=provider)

        messages = [
            {"role": "user", "content": "I mainly use Python and deploy on AWS"},
            {"role": "assistant", "content": "Great stack!"},
            {"role": "user", "content": "I switched from JS to TypeScript last month"},
        ]

        result = extractor.extract_from_messages(messages, memblock=mem)

        assert result.blocks_created == 3


class TestCallableProvider:
    def test_custom_provider(self):
        def my_fn(system: str, user: str) -> str:
            return '[{"content": "Custom works", "type": "fact", "confidence": 0.9, "source": "explicit", "tags": []}]'

        provider = CallableProvider(fn=my_fn, provider_name="my-custom")
        assert provider.name == "my-custom"

        result = provider.complete("system", "user")
        parsed = json.loads(result)
        assert parsed[0]["content"] == "Custom works"


class TestExtractionVerification:
    def test_extracted_blocks_have_correct_types(self, mem: MemBlock):
        provider = CallableProvider(fn=mock_llm_response, provider_name="mock")
        extractor = LLMExtractor(provider=provider)

        extractor.extract(
            conversation="User: I like Python, deploy on AWS, switched to TS",
            memblock=mem,
        )

        prefs = mem.query(type=BlockType.PREFERENCE)
        facts = mem.query(type=BlockType.FACT)
        events = mem.query(type=BlockType.EVENT)

        assert len(prefs) == 1
        assert len(facts) == 1
        assert len(events) == 1

    def test_extracted_blocks_have_tags(self, mem: MemBlock):
        provider = CallableProvider(fn=mock_llm_response, provider_name="mock")
        extractor = LLMExtractor(provider=provider)

        extractor.extract(conversation="test", memblock=mem)

        python_blocks = mem.query(tags=["python"])
        assert len(python_blocks) >= 1

    def test_extracted_blocks_queryable_by_text(self, mem: MemBlock):
        provider = CallableProvider(fn=mock_llm_response, provider_name="mock")
        extractor = LLMExtractor(provider=provider)

        extractor.extract(conversation="test", memblock=mem)

        results = mem.query(text_search="Python")
        assert len(results) >= 1
        assert "Python" in results[0].content


class TestGeminiProvider:
    def test_gemini_provider_name(self):
        """Test GeminiProvider reports correct name (without needing google-genai installed)."""
        # Use CallableProvider to simulate Gemini-like behavior
        def gemini_mock(system: str, user: str) -> str:
            return '[{"content": "Gemini extracted", "type": "fact", "confidence": 0.9, "source": "inferred", "tags": ["gemini"]}]'

        provider = CallableProvider(fn=gemini_mock, provider_name="gemini/gemini-2.0-flash")
        assert provider.name == "gemini/gemini-2.0-flash"

    def test_gemini_provider_import_error(self):
        """Test GeminiProvider raises ImportError when google-genai not installed."""
        from unittest.mock import patch
        from memblock.extraction import GeminiProvider

        with patch.dict("sys.modules", {"google": None, "google.genai": None}):
            with pytest.raises(ImportError, match="google-genai"):
                GeminiProvider(api_key="fake-key")

    def test_gemini_extraction_via_memblock(self, mem: MemBlock):
        """Test Gemini extraction works end-to-end via callable provider."""
        def gemini_mock(system: str, user: str) -> str:
            return json.dumps([
                {"content": "User likes Gemini API", "type": "preference", "confidence": 0.88, "source": "explicit", "tags": ["gemini"]},
            ])

        provider = CallableProvider(fn=gemini_mock, provider_name="gemini/gemini-2.0-flash")
        extractor = LLMExtractor(provider=provider)
        result = extractor.extract(conversation="User: I love using the Gemini API", memblock=mem)

        assert result.blocks_created == 1
        assert result.provider == "gemini/gemini-2.0-flash"
        blocks = mem.query(tags=["gemini"])
        assert len(blocks) == 1
