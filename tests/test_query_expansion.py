"""Tests for `memblock.query_expansion.expand_query`."""

from __future__ import annotations

import pytest

from memblock.query_expansion import expand_query


# ─── Pure-function tests, no DB / async needed ──────────────────────


class TestReferentialDetection:
    def test_tell_me_more_expands(self):
        out = expand_query(
            "tell me more",
            [{"role": "user", "content": "what is a balanced fund"},
             {"role": "assistant", "content": "A Balanced Fund holds equity and debt"}],
        )
        assert out == "tell me more about Balanced Fund"

    def test_tell_me_more_about_it_expands(self):
        out = expand_query(
            "tell me more about it",
            [{"role": "user", "content": "explain SIP"},
             {"role": "assistant", "content": "SIP is a Systematic Investment Plan"}],
        )
        # "Systematic Investment Plan" is the capitalized NP
        assert "Systematic Investment Plan" in out

    def test_what_about_that_expands(self):
        out = expand_query(
            "what about that?",
            [{"role": "assistant", "content": "Parag Parikh Flexi Cap is a flexi-cap fund"}],
        )
        assert "what about Parag Parikh Flexi Cap" == out

    def test_compare_them_expands(self):
        out = expand_query(
            "compare them",
            [{"role": "assistant", "content": "HDFC Top 100 and SBI Bluechip are large-cap"}],
        )
        # Should pick one of the capitalized fund names
        assert out.startswith("compare ")
        assert ("HDFC Top" in out) or ("SBI Bluechip" in out)

    def test_explain_expands(self):
        out = expand_query(
            "explain",
            [{"role": "assistant", "content": "XIRR is the annualized return metric"}],
        )
        assert "explain" in out and "XIRR" in out

    def test_how_does_it_work(self):
        out = expand_query(
            "how does it work?",
            [{"role": "assistant", "content": "ELSS gives tax savings under 80C"}],
        )
        assert "ELSS" in out

    def test_bare_continuation_expands(self):
        for query in ("and?", "go on", "continue", "so?"):
            out = expand_query(
                query,
                [{"role": "assistant", "content": "Index Funds track a benchmark"}],
            )
            assert "Index Funds" in out, f"failed on {query!r}: got {out!r}"


class TestNonReferential:
    """Well-formed queries should not be expanded — never degrade
    a query that already has signal."""

    def test_full_question_unchanged(self):
        q = "What's the difference between ELSS and PPF?"
        out = expand_query(q, [{"role": "assistant", "content": "blah blah blah"}])
        assert out == q

    def test_named_entity_query_unchanged(self):
        q = "How is Parag Parikh Flexi Cap performing?"
        out = expand_query(q, [{"role": "assistant", "content": "Other context"}])
        assert out == q

    def test_empty_query_unchanged(self):
        out = expand_query("", [{"role": "assistant", "content": "X"}])
        assert out == ""


class TestNoContextFallback:
    def test_referential_but_no_context_returns_unchanged(self):
        out = expand_query("tell me more", [])
        assert out == "tell me more"

    def test_referential_but_only_stopwords_returns_unchanged(self):
        # Last assistant turn is just stopwords → no extractable topic
        out = expand_query(
            "tell me more",
            [{"role": "assistant", "content": "yes ok sure"}],
        )
        assert out == "tell me more"

    def test_max_lookback_respected(self):
        """With max_lookback=1 we only see the last turn."""
        out = expand_query(
            "tell me more",
            [
                {"role": "assistant", "content": "RetirementPlanning is great"},
                {"role": "user", "content": "yes ok"},  # nothing extractable
            ],
            max_lookback=1,
        )
        # Last turn has nothing → unchanged
        assert out == "tell me more"


class TestTopicSelection:
    def test_capitalized_phrase_preferred_over_lowercase(self):
        # Both "balanced fund" (lowercase) and "HDFC Top 100"
        # (capitalized) appear; the proper noun should win.
        out = expand_query(
            "tell me more",
            [{"role": "assistant",
              "content": "balanced fund options include HDFC Top 100"}],
        )
        assert "HDFC Top" in out

    def test_longest_phrase_wins_among_capitalized(self):
        out = expand_query(
            "tell me more",
            [{"role": "assistant",
              "content": "Parag Parikh Flexi Cap and Mirae are both equity"}],
        )
        # Longest capitalized phrase is "Parag Parikh Flexi Cap"
        assert "Parag Parikh Flexi Cap" in out
