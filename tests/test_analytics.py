"""Tests for org-level Q&A analytics."""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone

from memblock import MemBlock
from memblock.analytics import NoiseFilter, OrgAnalytics, QuestionRecord, QuestionTrend
from memblock.errors import AnalyticsError
from memblock.licensing import generate_license, validate_license, LicenseInfo


# ─── NoiseFilter Tests ────────────────────────────────────────────────────


class TestNoiseFilter:
    def test_default_noise_words_filtered(self):
        nf = NoiseFilter()
        assert nf.is_noise("hello")
        assert nf.is_noise("hi")
        assert nf.is_noise("thanks")
        assert nf.is_noise("ok")
        assert nf.is_noise("bye")

    def test_noise_case_insensitive(self):
        nf = NoiseFilter()
        assert nf.is_noise("Hello")
        assert nf.is_noise("THANKS")
        assert nf.is_noise("Hi")

    def test_noise_with_punctuation(self):
        nf = NoiseFilter()
        assert nf.is_noise("hello?")
        assert nf.is_noise("thanks!")
        assert nf.is_noise("ok.")

    def test_short_strings_filtered(self):
        nf = NoiseFilter()
        assert nf.is_noise("hi")
        assert nf.is_noise("ok")
        assert nf.is_noise("a")
        assert nf.is_noise("")

    def test_legitimate_questions_pass(self):
        nf = NoiseFilter()
        assert not nf.is_noise("how do I reset my password")
        assert not nf.is_noise("what is the refund policy")
        assert not nf.is_noise("can you help me with billing")

    def test_custom_noise_words(self):
        nf = NoiseFilter(custom_noise={"foobar", "bazqux"})
        assert nf.is_noise("foobar")
        assert nf.is_noise("bazqux")
        # Default still works
        assert nf.is_noise("hello")

    def test_add_noise_words(self):
        nf = NoiseFilter()
        assert not nf.is_noise("test123")
        nf.add_noise_words({"test123"})
        assert nf.is_noise("test123")

    def test_remove_noise_words(self):
        nf = NoiseFilter()
        assert nf.is_noise("hello")
        nf.remove_noise_words({"hello"})
        assert not nf.is_noise("hello")


# ─── Normalization Tests ──────────────────────────────────────────────────


class TestNormalization:
    def test_lowercase(self):
        assert OrgAnalytics.normalize_question("HOW DO I RESET") == "how do i reset"

    def test_whitespace_collapse(self):
        assert OrgAnalytics.normalize_question("how  do   i   reset") == "how do i reset"

    def test_trailing_punctuation_strip(self):
        assert OrgAnalytics.normalize_question("how do i reset?") == "how do i reset"
        assert OrgAnalytics.normalize_question("help me!!!") == "help me"
        assert OrgAnalytics.normalize_question("what is this...") == "what is this"

    def test_leading_trailing_whitespace(self):
        assert OrgAnalytics.normalize_question("  how do i reset  ") == "how do i reset"

    def test_combined_normalization(self):
        assert OrgAnalytics.normalize_question("  HOW  Do  I  Reset??  ") == "how do i reset"


# ─── QuestionRecord Tests ────────────────────────────────────────────────


class TestQuestionRecord:
    def test_to_dict_roundtrip(self):
        now = datetime.now(timezone.utc)
        record = QuestionRecord(
            id="qa_test123",
            org_id="org_1",
            normalized_text="how do i reset",
            frequency_count=5,
            unique_user_count=3,
            first_asked=now,
            last_asked=now,
        )
        data = record.to_dict()
        restored = QuestionRecord.from_dict(data)
        assert restored.id == record.id
        assert restored.org_id == record.org_id
        assert restored.normalized_text == record.normalized_text
        assert restored.frequency_count == record.frequency_count
        assert restored.unique_user_count == record.unique_user_count


# ─── Integration Tests (SQLite in-memory) ────────────────────────────────


class TestAnalyticsIntegration:
    """Integration tests using MemBlock with enable_analytics=True."""

    def _make_mem(self, **kwargs):
        return MemBlock(storage="sqlite:///:memory:", enable_analytics=True, **kwargs)

    def test_log_question_creates_record(self):
        mem = self._make_mem()
        record = mem.log_question("How do I reset my password?", user_id="u1", org_id="org1")
        assert record is not None
        assert record.normalized_text == "how do i reset my password"
        assert record.frequency_count == 1
        assert record.unique_user_count == 1
        mem.close()

    def test_duplicate_question_increments_frequency(self):
        mem = self._make_mem()
        mem.log_question("How do I reset my password?", user_id="u1", org_id="org1")
        record = mem.log_question("how do i reset my password", user_id="u1", org_id="org1")
        assert record.frequency_count == 2
        assert record.unique_user_count == 1
        mem.close()

    def test_different_user_increments_unique_users(self):
        mem = self._make_mem()
        mem.log_question("How do I reset my password?", user_id="u1", org_id="org1")
        record = mem.log_question("How do I reset my password?", user_id="u2", org_id="org1")
        assert record.frequency_count == 2
        assert record.unique_user_count == 2
        mem.close()

    def test_noise_filtered(self):
        mem = self._make_mem()
        result = mem.log_question("hello", user_id="u1", org_id="org1")
        assert result is None
        # Verify nothing was stored
        top = mem.get_top_questions(org_id="org1")
        assert len(top) == 0
        mem.close()

    def test_get_top_questions(self):
        mem = self._make_mem()
        # Log some questions with different frequencies
        for _ in range(5):
            mem.log_question("What is the refund policy?", user_id="u1", org_id="org1")
        for _ in range(3):
            mem.log_question("How do I change my email?", user_id="u1", org_id="org1")
        mem.log_question("Where is my order?", user_id="u1", org_id="org1")

        top = mem.get_top_questions(org_id="org1", limit=10)
        assert len(top) == 3
        assert top[0].normalized_text == "what is the refund policy"
        assert top[0].frequency_count == 5
        assert top[1].frequency_count == 3
        assert top[2].frequency_count == 1
        mem.close()

    def test_org_isolation(self):
        mem = self._make_mem()
        mem.log_question("How do I reset?", user_id="u1", org_id="org_a")
        mem.log_question("Where is my order?", user_id="u1", org_id="org_b")

        top_a = mem.get_top_questions(org_id="org_a")
        top_b = mem.get_top_questions(org_id="org_b")

        assert len(top_a) == 1
        assert top_a[0].normalized_text == "how do i reset"
        assert len(top_b) == 1
        assert top_b[0].normalized_text == "where is my order"
        mem.close()

    def test_question_stats(self):
        mem = self._make_mem()
        for _ in range(3):
            mem.log_question("What is the refund policy?", user_id="u1", org_id="org1")
        mem.log_question("How do I change email?", user_id="u1", org_id="org1")

        stats = mem.question_stats(org_id="org1")
        assert stats["total_unique_questions"] == 2
        assert stats["total_asks"] == 4
        assert stats["top_question"]["text"] == "what is the refund policy"
        assert stats["top_question"]["frequency"] == 3
        mem.close()

    def test_question_breakdown_daily(self):
        mem = self._make_mem()
        mem.log_question("How do I reset?", user_id="u1", org_id="org1")
        mem.log_question("How do I reset?", user_id="u1", org_id="org1")

        breakdown = mem.get_question_breakdown("How do I reset?", org_id="org1", granularity="daily")
        assert len(breakdown) > 0
        # All events happened today
        today_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert breakdown.get(today_key, 0) == 2
        mem.close()

    def test_question_breakdown_monthly(self):
        mem = self._make_mem()
        mem.log_question("How do I reset?", user_id="u1", org_id="org1")

        breakdown = mem.get_question_breakdown("How do I reset?", org_id="org1", granularity="monthly")
        month_key = datetime.now(timezone.utc).strftime("%Y-%m")
        assert breakdown.get(month_key, 0) == 1
        mem.close()

    def test_trending_questions(self):
        mem = self._make_mem()
        # Log questions - all recent so they should trend
        for _ in range(5):
            mem.log_question("What is the refund policy?", user_id="u1", org_id="org1")
        for _ in range(2):
            mem.log_question("How do I reset?", user_id="u1", org_id="org1")

        trending = mem.get_trending_questions(org_id="org1", window_days=7, limit=5)
        assert len(trending) > 0
        # Both questions were asked recently, so they should appear
        texts = [t.question.normalized_text for t in trending]
        assert "what is the refund policy" in texts
        mem.close()

    def test_custom_noise_words(self):
        mem = self._make_mem(analytics_noise_words={"foobar"})
        result = mem.log_question("foobar", user_id="u1", org_id="org1")
        assert result is None
        mem.close()

    def test_default_org_id(self):
        """When no org_id is passed, defaults to 'default'."""
        mem = self._make_mem()
        record = mem.log_question("How do I reset?", user_id="u1")
        assert record.org_id == "default"
        mem.close()

    def test_constructor_org_id_used(self):
        """When org_id is set in constructor, analytics uses it."""
        mem = MemBlock(storage="sqlite:///:memory:", enable_analytics=True, org_id="my_org")
        record = mem.log_question("How do I reset?", user_id="u1")
        assert record.org_id == "my_org"
        mem.close()

    def test_case_normalization_dedup(self):
        """'HOW DO I RESET?' and 'how do i reset' should be the same question."""
        mem = self._make_mem()
        mem.log_question("HOW DO I RESET?", user_id="u1", org_id="org1")
        record = mem.log_question("how do i reset", user_id="u1", org_id="org1")
        assert record.frequency_count == 2
        mem.close()


# ─── Analytics Disabled Tests ─────────────────────────────────────────────


class TestAnalyticsDisabled:
    def test_log_question_raises(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        with pytest.raises(AnalyticsError, match="Analytics not enabled"):
            mem.log_question("test question")
        mem.close()

    def test_get_top_questions_raises(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        with pytest.raises(AnalyticsError, match="Analytics not enabled"):
            mem.get_top_questions()
        mem.close()

    def test_get_trending_questions_raises(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        with pytest.raises(AnalyticsError, match="Analytics not enabled"):
            mem.get_trending_questions()
        mem.close()

    def test_get_question_breakdown_raises(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        with pytest.raises(AnalyticsError, match="Analytics not enabled"):
            mem.get_question_breakdown("test")
        mem.close()

    def test_question_stats_raises(self):
        mem = MemBlock(storage="sqlite:///:memory:")
        with pytest.raises(AnalyticsError, match="Analytics not enabled"):
            mem.question_stats()
        mem.close()


# ─── Licensing Tier Tests ─────────────────────────────────────────────────


class TestLicensingTier:
    SECRET = "test-secret-key-12345"

    def test_default_tier_is_community(self):
        key = generate_license("TestCorp", self.SECRET)
        info = validate_license(key, self.SECRET)
        assert info.tier == "community"

    def test_pro_tier(self):
        key = generate_license("TestCorp", self.SECRET, tier="pro")
        info = validate_license(key, self.SECRET)
        assert info.tier == "pro"

    def test_tier_in_signature(self):
        """Changing tier after signing should fail validation."""
        import base64
        import json

        key = generate_license("TestCorp", self.SECRET, tier="community")
        # Decode, tamper with tier, re-encode
        raw = base64.urlsafe_b64decode(key.strip()).decode("utf-8")
        payload = json.loads(raw)
        payload["tier"] = "pro"  # Tamper
        tampered = base64.urlsafe_b64encode(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")

        with pytest.raises(Exception):  # LicenseError for signature mismatch
            validate_license(tampered, self.SECRET)

    def test_backward_compatible_no_tier(self):
        """Old license keys without tier field should default to 'community'.
        Note: Old keys won't validate because the signature now includes tier,
        but new code generating keys will always include it."""
        # This tests that LicenseInfo defaults to "community"
        info = LicenseInfo(
            id="lic_test",
            customer="Test",
            issued_at=datetime.now().date(),
            expires_at=None,
        )
        assert info.tier == "community"
