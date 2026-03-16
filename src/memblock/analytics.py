"""Org-level Q&A analytics — deduplicated question tracking with frequency counts."""

from __future__ import annotations

import re
import unicodedata
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, TYPE_CHECKING

from memblock.errors import AnalyticsError

if TYPE_CHECKING:
    from memblock.storage.base import StorageAdapter


# ─── Data Models ──────────────────────────────────────────────────────────


@dataclass
class QuestionRecord:
    """A single tracked question at the org level."""

    id: str
    org_id: str
    normalized_text: str
    frequency_count: int
    unique_user_count: int
    first_asked: datetime
    last_asked: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "org_id": self.org_id,
            "normalized_text": self.normalized_text,
            "frequency_count": self.frequency_count,
            "unique_user_count": self.unique_user_count,
            "first_asked": self.first_asked.isoformat(),
            "last_asked": self.last_asked.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuestionRecord:
        return cls(
            id=data["id"],
            org_id=data["org_id"],
            normalized_text=data["normalized_text"],
            frequency_count=data["frequency_count"],
            unique_user_count=data["unique_user_count"],
            first_asked=(
                datetime.fromisoformat(data["first_asked"])
                if isinstance(data["first_asked"], str)
                else data["first_asked"]
            ),
            last_asked=(
                datetime.fromisoformat(data["last_asked"])
                if isinstance(data["last_asked"], str)
                else data["last_asked"]
            ),
        )


@dataclass
class QuestionTrend:
    """A question with its trending score."""

    question: QuestionRecord
    trend_score: float  # ratio of recent frequency to historical average


# ─── Noise Filter ─────────────────────────────────────────────────────────


class NoiseFilter:
    """Filters out noise questions like greetings and short strings."""

    DEFAULT_NOISE: set[str] = {
        "hello", "hi", "hey", "thanks", "thank you", "ok", "okay",
        "bye", "goodbye", "yes", "no", "sure", "cool", "nice",
        "good", "great", "fine", "help", "please", "yo", "sup",
        "hola", "greetings", "howdy", "cheers", "thx", "ty",
        "np", "nope", "yep", "yeah", "yea", "nah",
    }

    MIN_LENGTH = 3

    def __init__(self, custom_noise: set[str] | None = None) -> None:
        self._noise = {w.lower() for w in self.DEFAULT_NOISE}
        if custom_noise:
            self._noise |= {w.lower() for w in custom_noise}

    def is_noise(self, text: str) -> bool:
        """Return True if the text is noise and should not be tracked."""
        cleaned = text.strip().lower().rstrip("?!.,;:")
        if len(cleaned) < self.MIN_LENGTH:
            return True
        return cleaned in self._noise

    def add_noise_words(self, words: set[str]) -> None:
        self._noise |= {w.lower() for w in words}

    def remove_noise_words(self, words: set[str]) -> None:
        self._noise -= {w.lower() for w in words}


# ─── Org Analytics Engine ─────────────────────────────────────────────────


class OrgAnalytics:
    """
    Org-level Q&A analytics engine.

    Tracks all questions asked across users in an organization,
    deduplicates them, and provides frequency analysis.
    """

    def __init__(
        self,
        storage: StorageAdapter,
        noise_filter: NoiseFilter | None = None,
    ) -> None:
        self._storage = storage
        self._noise = noise_filter or NoiseFilter()

    @staticmethod
    def normalize_question(text: str) -> str:
        """Normalize a question for consistent matching.

        - Unicode NFC normalization
        - Lowercase
        - Collapse whitespace
        - Strip trailing punctuation
        """
        normalized = unicodedata.normalize("NFC", text)
        normalized = normalized.strip().lower()
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = normalized.rstrip("?!.,;:")
        return normalized

    def log_question(
        self,
        org_id: str,
        user_id: str,
        question: str,
    ) -> QuestionRecord | None:
        """Log a user question for org-level analytics.

        Returns None if the question is filtered as noise.
        Deduplicates on normalized_text within the same org_id.
        """
        if self._noise.is_noise(question):
            return None

        normalized = self.normalize_question(question)
        if not normalized:
            return None

        now = datetime.now(timezone.utc)
        data = self._storage.upsert_question(
            org_id=org_id,
            normalized_text=normalized,
            user_id=user_id,
            asked_at=now,
        )
        if not data:
            raise AnalyticsError(
                "Storage adapter does not support analytics operations. "
                "Ensure you are using SQLiteAdapter or PostgreSQLAdapter."
            )
        return QuestionRecord.from_dict(data)

    def get_top_questions(
        self,
        org_id: str,
        limit: int = 20,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[QuestionRecord]:
        """Get the most frequently asked questions for an org."""
        rows = self._storage.get_top_questions(
            org_id=org_id, limit=limit, since=since, until=until,
        )
        return [QuestionRecord.from_dict(r) for r in rows]

    def get_trending(
        self,
        org_id: str,
        window_days: int = 7,
        limit: int = 10,
    ) -> list[QuestionTrend]:
        """Get questions trending upward in frequency.

        Compares frequency in the recent window vs. the overall average.
        """
        all_questions = self._storage.get_all_questions(org_id=org_id)
        if not all_questions:
            return []

        now = datetime.now(timezone.utc)
        window_start = now - timedelta(days=window_days)

        trends: list[QuestionTrend] = []
        for q_data in all_questions:
            question = QuestionRecord.from_dict(q_data)
            events = self._storage.get_question_events(
                org_id=org_id,
                question_id=question.id,
            )

            total_count = len(events)
            if total_count == 0:
                continue

            # Count events in the recent window
            recent_count = 0
            for ev in events:
                asked_at = ev["asked_at"]
                if isinstance(asked_at, str):
                    asked_at = datetime.fromisoformat(asked_at)
                if asked_at.tzinfo is None:
                    asked_at = asked_at.replace(tzinfo=timezone.utc)
                if asked_at >= window_start:
                    recent_count += 1

            if recent_count == 0:
                continue

            # Calculate total lifespan in days
            first = question.first_asked
            if first.tzinfo is None:
                first = first.replace(tzinfo=timezone.utc)
            lifespan_days = max((now - first).days, 1)

            # Daily average over entire lifespan
            daily_avg = total_count / lifespan_days
            # Daily rate in recent window
            recent_daily = recent_count / window_days

            # Trend score: how much faster is recent compared to average
            trend_score = recent_daily / daily_avg if daily_avg > 0 else float(recent_count)

            trends.append(QuestionTrend(question=question, trend_score=trend_score))

        # Sort by trend_score descending
        trends.sort(key=lambda t: t.trend_score, reverse=True)
        return trends[:limit]

    def get_question_breakdown(
        self,
        org_id: str,
        question_text: str,
        granularity: str = "daily",
    ) -> dict[str, int]:
        """Get frequency breakdown for a specific question.

        Args:
            granularity: "daily", "weekly", or "monthly"

        Returns:
            Dict mapping date keys to counts.
            Daily: "2026-03-16", Weekly: "2026-W11", Monthly: "2026-03"
        """
        normalized = self.normalize_question(question_text)
        q_data = self._storage.get_question_by_text(org_id=org_id, normalized_text=normalized)
        if not q_data:
            return {}

        events = self._storage.get_question_events(
            org_id=org_id,
            question_id=q_data["id"],
        )

        breakdown: dict[str, int] = defaultdict(int)
        for ev in events:
            asked_at = ev["asked_at"]
            if isinstance(asked_at, str):
                asked_at = datetime.fromisoformat(asked_at)

            if granularity == "daily":
                key = asked_at.strftime("%Y-%m-%d")
            elif granularity == "weekly":
                key = f"{asked_at.isocalendar()[0]}-W{asked_at.isocalendar()[1]:02d}"
            elif granularity == "monthly":
                key = asked_at.strftime("%Y-%m")
            else:
                raise AnalyticsError(f"Invalid granularity: {granularity}. Use 'daily', 'weekly', or 'monthly'.")

            breakdown[key] += 1

        return dict(breakdown)

    def question_stats(self, org_id: str) -> dict[str, Any]:
        """Get summary analytics stats for an org."""
        all_questions = self._storage.get_all_questions(org_id=org_id)

        total_unique = len(all_questions)
        total_asks = sum(q["frequency_count"] for q in all_questions)

        top_question = None
        if all_questions:
            top = max(all_questions, key=lambda q: q["frequency_count"])
            top_question = {
                "text": top["normalized_text"],
                "frequency": top["frequency_count"],
                "unique_users": top["unique_user_count"],
            }

        return {
            "org_id": org_id,
            "total_unique_questions": total_unique,
            "total_asks": total_asks,
            "top_question": top_question,
        }
