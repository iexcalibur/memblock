"""Temporal query parser — extract time constraints from natural language queries.

Pure Python, zero external dependencies. Parses temporal expressions like:
- "yesterday", "last week", "3 months ago"
- "in January 2024", "on March 10th", "in 2023"
- "before", "after", "during"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable


@dataclass
class TemporalConstraint:
    """Parsed temporal constraint from a query."""

    after: datetime | None = None  # results must be after this time
    before: datetime | None = None  # results must be before this time
    cleaned_query: str | None = None  # query with temporal words removed


# Month name -> number mapping
_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2,
    "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

# Relative time words
_RELATIVE_PATTERNS: list[tuple[re.Pattern, Callable[[re.Match, datetime], TemporalConstraint]]] = []


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _register(pattern: str, handler: Callable[[re.Match, datetime], TemporalConstraint]) -> None:
    _RELATIVE_PATTERNS.append((re.compile(pattern, re.IGNORECASE), handler))


# "yesterday"
def _yesterday(m: re.Match, now: datetime) -> TemporalConstraint:
    start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return TemporalConstraint(after=start, before=end)

_register(r"\byesterday\b", _yesterday)


# "today"
def _today(m: re.Match, now: datetime) -> TemporalConstraint:
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return TemporalConstraint(after=start, before=end)

_register(r"\btoday\b", _today)


# "last week"
def _last_week(m: re.Match, now: datetime) -> TemporalConstraint:
    start = (now - timedelta(weeks=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return TemporalConstraint(after=start, before=now)

_register(r"\blast\s+week\b", _last_week)


# "last month"
def _last_month(m: re.Match, now: datetime) -> TemporalConstraint:
    start = (now - timedelta(days=30)).replace(hour=0, minute=0, second=0, microsecond=0)
    return TemporalConstraint(after=start, before=now)

_register(r"\blast\s+month\b", _last_month)


# "last year"
def _last_year(m: re.Match, now: datetime) -> TemporalConstraint:
    start = (now - timedelta(days=365)).replace(hour=0, minute=0, second=0, microsecond=0)
    return TemporalConstraint(after=start, before=now)

_register(r"\blast\s+year\b", _last_year)


# "N days/weeks/months/years ago"
def _n_units_ago(m: re.Match, now: datetime) -> TemporalConstraint:
    n = int(m.group(1))
    unit = m.group(2).lower().rstrip("s")
    if unit == "day":
        delta = timedelta(days=n)
    elif unit == "week":
        delta = timedelta(weeks=n)
    elif unit == "month":
        delta = timedelta(days=n * 30)
    elif unit == "year":
        delta = timedelta(days=n * 365)
    else:
        return TemporalConstraint()
    target = now - delta
    # Give a window around the target
    start = (target - timedelta(days=max(1, n))).replace(hour=0, minute=0, second=0, microsecond=0)
    end = (target + timedelta(days=max(1, n)))
    return TemporalConstraint(after=start, before=end)

_register(r"\b(\d+)\s+(days?|weeks?|months?|years?)\s+ago\b", _n_units_ago)


# "recently" / "recent"
def _recently(m: re.Match, now: datetime) -> TemporalConstraint:
    start = (now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
    return TemporalConstraint(after=start, before=now)

_register(r"\brecent(?:ly)?\b", _recently)


# "in January 2024" / "in March" / "in 2023"
def _in_month_year(m: re.Match, now: datetime) -> TemporalConstraint:
    month_str = m.group(1).lower() if m.group(1) else None
    year_str = m.group(2) if m.group(2) else None

    if month_str and month_str in _MONTHS:
        month = _MONTHS[month_str]
        year = int(year_str) if year_str else now.year
        start = datetime(year, month, 1, tzinfo=timezone.utc)
        # End of month
        if month == 12:
            end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
        return TemporalConstraint(after=start, before=end)
    elif year_str:
        year = int(year_str)
        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        return TemporalConstraint(after=start, before=end)
    return TemporalConstraint()

_register(
    r"\bin\s+(" + "|".join(_MONTHS.keys()) + r")(?:\s+(\d{4}))?\b",
    _in_month_year,
)
_register(r"\bin\s+(\d{4})\b", lambda m, now: TemporalConstraint(
    after=datetime(int(m.group(1)), 1, 1, tzinfo=timezone.utc),
    before=datetime(int(m.group(1)) + 1, 1, 1, tzinfo=timezone.utc),
))


# "before <month> <year>" / "before <year>"
def _before_time(m: re.Match, now: datetime) -> TemporalConstraint:
    text = m.group(1).strip().lower()
    # Try year
    year_match = re.match(r"^(\d{4})$", text)
    if year_match:
        return TemporalConstraint(before=datetime(int(year_match.group(1)), 1, 1, tzinfo=timezone.utc))
    # Try month year
    for month_name, month_num in _MONTHS.items():
        if text.startswith(month_name):
            rest = text[len(month_name):].strip()
            year = int(rest) if rest.isdigit() else now.year
            return TemporalConstraint(before=datetime(year, month_num, 1, tzinfo=timezone.utc))
    return TemporalConstraint()

_register(r"\bbefore\s+(.+?)(?:\s*[,.]|\s+(?:did|was|do|the|what|where|who|how|when)|\s*$)", _before_time)


# "after <month> <year>" / "after <year>"
def _after_time(m: re.Match, now: datetime) -> TemporalConstraint:
    text = m.group(1).strip().lower()
    year_match = re.match(r"^(\d{4})$", text)
    if year_match:
        return TemporalConstraint(after=datetime(int(year_match.group(1)) + 1, 1, 1, tzinfo=timezone.utc))
    for month_name, month_num in _MONTHS.items():
        if text.startswith(month_name):
            rest = text[len(month_name):].strip()
            year = int(rest) if rest.isdigit() else now.year
            if month_num == 12:
                return TemporalConstraint(after=datetime(year + 1, 1, 1, tzinfo=timezone.utc))
            return TemporalConstraint(after=datetime(year, month_num + 1, 1, tzinfo=timezone.utc))
    return TemporalConstraint()

_register(r"\bafter\s+(.+?)(?:\s*[,.]|\s+(?:did|was|do|the|what|where|who|how|when)|\s*$)", _after_time)


# "last Tuesday" / "last Monday" etc.
_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

def _last_weekday(m: re.Match, now: datetime) -> TemporalConstraint:
    day_name = m.group(1).lower()
    if day_name not in _WEEKDAYS:
        return TemporalConstraint()
    target_day = _WEEKDAYS[day_name]
    current_day = now.weekday()
    days_back = (current_day - target_day) % 7
    if days_back == 0:
        days_back = 7
    target = (now - timedelta(days=days_back)).replace(hour=0, minute=0, second=0, microsecond=0)
    return TemporalConstraint(after=target, before=target + timedelta(days=1))

_register(r"\blast\s+(" + "|".join(_WEEKDAYS.keys()) + r")\b", _last_weekday)


# Temporal stop words to remove from queries
_TEMPORAL_STOP_PATTERNS = [
    r"\byesterday\b",
    r"\btoday\b",
    r"\blast\s+(?:week|month|year|" + "|".join(_WEEKDAYS.keys()) + r")\b",
    r"\b\d+\s+(?:days?|weeks?|months?|years?)\s+ago\b",
    r"\brecent(?:ly)?\b",
    r"\bin\s+(?:" + "|".join(_MONTHS.keys()) + r")(?:\s+\d{4})?\b",
    r"\bin\s+\d{4}\b",
    r"\bbefore\s+(?:" + "|".join(_MONTHS.keys()) + r"|" + r"\d{4})(?:\s+\d{4})?\b",
    r"\bafter\s+(?:" + "|".join(_MONTHS.keys()) + r"|" + r"\d{4})(?:\s+\d{4})?\b",
]


class TemporalQueryParser:
    """Parse temporal expressions from natural language queries."""

    def parse(self, query: str) -> TemporalConstraint | None:
        """
        Extract temporal constraints from a query string.

        Returns TemporalConstraint with after/before bounds and cleaned query,
        or None if no temporal expression found.
        """
        now = _now()

        for pattern, handler in _RELATIVE_PATTERNS:
            match = pattern.search(query)
            if match:
                constraint = handler(match, now)
                if constraint.after is not None or constraint.before is not None:
                    # Clean temporal words from query
                    cleaned = query
                    for stop_pat in _TEMPORAL_STOP_PATTERNS:
                        cleaned = re.sub(stop_pat, "", cleaned, flags=re.IGNORECASE)
                    cleaned = re.sub(r"\s+", " ", cleaned).strip()
                    constraint.cleaned_query = cleaned if cleaned else query
                    return constraint

        return None
