"""Развёртка повторяющихся записей. Правила хранятся в формате RRULE (RFC 5545)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime, time, timedelta

from dateutil.rrule import rrulestr

from orbita.domain.models import Item, Occurrence


def occurrences_between(item: Item, start: date, end: date) -> Iterator[Occurrence]:
    """Появления записи в закрытом диапазоне дат."""
    if item.day is None:
        return
    if not item.recurrence:
        if start <= item.day <= end:
            yield Occurrence(item, item.day, item.at)
        return
    rule = rrulestr(
        f"RRULE:{item.recurrence}",
        dtstart=datetime.combine(item.day, item.at or time.min),
    )
    window_end = datetime.combine(end, time.max)
    window_start = datetime.combine(max(start, item.day), time.min)
    for moment in rule.xafter(window_start, inc=True):
        if moment > window_end:
            return
        yield Occurrence(item, moment.date(), item.at)


def next_occurrence_after(item: Item, day: date) -> date | None:
    """Первая дата повторения строго позже ``day``."""
    if item.day is None or not item.recurrence:
        return None
    rule = rrulestr(
        f"RRULE:{item.recurrence}",
        dtstart=datetime.combine(item.day, item.at or time.min),
    )
    following = rule.after(datetime.combine(day, time.max))
    return following.date() if following else None


def first_occurrence_on_or_after(item: Item, day: date) -> date | None:
    return next_occurrence_after(item, day - timedelta(days=1))
