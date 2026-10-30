"""Результат разбора временного выражения."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time


@dataclass(frozen=True, slots=True)
class DateRange:
    """Закрытый диапазон дат, обе границы включены."""

    start: date
    end: date

    def contains(self, day: date) -> bool:
        return self.start <= day <= self.end


@dataclass(slots=True)
class Temporal:
    day: date | None = None
    at: time | None = None
    period: DateRange | None = None
    end_at: time | None = None
    duration_minutes: int | None = None
    deadline: bool = False
    recurrence: str | None = None
    remind_before_minutes: int | None = None
    shift_minutes: int | None = None
    # Дата выведена, а не названа: «в 10 утра» вечером означает завтрашнее утро.
    day_inferred: bool = False
    # Час без «утра/вечера» из рабочего диапазона: «в 3» трактуется как 15:00.
    meridiem_assumed: bool = False
    spans: list[tuple[int, int]] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not any(
            (
                self.day,
                self.at,
                self.period,
                self.duration_minutes,
                self.recurrence,
                self.remind_before_minutes,
                self.shift_minutes is not None,
            )
        )
