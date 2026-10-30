"""Сущности предметной области. Не зависят ни от БД, ни от NLU, ни от транспорта."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta
from enum import StrEnum


class ItemKind(StrEnum):
    TASK = "task"
    REMINDER = "reminder"
    MEETING = "meeting"


class ItemStatus(StrEnum):
    PENDING = "pending"
    DONE = "done"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class Item:
    """Запись пользователя.

    Дата и время хранятся как локальные для пользователя («плавающие»): встреча в 10 утра
    остаётся в 10 утра при смене часового пояса или перехода на другое время, а серия
    повторений не расползается. В UTC хранится только ``remind_at`` — момент срабатывания.
    """

    user_id: int
    kind: ItemKind
    title: str
    id: int | None = None
    day: date | None = None
    at: time | None = None
    duration_minutes: int | None = None
    deadline: bool = False
    status: ItemStatus = ItemStatus.PENDING
    recurrence: str | None = None
    remind_at: datetime | None = None
    remind_before_minutes: int | None = None
    notes: str | None = None
    source_text: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    deleted_at: datetime | None = None

    @property
    def is_recurring(self) -> bool:
        return self.recurrence is not None

    def ends_at(self, default_minutes: int) -> time | None:
        if self.at is None:
            return None
        start = datetime.combine(date.min, self.at)
        return (start + timedelta(minutes=self.duration_minutes or default_minutes)).time()

    def copy(self, **changes: object) -> Item:
        return replace(self, **changes)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class Occurrence:
    """Конкретное появление записи в календаре. У разовой записи оно одно."""

    item: Item
    day: date
    at: time | None

    @property
    def sort_key(self) -> tuple[date, bool, time]:
        # Записи без времени идут после тех, что привязаны к часу.
        return (self.day, self.at is None, self.at or time.min)


@dataclass(frozen=True, slots=True)
class Conflict:
    occurrence: Occurrence
    suggested_at: time | None


@dataclass(frozen=True, slots=True)
class SearchHit:
    item: Item
    score: float
    matched_by: str


@dataclass(slots=True)
class User:
    name: str
    id: int | None = None
    timezone: str = "Europe/Moscow"
    calendar_token: str = ""
    created_at: datetime | None = None


@dataclass(slots=True)
class Reminder:
    item: Item
    fired_at: datetime
    meta: dict[str, object] = field(default_factory=dict)
