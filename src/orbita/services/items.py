"""Бизнес-логика записей: создание, изменение, расписание, конфликты, напоминания."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from orbita.clock import Clock
from orbita.domain.models import Conflict, Item, ItemKind, ItemStatus, Occurrence
from orbita.domain.recurrence import (
    first_occurrence_on_or_after,
    next_occurrence_after,
    occurrences_between,
)
from orbita.storage.repositories import ItemRepository, item_snapshot

_DAY_END = time(21, 0)
_SLOT_STEP_MINUTES = 15
_UNTIMED_REMINDER_HOUR = time(9, 0)


@dataclass(slots=True)
class ItemDraft:
    kind: ItemKind
    title: str
    day: date | None = None
    at: time | None = None
    duration_minutes: int | None = None
    deadline: bool = False
    recurrence: str | None = None
    remind_before_minutes: int | None = None
    source_text: str | None = None


@dataclass(frozen=True, slots=True)
class Agenda:
    occurrences: list[Occurrence]
    overdue: list[Item]
    undated: list[Item]


class ItemService:
    def __init__(
        self,
        repository: ItemRepository,
        clock: Clock,
        timezone: ZoneInfo,
        *,
        meeting_default_minutes: int = 60,
        meeting_remind_before_minutes: int = 15,
        on_saved: Callable[[Item], None] | None = None,
    ) -> None:
        self._items = repository
        self._clock = clock
        self._tz = timezone
        self._meeting_minutes = meeting_default_minutes
        self._meeting_remind = meeting_remind_before_minutes
        self._on_saved = on_saved or (lambda _: None)

    @property
    def now(self) -> datetime:
        return self._clock.now(self._tz)

    @property
    def today(self) -> date:
        return self.now.date()

    def create(self, user_id: int, draft: ItemDraft) -> Item:
        moment = self.now
        item = Item(
            user_id=user_id,
            kind=draft.kind,
            title=draft.title,
            day=draft.day,
            at=draft.at,
            duration_minutes=self._duration_for(draft),
            deadline=draft.deadline,
            recurrence=draft.recurrence,
            remind_before_minutes=draft.remind_before_minutes,
            source_text=draft.source_text,
            created_at=moment,
            updated_at=moment,
        )
        item.remind_at = self.compute_remind_at(item, moment)
        self._items.add(item)
        self._items.record_revision(item, "create", None, moment)
        self._on_saved(item)
        return item

    def _duration_for(self, draft: ItemDraft) -> int | None:
        if draft.duration_minutes:
            return draft.duration_minutes
        return self._meeting_minutes if draft.kind is ItemKind.MEETING else None

    def get(self, user_id: int, item_id: int) -> Item:
        return self._items.get(user_id, item_id)

    def get_many(self, user_id: int, item_ids: list[int]) -> list[Item]:
        return self._items.get_many(user_id, item_ids)

    def update(self, item: Item, **changes: Any) -> Item:
        moment = self.now
        before = item_snapshot(item)
        updated = item.copy(**changes, updated_at=moment)
        if {"day", "at", "status", "recurrence", "remind_before_minutes"} & changes.keys():
            updated.remind_at = self.compute_remind_at(updated, moment)
        self._items.save(updated)
        self._items.record_revision(updated, "update", before, moment)
        if "title" in changes:
            self._on_saved(updated)
        return updated

    def complete(self, item: Item) -> Item:
        """Отметить выполненной. Повторяющаяся задача не закрывается, а уходит на следующий раз."""
        moment = self.now
        if item.is_recurring and item.day is not None:
            following = next_occurrence_after(item, max(item.day, self.today))
            if following:
                return self.update(item, day=following)
        return self.update(item, status=ItemStatus.DONE, completed_at=moment)

    def delete(self, item: Item) -> Item:
        moment = self.now
        before = item_snapshot(item)
        deleted = item.copy(deleted_at=moment, updated_at=moment, remind_at=None)
        self._items.save(deleted)
        self._items.record_revision(deleted, "delete", before, moment)
        return deleted

    def undo(self, user_id: int) -> tuple[Item, str] | None:
        return self._items.undo_last(user_id, self.now)

    def agenda(
        self,
        user_id: int,
        start: date,
        end: date,
        *,
        kind: ItemKind | None = None,
        include_done: bool = False,
    ) -> Agenda:
        """Расписание за период, включая просроченное, если период захватывает сегодня."""
        occurrences = [
            occurrence
            for item in self._items.in_window(user_id, start, end)
            for occurrence in occurrences_between(item, start, end)
            if self._matches(item, kind, include_done)
        ]
        occurrences.sort(key=lambda o: o.sort_key)
        overdue: list[Item] = []
        undated: list[Item] = []
        if start <= self.today <= end and not include_done:
            overdue = [
                item
                for item in self._items.overdue(user_id, self.today)
                if kind is None or item.kind is kind
            ]
            overdue.sort(key=lambda item: (item.day or date.max, item.created_at))
        return Agenda(occurrences, overdue, undated)

    def done(self, user_id: int, kind: ItemKind | None, limit: int = 20) -> list[Item]:
        items = [
            item
            for item in self._items.live(user_id)
            if item.status is ItemStatus.DONE and (kind is None or item.kind is kind)
        ]
        items.sort(key=lambda item: item.completed_at or item.updated_at or self.now, reverse=True)
        return items[:limit]

    def overdue(self, user_id: int, kind: ItemKind | None) -> list[Item]:
        items = self._items.overdue(user_id, self.today)
        return [item for item in items if kind is None or item.kind is kind]

    def undated(self, user_id: int) -> list[Item]:
        return self._items.undated(user_id)

    @staticmethod
    def _matches(item: Item, kind: ItemKind | None, include_done: bool) -> bool:
        if kind is not None and item.kind is not kind:
            return False
        if include_done:
            return item.status is ItemStatus.DONE
        return item.status is ItemStatus.PENDING

    def find_conflict(
        self,
        user_id: int,
        day: date,
        at: time,
        duration_minutes: int | None = None,
        exclude_id: int | None = None,
    ) -> Conflict | None:
        """Ближайшая встреча, пересекающая указанный слот, и свободное время рядом."""
        minutes = duration_minutes or self._meeting_minutes
        busy = self._busy_intervals(user_id, day, exclude_id)
        start = datetime.combine(day, at)
        end = start + timedelta(minutes=minutes)
        clashes = [slot for slot in busy if slot[0] < end and start < slot[1]]
        if not clashes:
            return None
        clashes.sort(key=lambda slot: slot[0])
        return Conflict(clashes[0][2], self._free_slot_after(start, minutes, busy))

    def _busy_intervals(
        self, user_id: int, day: date, exclude_id: int | None
    ) -> list[tuple[datetime, datetime, Occurrence]]:
        intervals = []
        for item in self._items.in_window(user_id, day, day):
            if item.id == exclude_id or item.kind is not ItemKind.MEETING:
                continue
            if item.status is not ItemStatus.PENDING or item.at is None:
                continue
            for occurrence in occurrences_between(item, day, day):
                begin = datetime.combine(day, item.at)
                length = item.duration_minutes or self._meeting_minutes
                intervals.append((begin, begin + timedelta(minutes=length), occurrence))
        return intervals

    @staticmethod
    def _free_slot_after(
        start: datetime,
        minutes: int,
        busy: list[tuple[datetime, datetime, Occurrence]],
    ) -> time | None:
        candidate = start
        latest_start = datetime.combine(start.date(), _DAY_END) - timedelta(minutes=minutes)
        while candidate <= latest_start:
            candidate_end = candidate + timedelta(minutes=minutes)
            blockers = [s for s in busy if s[0] < candidate_end and candidate < s[1]]
            if not blockers:
                return candidate.time()
            candidate = max(s[1] for s in blockers)
            # Округляем до шага сетки, чтобы не предлагать 15:47.
            overflow = candidate.minute % _SLOT_STEP_MINUTES
            if overflow:
                candidate += timedelta(minutes=_SLOT_STEP_MINUTES - overflow)
        return None

    def compute_remind_at(self, item: Item, moment: datetime) -> datetime | None:
        """Момент следующего срабатывания в UTC либо None, если напоминать не о чем."""
        if item.status is not ItemStatus.PENDING or item.day is None:
            return None
        if item.kind is ItemKind.TASK and item.at is None and item.remind_before_minutes is None:
            return None
        day: date | None = item.day
        if item.is_recurring:
            day = first_occurrence_on_or_after(item, self.today) or item.day
        # У повторений проверяем два ближайших дня: сегодняшний слот мог уже пройти.
        for _ in range(2):
            if day is None:
                return None
            target = self._reminder_moment(item, day)
            if target > moment:
                return target.astimezone(UTC)
            if not item.is_recurring:
                return None
            day = next_occurrence_after(item, day)
        return None

    def _reminder_moment(self, item: Item, day: date) -> datetime:
        start = datetime.combine(day, item.at or _UNTIMED_REMINDER_HOUR, tzinfo=self._tz)
        before = item.remind_before_minutes
        if before is None:
            before = self._meeting_remind if item.kind is ItemKind.MEETING else 0
        return start - timedelta(minutes=before)

    def take_due_reminders(self) -> list[Item]:
        """Забрать сработавшие напоминания и назначить следующие для серий."""
        moment = self.now
        due = self._items.due_reminders(moment)
        for item in due:
            item.remind_at = self.compute_remind_at(item, moment)
            self._items.save(item)
        return due
