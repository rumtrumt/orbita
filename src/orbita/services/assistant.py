"""Сценарий одной реплики: транзакция, диалог, история сообщений, уведомления."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

import structlog
from sqlalchemy.orm import Session

from orbita.container import Container
from orbita.dialog import format as fmt
from orbita.dialog.manager import DialogManager, TurnResult
from orbita.domain.models import Item, Occurrence, User
from orbita.services.calendar import render_ics
from orbita.services.items import ItemService
from orbita.services.search import SearchService
from orbita.storage.repositories import (
    DialogRepository,
    EmbeddingRepository,
    ItemRepository,
    MessageRepository,
)
from orbita.voice.speech_text import to_speech

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class Scope:
    """Сервисы, привязанные к одной транзакции."""

    session: Session
    items: ItemService
    search: SearchService
    timezone: ZoneInfo


def item_payload(item: Item, service: ItemService, default_minutes: int) -> dict[str, Any]:
    end = item.ends_at(default_minutes) if item.kind.value == "meeting" else None
    return {
        "id": item.id,
        "kind": item.kind.value,
        "title": item.title,
        "day": item.day.isoformat() if item.day else None,
        "at": item.at.strftime("%H:%M") if item.at else None,
        "end": end.strftime("%H:%M") if end else None,
        "deadline": item.deadline,
        "status": item.status.value,
        "recurring": item.is_recurring,
        "when": fmt.item_when(item, service.today),
    }


def occurrence_payload(
    occurrence: Occurrence, service: ItemService, default_minutes: int
) -> dict[str, Any]:
    return item_payload(occurrence.item, service, default_minutes) | {
        "day": occurrence.day.isoformat(),
    }


class Assistant:
    def __init__(self, container: Container) -> None:
        self._c = container

    @contextmanager
    def scope(self, user: User) -> Iterator[Scope]:
        timezone = ZoneInfo(user.timezone)
        with self._c.database.session() as session:
            repository = ItemRepository(session)
            search = SearchService(
                repository, EmbeddingRepository(session), self._c.embedder, self._c.reranker
            )
            items = ItemService(
                repository,
                self._c.clock,
                timezone,
                meeting_default_minutes=self._c.settings.meeting_default_minutes,
                meeting_remind_before_minutes=self._c.settings.meeting_remind_before_minutes,
                on_saved=search.index,
            )
            yield Scope(session, items, search, timezone)

    def turn(self, user: User, text: str, *, via: str = "text") -> dict[str, Any]:
        assert user.id is not None
        with self.scope(user) as scope:
            manager = DialogManager(
                self._c.interpreter,
                scope.items,
                scope.search,
                DialogRepository(scope.session),
                scope.timezone,
            )
            messages = MessageRepository(scope.session)
            moment = scope.items.now
            messages.add(user.id, "user", text, {"via": via}, moment)
            result = manager.handle(user.id, text)
            payload = self._payload(result, scope.items)
            messages.add(user.id, "assistant", result.reply, payload, scope.items.now)
        log.info("turn.done", action=result.action, nlu=result.nlu_source, awaiting=result.awaiting)
        if result.changed:
            self._c.bus.publish(user.id, {"type": "changed"})
        return payload

    def _payload(self, result: TurnResult, items: ItemService) -> dict[str, Any]:
        minutes = self._c.settings.meeting_default_minutes
        return {
            "reply": result.reply,
            "speech": to_speech(result.reply),
            "action": result.action,
            "items": [item_payload(item, items, minutes) for item in result.items],
            "awaiting": result.awaiting,
            "suggestions": result.suggestions,
            "nlu": result.nlu_source,
            "changed": result.changed,
        }

    def agenda(self, user: User, start: Any, end: Any) -> dict[str, Any]:
        assert user.id is not None
        minutes = self._c.settings.meeting_default_minutes
        with self.scope(user) as scope:
            agenda = scope.items.agenda(user.id, start, end)
            return {
                "today": scope.items.today.isoformat(),
                "now": scope.items.now.strftime("%H:%M"),
                "occurrences": [
                    occurrence_payload(o, scope.items, minutes) for o in agenda.occurrences
                ],
                "overdue": [item_payload(i, scope.items, minutes) for i in agenda.overdue],
                "undated": [
                    item_payload(i, scope.items, minutes) for i in scope.items.undated(user.id)
                ],
            }

    def calendar(self, user: User) -> str:
        assert user.id is not None
        with self.scope(user) as scope:
            items = ItemRepository(scope.session).live(user.id)
            return render_ics(items, scope.timezone, self._c.settings.meeting_default_minutes)
