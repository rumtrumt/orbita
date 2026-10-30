"""Фоновая доставка напоминаний."""

from __future__ import annotations

import asyncio
from zoneinfo import ZoneInfo

import structlog

from orbita.container import Container
from orbita.dialog import format as fmt
from orbita.domain.models import Item, ItemKind
from orbita.services.items import ItemService
from orbita.storage.repositories import ItemRepository, MessageRepository, UserRepository
from orbita.voice.speech_text import to_speech

log = structlog.get_logger(__name__)


def reminder_text(item: Item) -> str:
    if item.kind is ItemKind.MEETING and item.at:
        return f"Скоро встреча: {item.title}, в {fmt.clock(item.at)}."
    return f"Напоминаю: {item.title}."


def fire_due_reminders(container: Container) -> int:
    """Доставить всё, что созрело. Состояние в БД, поэтому после рестарта ничего не теряется."""
    settings = container.settings
    with container.database.session() as session:
        repository = ItemRepository(session)
        service = ItemService(
            repository,
            container.clock,
            ZoneInfo(settings.default_timezone),
            meeting_default_minutes=settings.meeting_default_minutes,
            meeting_remind_before_minutes=settings.meeting_remind_before_minutes,
        )
        due = service.take_due_reminders()
        users = UserRepository(session)
        messages = MessageRepository(session)
        for item in due:
            user = users.get(item.user_id)
            if user is None:
                continue
            text = reminder_text(item)
            messages.add(item.user_id, "assistant", text, {"reminder": True}, service.now)
            container.bus.publish(
                item.user_id,
                {"type": "reminder", "text": text, "speech": to_speech(text), "id": item.id},
            )
    return len(due)


async def reminder_loop(container: Container) -> None:
    while True:
        try:
            await asyncio.to_thread(fire_due_reminders, container)
        except Exception:
            log.exception("reminders.tick_failed")
        await asyncio.sleep(container.settings.reminder_poll_seconds)
