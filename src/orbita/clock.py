"""Источник времени. Всё, что зависит от «сейчас», получает его отсюда ради проверяемости."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol
from zoneinfo import ZoneInfo


class Clock(Protocol):
    def now(self, tz: ZoneInfo) -> datetime:
        """Текущий момент в указанном часовом поясе."""
        ...


class SystemClock:
    def now(self, tz: ZoneInfo) -> datetime:
        return datetime.now(tz)


class FixedClock:
    """Часы для тестов и детерминированных прогонов."""

    def __init__(self, moment: datetime) -> None:
        self.moment = moment

    def now(self, tz: ZoneInfo) -> datetime:
        return self.moment.astimezone(tz)
