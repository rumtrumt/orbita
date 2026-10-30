"""Экспорт в iCalendar (RFC 5545).

Ссылку на ленту можно добавить как подписку в Google, Яндекс или Apple Calendar. Синхронизация
односторонняя, зато не требует OAuth и хранения чужих токенов.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from orbita.domain.models import Item, ItemStatus

_PRODID = "-//Orbita//Voice Assistant//RU"


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def _fold(line: str) -> str:
    """Перенос строк длиннее 75 байт, как требует стандарт."""
    encoded = line.encode()
    if len(encoded) <= 75:
        return line
    chunks: list[str] = []
    current = ""
    size = 0
    limit = 75
    for char in line:
        width = len(char.encode())
        if size + width > limit:
            chunks.append(current)
            current, size, limit = "", 0, 74
        current += char
        size += width
    chunks.append(current)
    return "\r\n ".join(chunks)


def _event_lines(item: Item, tz: ZoneInfo, stamp: str, default_minutes: int) -> list[str]:
    assert item.day is not None
    lines = [
        "BEGIN:VEVENT",
        f"UID:orbita-{item.id}@orbita.local",
        f"DTSTAMP:{stamp}",
        f"SUMMARY:{_escape(item.title)}",
    ]
    if item.at is None:
        day = item.day.strftime("%Y%m%d")
        following = (item.day + timedelta(days=1)).strftime("%Y%m%d")
        lines += [f"DTSTART;VALUE=DATE:{day}", f"DTEND;VALUE=DATE:{following}"]
    else:
        start = datetime.combine(item.day, item.at, tzinfo=tz)
        end = start + timedelta(minutes=item.duration_minutes or default_minutes)
        lines += [
            f"DTSTART;TZID={tz.key}:{start:%Y%m%dT%H%M%S}",
            f"DTEND;TZID={tz.key}:{end:%Y%m%dT%H%M%S}",
        ]
    if item.recurrence:
        lines.append(f"RRULE:{item.recurrence}")
    lines.append("END:VEVENT")
    return lines


def render_ics(items: list[Item], tz: ZoneInfo, default_minutes: int = 60) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        f"PRODID:{_PRODID}",
        "CALSCALE:GREGORIAN",
        "X-WR-CALNAME:Орбита",
        f"X-WR-TIMEZONE:{tz.key}",
    ]
    for item in items:
        if item.day is not None and item.status is ItemStatus.PENDING:
            lines += _event_lines(item, tz, stamp, default_minutes)
    lines.append("END:VCALENDAR")
    return "\r\n".join(_fold(line) for line in lines) + "\r\n"
