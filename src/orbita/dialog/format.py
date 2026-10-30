"""Формулировки дат и списков по-русски."""

from __future__ import annotations

from datetime import date, time, timedelta

from orbita.domain.models import Item, ItemKind, Occurrence
from orbita.nlu.temporal import DateRange

MONTHS_GENITIVE = (
    "января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября",
    "октября", "ноября", "декабря",
)  # fmt: skip
_WEEKDAYS_ACCUSATIVE = (
    "понедельник", "вторник", "среду", "четверг", "пятницу", "субботу", "воскресенье",
)  # fmt: skip
_WEEKDAY_PREPOSITION = ("в", "во", "в", "в", "в", "в", "в")
_DAY_WORDS = {-2: "позавчера", -1: "вчера", 0: "сегодня", 1: "завтра", 2: "послезавтра"}
_RULE_DAYS = {
    "MO": "понедельникам", "TU": "вторникам", "WE": "средам", "TH": "четвергам",
    "FR": "пятницам", "SA": "субботам", "SU": "воскресеньям",
}  # fmt: skip


def plural(count: int, one: str, few: str, many: str) -> str:
    tail = count % 100
    if 11 <= tail <= 14:
        return many
    return {1: one, 2: few, 3: few, 4: few}.get(count % 10, many)


def count_words(count: int, one: str, few: str, many: str) -> str:
    return f"{count} {plural(count, one, few, many)}"


def clock(at: time) -> str:
    return at.strftime("%H:%M")


def date_label(day: date, today: date) -> str:
    label = f"{day.day} {MONTHS_GENITIVE[day.month - 1]}"
    return label if day.year == today.year else f"{label} {day.year}"


def human_day(day: date, today: date) -> str:
    """«завтра, 31 октября» или «в пятницу, 6 ноября»."""
    delta = (day - today).days
    label = date_label(day, today)
    if delta in _DAY_WORDS:
        return f"{_DAY_WORDS[delta]}, {label}"
    if abs(delta) <= 13:
        index = day.weekday()
        return f"{_WEEKDAY_PREPOSITION[index]} {_WEEKDAYS_ACCUSATIVE[index]}, {label}"
    return label


def human_moment(day: date, at: time | None, today: date) -> str:
    text = human_day(day, today)
    return f"{text}, в {clock(at)}" if at else text


def human_period(period: DateRange, today: date) -> str:
    """Подпись к периоду в заголовке ответа."""
    if period.start == period.end:
        return human_day(period.start, today)
    monday = today - timedelta(days=today.weekday())
    if period.start == monday and period.end == monday + timedelta(days=6):
        return "на этой неделе"
    if period.start == monday + timedelta(weeks=1) and (period.end - period.start).days == 6:
        return "на следующей неделе"
    if period.end == monday + timedelta(days=6) and period.start.weekday() >= 5:
        return "на выходных"
    return f"с {date_label(period.start, today)} по {date_label(period.end, today)}"


def describe_recurrence(rule: str) -> str:
    parts = dict(chunk.split("=", 1) for chunk in rule.split(";"))
    interval = int(parts.get("INTERVAL", "1"))
    freq = parts.get("FREQ")
    if "BYDAY" in parts:
        days = parts["BYDAY"].split(",")
        if days == ["MO", "TU", "WE", "TH", "FR"]:
            return "по будням"
        if days == ["SA", "SU"]:
            return "по выходным"
        names = [_RULE_DAYS[d] for d in days]
        joined = " и ".join(names) if len(names) <= 2 else ", ".join(names)
        return f"по {joined}"
    if "BYMONTHDAY" in parts:
        return f"каждый месяц {parts['BYMONTHDAY']}-го числа"
    if interval > 1:
        unit = {
            "DAILY": ("день", "дня", "дней"),
            "WEEKLY": ("неделю", "недели", "недель"),
            "MONTHLY": ("месяц", "месяца", "месяцев"),
        }.get(freq or "", ("раз", "раза", "раз"))
        return f"каждые {count_words(interval, *unit)}"
    return {
        "DAILY": "каждый день",
        "WEEKLY": "каждую неделю",
        "MONTHLY": "каждый месяц",
        "YEARLY": "каждый год",
    }.get(freq or "", "регулярно")


def kind_label(kind: ItemKind) -> str:
    return {ItemKind.TASK: "задачу", ItemKind.REMINDER: "напоминание", ItemKind.MEETING: "встречу"}[
        kind
    ]


def item_when(item: Item, today: date) -> str:
    if item.recurrence:
        rule = describe_recurrence(item.recurrence)
        return f"{rule}, в {clock(item.at)}" if item.at else rule
    if item.day is None:
        return "без даты"
    if item.deadline:
        moment = human_day(item.day, today).split(", ")[-1]
        return f"до {clock(item.at)}, {moment}" if item.at else f"к {moment}"
    return human_moment(item.day, item.at, today)


def occurrence_line(occurrence: Occurrence, show_day: bool, today: date) -> str:
    """Строка расписания: «15:00 — Встреча с командой»."""
    item = occurrence.item
    prefix = clock(occurrence.at) if occurrence.at else "в течение дня"
    if show_day:
        prefix = f"{date_label(occurrence.day, today)}, {prefix}"
    return f"{prefix} — {item.title}"


def join_natural(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " и " + parts[-1]
