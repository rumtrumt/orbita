"""Детерминированный разбор русских выражений времени.

Парсер — единственное место в системе, где выполняется арифметика дат: и правила, и LLM
передают сюда текст выражения. Языковые модели ошибаются в календарных вычислениях, поэтому
им эта работа не доверяется (см. docs/adr/0002-dates-are-computed-not-generated.md).

Работа устроена как конвейер проходов по нормализованной строке: каждый проход находит своё
выражение и маскирует его пробелами, чтобы следующие не приняли те же символы за другое.
Порядок важен: сначала длинные и специфичные конструкции, затем общие.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

from orbita.nlu.temporal.model import DateRange, Temporal
from orbita.nlu.temporal.numbers import Normalized, normalize

_MONTH_STEMS = (
    "январ", "феврал", "март", "апрел", "ма", "июн", "июл", "август", "сентябр", "октябр",
    "ноябр", "декабр",
)  # fmt: skip
_MONTH_RE = (
    r"(январ[ьяею]|феврал[ьяею]|март[аеу]?|апрел[ьяею]|ма[йяею]|июн[ьяею]|июл[ьяею]"
    r"|август[аеу]?|сентябр[ьяею]|октябр[ьяею]|ноябр[ьяею]|декабр[ьяею])"
)
_WEEKDAY_RE = (
    r"(понедельник\w{0,2}|вторник\w{0,2}|сред[аеуы]|четверг\w{0,2}|пятниц[аеуы]"
    r"|суббот[аеуы]|воскресень[еюя]м?)"
)
_WEEKDAY_STEMS = ("понедельник", "вторник", "сред", "четверг", "пятниц", "суббот", "воскресен")
_RRULE_DAYS = ("MO", "TU", "WE", "TH", "FR", "SA", "SU")

_DAYPARTS = {"утра": "morning", "дня": "day", "вечера": "evening", "ночи": "night"}
_DAYPART_DEFAULTS = {
    "morning": time(9, 0),
    "day": time(13, 0),
    "evening": time(19, 0),
    "night": time(23, 0),
}
_DEADLINE_PREFIXES = {"до", "к", "ко"}
_QUANT = r"(\d+|пару|пара|несколько|полтора|полторы)"
_NEXT = r"следующ\w+|будущ\w+"
_THIS = r"эт\w+|текущ\w+|ближайш\w+"
_PREV = r"прошл\w+|предыдущ\w+|минувш\w+"
_UNIT_MINUTES = {"м": 1, "ч": 60, "д": 1440, "с": 1440, "н": 10080}


def _weekday_index(word: str) -> int:
    return next(i for i, stem in enumerate(_WEEKDAY_STEMS) if word.startswith(stem))


def _month_index(word: str) -> int:
    for i, stem in enumerate(_MONTH_STEMS):
        if word.startswith(stem) and (stem != "ма" or len(word) <= 3):
            return i + 1
    raise ValueError(word)


def _week_start(day: date) -> date:
    return day - timedelta(days=day.weekday())


def _month_range(year: int, month: int) -> DateRange:
    return DateRange(date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1]))


def _shift_month(day: date, months: int) -> date:
    total = day.year * 12 + day.month - 1 + months
    year, month = total // 12, total % 12 + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def _quantity(raw: str | None) -> float:
    if raw is None:
        return 1.0
    words = {"пару": 2.0, "пара": 2.0, "несколько": 3.0, "полтора": 1.5, "полторы": 1.5}
    return words.get(raw) or float(raw)


def _next_weekday(today: date, days: list[int]) -> date:
    return min(today + timedelta(days=(d - today.weekday()) % 7) for d in days)


@dataclass(slots=True)
class _Scan:
    """Изменяемое состояние одного прогона парсера."""

    source: Normalized
    now: datetime
    past: bool
    text: str = ""
    result: Temporal = field(default_factory=Temporal)
    daypart: str | None = None
    bare_hour: bool = False

    def __post_init__(self) -> None:
        self.text = self.source.text

    @property
    def today(self) -> date:
        return self.now.date()

    def take(self, match: re.Match[str]) -> None:
        start, end = match.span()
        self.text = self.text[:start] + " " * (end - start) + self.text[end:]
        self.result.spans.append(self.source.original_span(start, end))


def _pass_recurrence(s: _Scan) -> None:
    # Дни недели: «каждый понедельник и среду», «по вторникам и пятницам».
    day_word = (
        r"(?:понедельник\w*|вторник\w*|сред[аеуы]\w*|четверг\w*|пятниц\w*|суббот\w*|воскресень\w*)"
    )
    pattern = (
        rf"\b(?:кажд\w+|по)\s+({day_word}(?:(?:\s*,\s*|\s+и\s+|\s+)(?:кажд\w+\s+)?{day_word})*)"
    )
    if match := re.search(pattern, s.text):
        found = re.findall(day_word, match.group(1))
        days = sorted({_weekday_index(w) for w in found})
        s.result.recurrence = "FREQ=WEEKLY;BYDAY=" + ",".join(_RRULE_DAYS[d] for d in days)
        s.result.day = _next_weekday(s.today, days)
        s.take(match)
        return

    simple: tuple[tuple[str, str, time | None], ...] = (
        (r"\bкаждое\s+утро\b", "FREQ=DAILY", time(9, 0)),
        (r"\bкаждый\s+вечер\b", "FREQ=DAILY", time(19, 0)),
        (r"\b(?:каждый\s+день|ежедневно)\b", "FREQ=DAILY", None),
        (r"\b(?:кажд\w+\s+недел\w+|еженедельно)\b", "FREQ=WEEKLY", None),
        (r"\b(?:кажд\w+\s+месяц\w*|ежемесячно)\b", "FREQ=MONTHLY", None),
        (r"\b(?:кажд\w+\s+год\w*|ежегодно)\b", "FREQ=YEARLY", None),
        (
            r"\b(?:по\s+будням|по\s+рабочим\s+дням|кажд\w+\s+будн\w+(?:\s+день)?)\b",
            "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
            None,
        ),
        (r"\b(?:по\s+выходным|кажд\w+\s+выходн\w+)\b", "FREQ=WEEKLY;BYDAY=SA,SU", None),
    )
    for rule_pattern, rule, default_time in simple:
        if match := re.search(rule_pattern, s.text):
            s.result.recurrence = rule
            if default_time:
                s.result.at = default_time
            s.take(match)
            if "BYDAY" in rule:
                days = [_RRULE_DAYS.index(d) for d in rule.split("BYDAY=")[1].split(",")]
                s.result.day = _next_weekday(s.today, days)
            return

    if match := re.search(r"\bкажд\w+\s+(\d+)\s+(дн\w+|ден\w+|недел\w+|месяц\w*)", s.text):
        freq = {"д": "DAILY", "н": "WEEKLY", "м": "MONTHLY"}[match.group(2)[0]]
        s.result.recurrence = f"FREQ={freq};INTERVAL={match.group(1)}"
        s.take(match)
        return

    if match := re.search(r"\bкажд\w+\s+(\d+)(?:-?го|-?е)?\s+числ\w+", s.text):
        day_of_month = int(match.group(1))
        s.result.recurrence = f"FREQ=MONTHLY;BYMONTHDAY={day_of_month}"
        s.result.day = _next_month_day(s.today, day_of_month)
        s.take(match)


def _pass_interval(s: _Scan) -> None:
    """«с 10 до 12», «с 10:30 до 11:15», «с 2 до 4 дня»."""
    pattern = (
        r"\bс\s+(\d{1,2})(?:[:.](\d{2}))?(?:\s*час\w*)?\s+"
        r"(?:до|по)\s+(\d{1,2})(?:[:.](\d{2}))?(?:\s*час\w*)?(?:\s+(утра|дня|вечера|ночи))?"
    )
    match = re.search(pattern, s.text)
    if not match:
        return
    start_hour, start_min = int(match.group(1)), int(match.group(2) or 0)
    end_hour, end_min = int(match.group(3)), int(match.group(4) or 0)
    if not (start_hour < 24 and end_hour < 25 and start_min < 60 and end_min < 60):
        return
    if match.group(5) in ("дня", "вечера"):
        end_hour = end_hour + 12 if end_hour < 12 else end_hour
        start_hour = start_hour + 12 if start_hour < 12 else start_hour
    elif end_hour < start_hour:
        # «с 11 до 2» — конец точно после обеда.
        end_hour += 12
    s.result.at = time(start_hour, start_min)
    s.result.end_at = time(end_hour % 24, end_min)
    s.result.duration_minutes = (end_hour - start_hour) * 60 + end_min - start_min
    s.take(match)


def _pass_reminder_offset(s: _Scan) -> None:
    """«за 15 минут», «за час до встречи», «за день до начала»."""
    pattern = (
        rf"\bза\s+(?:{_QUANT}\s+)?(минут\w*|час\w*|дн\w+|день|сутки|недел\w+)"
        r"(?:\s+до\s+(?:встреч\w+|начал\w+|событи\w+|нее|него))?"
    )
    if match := re.search(pattern, s.text):
        minutes = _UNIT_MINUTES[match.group(2)[0]]
        s.result.remind_before_minutes = int(_quantity(match.group(1)) * minutes)
        s.take(match)


def _pass_shift(s: _Scan) -> None:
    """«на час позже», «на 2 дня раньше», «на полчаса вперед»."""
    pattern = (
        rf"\bна\s+(?:(полчаса)|(?:{_QUANT}\s+)?(минут\w*|час\w*|дн\w+|день|недел\w+))\s+"
        r"(позже|попозже|вперед|раньше|пораньше|назад)"
    )
    if match := re.search(pattern, s.text):
        if match.group(1):
            minutes = 30
        else:
            minutes = int(_quantity(match.group(2)) * _UNIT_MINUTES[match.group(3)[0]])
        forward = match.group(4) in ("позже", "попозже", "вперед")
        s.result.shift_minutes = minutes if forward else -minutes
        s.take(match)
        return
    if match := re.search(r"\b(попозже|пораньше)\b", s.text):
        s.result.shift_minutes = 60 if match.group(1) == "попозже" else -60
        s.take(match)


def _pass_relative_offset(s: _Scan) -> None:
    """«через два часа», «через неделю», «через полчаса», «спустя 3 дня»."""
    match = re.search(
        rf"\b(?:через|спустя)\s+(?:(полчаса)|(?:{_QUANT}\s+)?"
        r"(минут\w*|час\w*|дн\w+|день|сутки|суток|недел\w+|месяц\w*|год\w*|лет))",
        s.text,
    )
    if not match:
        return

    def set_moment(minutes: float) -> None:
        moment = s.now + timedelta(minutes=minutes)
        s.result.day = moment.date()
        s.result.at = moment.time().replace(second=0, microsecond=0)

    if match.group(1):
        set_moment(30)
    else:
        amount = _quantity(match.group(2))
        unit = match.group(3)
        if unit.startswith("минут"):
            set_moment(amount)
        elif unit.startswith("час"):
            set_moment(amount * 60)
        elif unit[0] in "дс":
            s.result.day = s.today + timedelta(days=int(amount))
        elif unit[0] == "н":
            s.result.day = s.today + timedelta(weeks=int(amount))
        elif unit.startswith("месяц"):
            s.result.day = _shift_month(s.today, int(amount))
        else:
            s.result.day = _shift_month(s.today, int(amount) * 12)
    s.take(match)


def _pass_duration(s: _Scan) -> None:
    """«на час», «на 30 минут», «на полтора часа», «на пару часов»."""
    if s.result.duration_minutes is not None:
        return
    match = re.search(
        rf"\bна\s+(?:(полчаса)|(?:{_QUANT}\s+)?(минут\w*|час\w{{0,2}}))(?=\s|$|[,.!?])", s.text
    )
    if not match:
        return
    if match.group(1):
        s.result.duration_minutes = 30
    else:
        scale = 60 if match.group(3)[0] == "ч" else 1
        s.result.duration_minutes = int(_quantity(match.group(2)) * scale)
    s.take(match)


def _pass_recent_range(s: _Scan) -> None:
    """«ближайшие 3 дня», «за последнюю неделю», «последние 2 дня»."""
    unit = r"(дн\w+|ден\w+|недел\w+|месяц\w*)"
    if match := re.search(rf"\b(?:на\s+|в\s+)?ближайш\w+\s+(?:(\d+)\s+)?{unit}", s.text):
        days = {"д": 1, "н": 7, "м": 30}[match.group(2)[0]] * int(match.group(1) or 1)
        s.result.period = DateRange(s.today, s.today + timedelta(days=days - 1))
        s.take(match)
        return
    if match := re.search(rf"\b(?:за\s+)?последн\w+\s+(?:(\d+)\s+)?{unit}", s.text):
        days = {"д": 1, "н": 7, "м": 30}[match.group(2)[0]] * int(match.group(1) or 1)
        s.result.period = DateRange(s.today - timedelta(days=days), s.today)
        s.take(match)


def _pass_edges(s: _Scan) -> None:
    """Границы периодов: «до конца недели», «в начале месяца», «к концу следующей недели»."""
    match = re.search(
        r"\b(до|к|в)\s+(конц\w*|начал\w*|середин\w*)\s+"
        rf"(?:({_THIS}|{_NEXT})\s+)?(рабочего\s+дня|дня|недел\w+|месяц\w*|года)",
        s.text,
    )
    if not match:
        return
    prefix, edge, modifier, unit = match.groups()
    next_unit = bool(modifier and re.match(_NEXT, modifier))
    if unit.endswith("дня"):
        target = s.today
    elif unit.startswith("недел"):
        monday = _week_start(s.today) + timedelta(weeks=1 if next_unit else 0)
        target = {
            "к": monday + timedelta(days=6),
            "н": monday,
            "с": monday + timedelta(days=2),
        }[edge[0]]
        if prefix == "в" and edge.startswith("конц"):
            target = monday + timedelta(days=4)  # «в конце недели» — обычно пятница
    elif unit.startswith("месяц"):
        anchor = _shift_month(s.today.replace(day=1), 1 if next_unit else 0)
        bounds = _month_range(anchor.year, anchor.month)
        target = {"к": bounds.end, "н": bounds.start, "с": bounds.start.replace(day=15)}[edge[0]]
    else:
        year = s.today.year + (1 if next_unit else 0)
        target = {"к": date(year, 12, 31), "н": date(year, 1, 1), "с": date(year, 7, 1)}[edge[0]]
    s.result.day = target
    s.result.deadline = prefix in ("до", "к")
    s.take(match)


def _week(monday: date, offset: int) -> DateRange:
    start = monday + timedelta(weeks=offset)
    return DateRange(start, start + timedelta(days=6))


def _weekend(monday: date, today: date, offset: int) -> DateRange:
    saturday = monday + timedelta(days=5, weeks=offset)
    start = max(saturday, today) if offset == 0 else saturday
    return DateRange(start, saturday + timedelta(days=1))


def _month_at(today: date, offset: int) -> DateRange:
    anchor = _shift_month(today.replace(day=1), offset)
    return _month_range(anchor.year, anchor.month)


def _year_at(year: int) -> DateRange:
    return DateRange(date(year, 1, 1), date(year, 12, 31))


def _pass_periods(s: _Scan) -> None:
    today = s.today
    monday = _week_start(today)
    patterns: tuple[tuple[str, DateRange], ...] = (
        (rf"\b(?:(?:на|в)\s+)?(?:{_NEXT})\s+недел\w+", _week(monday, 1)),
        (rf"\b(?:(?:на|в|за)\s+)?(?:{_PREV})\s+недел\w+", _week(monday, -1)),
        (rf"\b(?:(?:на|в|за)\s+)?(?:{_THIS})\s+недел\w+", _week(monday, 0)),
        (r"\b(?:на|в)\s+(?:эти\s+|ближайшие\s+)?выходн\w+", _weekend(monday, today, 0)),
        (rf"\b(?:на|в)\s+(?:{_NEXT})\s+выходн\w+", _weekend(monday, today, 1)),
        (rf"\b(?:на|в)\s+(?:{_PREV})\s+выходн\w+", _weekend(monday, today, -1)),
        (rf"\b(?:на|в)\s+(?:{_NEXT})\s+месяц\w*", _month_at(today, 1)),
        (rf"\b(?:на|в|за)\s+(?:{_PREV})\s+месяц\w*", _month_at(today, -1)),
        (rf"\b(?:на|в|за)\s+(?:{_THIS})\s+месяц\w*", _month_at(today, 0)),
        (rf"\b(?:на|в|за)\s+(?:{_THIS})\s+год\w*", _year_at(today.year)),
        (rf"\b(?:на|в)\s+(?:{_NEXT})\s+год\w*", _year_at(today.year + 1)),
    )
    for pattern, period in patterns:
        if match := re.search(pattern, s.text):
            s.result.period = period
            s.take(match)
            return


def _full_year(text: str) -> int:
    return int(text) + 2000 if len(text) == 2 else int(text)


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _with_inferred_year(s: _Scan, day: int, month: int, year: int | None) -> date | None:
    if year is not None:
        return _safe_date(year, month, day)
    candidate = _safe_date(s.today.year, month, day)
    if candidate is None:
        return None
    if s.past and candidate > s.today:
        return _safe_date(s.today.year - 1, month, day)
    if not s.past and candidate < s.today:
        return _safe_date(s.today.year + 1, month, day)
    return candidate


def _next_month_day(today: date, day: int) -> date | None:
    for offset in (0, 1, 2):
        anchor = _shift_month(today.replace(day=1), offset)
        candidate = _safe_date(anchor.year, anchor.month, day)
        if candidate and candidate >= today:
            return candidate
    return None


def _set_day(s: _Scan, day: date, prefix: str | None, match: re.Match[str]) -> None:
    s.result.day = day
    s.result.deadline = prefix in _DEADLINE_PREFIXES
    s.take(match)


def _pass_explicit_date(s: _Scan) -> None:
    # ISO-формат.
    if match := re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", s.text):
        resolved = _safe_date(int(match[1]), int(match[2]), int(match[3]))
        if resolved:
            _set_day(s, resolved, None, match)
            return

    pattern = r"(?:\b(до|к|ко|на|в|с)\s+)?\b(\d{1,2})\.(\d{1,2})(?:\.(\d{2,4}))?\b"
    for match in re.finditer(pattern, s.text):
        prefix, first, second, year_text = match.groups()
        # «в 10.30» — время, «15.11» и «на 15.11» — дата.
        if year_text is None and (prefix == "в" or int(second) > 12 or second == "00"):
            continue
        year = _full_year(year_text) if year_text else None
        if resolved := _with_inferred_year(s, int(first), int(second), year):
            _set_day(s, resolved, prefix, match)
            return

    # «15 ноября», «пятнадцатого ноября 2027», «к 1 января».
    match = re.search(
        rf"(?:\b(до|к|ко|на|в|с)\s+)?\b(\d{{1,2}})(?:-?го|-?е)?\s+{_MONTH_RE}"
        r"(?:\s+(\d{4})(?:\s*г\w*)?)?\b",
        s.text,
    )
    if match:
        prefix, day_text, month_text, year_text = match.groups()
        year = int(year_text) if year_text else None
        if resolved := _with_inferred_year(s, int(day_text), _month_index(month_text), year):
            _set_day(s, resolved, prefix, match)
            return

    # «в ноябре» — весь месяц.
    if match := re.search(rf"\bв\s+{_MONTH_RE}\b", s.text):
        month = _month_index(match.group(1))
        year = s.today.year
        if not s.past and month < s.today.month:
            year += 1
        elif s.past and month > s.today.month:
            year -= 1
        s.result.period = _month_range(year, month)
        s.take(match)
        return

    # «15-го», «к пятому» — ближайшее такое число.
    match = re.search(r"(?:\b(до|к|ко|на|в)\s+)\b(\d{1,2})-?го\b(?!\s*час)", s.text)
    if match and (target := _next_month_day(s.today, int(match.group(2)))):
        _set_day(s, target, match.group(1), match)


def _pass_relative_day(s: _Scan) -> None:
    match = re.search(
        r"(?:\b(до|к|на|за)\s+)?\b(послепослезавтра|послезавтра|завтрашн\w+(?:\s+дн\w+)?|завтра"
        r"|сегодняшн\w+(?:\s+дн\w+)?|сегодня|вчерашн\w+(?:\s+дн\w+)?|вчера|позавчера)\b",
        s.text,
    )
    if not match:
        return
    word = match.group(2)
    offsets = (
        ("послепосле", 3), ("после", 2), ("завтр", 1), ("сегодня", 0), ("вчер", -1), ("поза", -2),
    )  # fmt: skip
    offset = next(value for stem, value in offsets if word.startswith(stem))
    _set_day(s, s.today + timedelta(days=offset), match.group(1), match)


def _pass_weekday(s: _Scan) -> None:
    match = re.search(
        rf"(?:\b(до|к|ко|в|во|на|по)\s+)?(?:\b({_THIS}|{_NEXT}|{_PREV})\s+)?\b{_WEEKDAY_RE}\b",
        s.text,
    )
    if not match:
        return
    prefix, modifier, name = match.groups()
    target = _weekday_index(name)
    monday = _week_start(s.today)
    delta = (target - s.today.weekday()) % 7
    back = ((s.today.weekday() - target) % 7) or 7
    if modifier and re.match(_PREV, modifier):
        day = s.today - timedelta(days=back)
    elif modifier and re.match(_NEXT, modifier):
        # «В следующую пятницу» — пятница следующей календарной недели.
        day = monday + timedelta(weeks=1, days=target)
    elif modifier:
        day = s.today + timedelta(days=delta)
    elif s.past:
        day = s.today - timedelta(days=back)
    else:
        # Без уточнения название сегодняшнего дня означает такой же день через неделю.
        day = s.today + timedelta(days=delta or 7)
    _set_day(s, day, prefix, match)


def _pass_special_times(s: _Scan) -> None:
    """Слова-времена, которые не сводятся к часу: «полдень», «полночь», «обед»."""
    table: tuple[tuple[str, time, int, bool], ...] = (
        (r"\b(?:в\s+)?полдень\b", time(12, 0), 0, False),
        (r"\b(?:к|до)\s+полудн\w+\b", time(12, 0), 0, True),
        (r"\b(?:в\s+)?полночь\b", time(0, 0), 1, False),
        (r"\bпосле\s+обеда\b", time(14, 0), 0, False),
        (r"\bдо\s+обеда\b", time(12, 0), 0, True),
        (r"\bк\s+обеду\b", time(12, 0), 0, True),
        (r"\b(?:в|на)\s+обед\b", time(13, 0), 0, False),
    )
    for pattern, moment, day_shift, is_deadline in table:
        if match := re.search(pattern, s.text):
            s.result.at = moment
            s.result.deadline = s.result.deadline or is_deadline
            if day_shift:
                s.result.day = (s.result.day or s.today) + timedelta(days=day_shift)
            s.take(match)
            return


def _with_daypart(hour: int, daypart: str) -> int:
    if daypart in ("day", "evening"):
        return hour + 12 if hour < 12 else hour
    if daypart == "night":
        return 0 if hour == 12 else hour + 12 if hour >= 9 else hour
    return hour


def _set_clock(s: _Scan, hour: int, minute: int, match: re.Match[str], prefix: str | None) -> bool:
    if not (0 <= hour < 25 and 0 <= minute < 60):
        return False
    s.result.at = time(hour % 24, minute)
    s.result.deadline = s.result.deadline or prefix in _DEADLINE_PREFIXES
    s.take(match)
    return True


def _pass_clock(s: _Scan) -> None:
    if s.result.at is not None:
        return

    # «без четверти 5», «без 10 5»
    match = re.search(r"\bв?\s*без\s+(четверти|\d{1,2})\s*(?:минут\w*\s+)?(\d{1,2})\b", s.text)
    if match:
        minutes = 15 if match.group(1) == "четверти" else int(match.group(1))
        if _set_clock(s, (int(match.group(2)) - 1) % 24, 60 - minutes, match, None):
            s.bare_hour = True
            return

    # «в половине третьего» (числительные уже нормализованы)
    match = re.search(r"\b(?:в\s+)?половин\w+\s+(\d{1,2})(?:-?го)?\b", s.text)
    if match and _set_clock(s, (int(match.group(1)) - 1) % 24, 30, match, None):
        s.bare_hour = True
        return

    # «в 15:00», «в 15.30», «к 9:45»
    match = re.search(r"(?:\b(в|во|к|до|на|около|после)\s+)?\b(\d{1,2})[:.](\d{2})\b", s.text)
    if match and _set_clock(s, int(match.group(2)), int(match.group(3)), match, match.group(1)):
        return

    # «в 3 часа дня», «10 утра», «в 7 вечера», «в час ночи»
    match = re.search(
        r"(?:\b(в|к|до|около)\s+)?\b(?:(\d{1,2})\s+)?(?:час\w*\s+)?(?:(\d{1,2})\s+минут\w*\s+)?"
        r"(утра|дня|вечера|ночи)\b",
        s.text,
    )
    if match and (match.group(2) or "час" in match.group(0)):
        hour = _with_daypart(int(match.group(2) or 1), _DAYPARTS[match.group(4)])
        if _set_clock(s, hour, int(match.group(3) or 0), match, match.group(1)):
            return

    # «в 15 30», «в 9 00»
    match = re.search(r"\b(в|к|до)\s+(\d{1,2})\s+(\d{2})\b", s.text)
    if match and _set_clock(s, int(match.group(2)), int(match.group(3)), match, match.group(1)):
        return

    # «в 11», «к 9», «в 8 часов», «на 16»
    match = re.search(r"\b(в|во|к|до|около|на)\s+(\d{1,2})(?:\s*час\w*)?(?![\d.:-])", s.text)
    if match and _set_clock(s, int(match.group(2)), 0, match, match.group(1)):
        s.bare_hour = 0 < int(match.group(2)) <= 12
        return

    # «в час», «к часу» без указания части суток
    if match := re.search(r"\b(в\s+час|к\s+часу)\b", s.text):
        prefix = "к" if match.group(0).startswith("к") else None
        if _set_clock(s, 1, 0, match, prefix):
            s.bare_hour = True


def _pass_daypart_word(s: _Scan) -> None:
    match = re.search(
        r"\b(?:(с|к|до|под)\s+)?(утром|утра|утру|днем|вечером|вечера|вечеру|ночью|ночи)\b", s.text
    )
    if not match:
        return
    prefix, word = match.groups()
    # «утра/вечера» без предлога — обрывок выражения времени, а не указание на часть суток.
    if word in ("утра", "вечера", "ночи") and prefix is None:
        return
    s.daypart = {"у": "morning", "д": "day", "в": "evening", "н": "night"}[word[0]]
    if prefix in ("к", "до"):
        s.result.deadline = True
    s.take(match)


def _resolve(s: _Scan) -> None:
    result = s.result
    if s.daypart:
        if result.at is None:
            result.at = _DAYPART_DEFAULTS[s.daypart]
        elif s.bare_hour:
            result.at = result.at.replace(hour=_with_daypart(result.at.hour, s.daypart) % 24)
            s.bare_hour = False
    elif s.bare_hour and result.at is not None and result.at.hour < 8:
        # Без части суток «в 3» — почти всегда день: в три ночи встречи не назначают.
        result.at = result.at.replace(hour=result.at.hour + 12)
        result.meridiem_assumed = True

    if result.day is None and result.period is None and result.at is not None and not s.past:
        result.day_inferred = True
        tz = s.now.tzinfo
        afternoon = result.at.hour + 12 if result.at.hour < 12 else result.at.hour
        if datetime.combine(s.today, result.at, tzinfo=tz) > s.now:
            result.day = s.today
        elif (
            s.bare_hour
            and datetime.combine(s.today, result.at.replace(hour=afternoon), tzinfo=tz) > s.now
        ):
            # «в 8» в половине седьмого вечера — это сегодня в 20:00, а не завтра утром.
            result.at = result.at.replace(hour=afternoon)
            result.day = s.today
            result.meridiem_assumed = True
        else:
            result.day = s.today + timedelta(days=1)

    if result.end_at is None and result.at and result.duration_minutes and not result.deadline:
        end = datetime.combine(s.today, result.at) + timedelta(minutes=result.duration_minutes)
        result.end_at = end.time()


_PASSES = (
    _pass_recurrence,
    _pass_interval,
    _pass_shift,
    _pass_recent_range,
    _pass_reminder_offset,
    _pass_relative_offset,
    _pass_edges,
    _pass_periods,
    _pass_explicit_date,
    _pass_relative_day,
    _pass_weekday,
    _pass_duration,
    _pass_special_times,
    _pass_clock,
    _pass_daypart_word,
)


def parse_temporal(text: str, now: datetime, *, past: bool = False) -> Temporal:
    """Разобрать все временные выражения в тексте относительно момента ``now``.

    ``past`` смещает неоднозначные выражения в прошлое: в вопросе «когда была встреча в
    четверг» речь о прошедшем четверге.
    """
    scan = _Scan(source=normalize(text), now=now, past=past)
    for step in _PASSES:
        step(scan)
    _resolve(scan)
    scan.result.spans.sort()
    return scan.result


def strip_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Вырезать найденные выражения из исходного текста."""
    chunks: list[str] = []
    cursor = 0
    for start, end in sorted(spans):
        if start > cursor:
            chunks.append(text[cursor:start])
        cursor = max(cursor, end)
    chunks.append(text[cursor:])
    return re.sub(r"\s+", " ", " ".join(chunks)).strip(" ,.;:-")
