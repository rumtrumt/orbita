"""Разбор времени. «Сегодня» во всех тестах — пятница, 30 октября 2026, 14:00."""

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pytest

from orbita.nlu.temporal import DateRange, parse_temporal, strip_spans

MSK = ZoneInfo("Europe/Moscow")
NOW = datetime(2026, 10, 30, 14, 0, tzinfo=MSK)


def p(text: str, now: datetime = NOW, **kw):
    return parse_temporal(text, now, **kw)


@pytest.mark.parametrize(
    ("text", "day", "at"),
    [
        ("сегодня", date(2026, 10, 30), None),
        ("завтра", date(2026, 10, 31), None),
        ("послезавтра", date(2026, 11, 1), None),
        ("вчера", date(2026, 10, 29), None),
        ("завтра в 10 утра", date(2026, 10, 31), time(10)),
        ("завтра в 15:00", date(2026, 10, 31), time(15)),
        ("завтра в 15.30", date(2026, 10, 31), time(15, 30)),
        ("завтра в 15 30", date(2026, 10, 31), time(15, 30)),
        ("завтра в девять ноль ноль", date(2026, 10, 31), time(9)),
        ("завтра в десять утра", date(2026, 10, 31), time(10)),
        ("завтра в семь вечера", date(2026, 10, 31), time(19)),
        ("завтра в 3 часа дня", date(2026, 10, 31), time(15)),
        ("завтра в час дня", date(2026, 10, 31), time(13)),
        ("завтра в половине третьего", date(2026, 10, 31), time(14, 30)),
        ("завтра без четверти пять", date(2026, 10, 31), time(16, 45)),
        ("завтра в полдень", date(2026, 10, 31), time(12)),
        ("завтра в обед", date(2026, 10, 31), time(13)),
        ("завтра утром", date(2026, 10, 31), time(9)),
        ("завтра вечером", date(2026, 10, 31), time(19)),
        ("завтра вечером в 8", date(2026, 10, 31), time(20)),
        ("завтра с утра", date(2026, 10, 31), time(9)),
        ("сегодня вечером", date(2026, 10, 30), time(19)),
        ("в пятницу", date(2026, 11, 6), None),
        ("в эту пятницу", date(2026, 10, 30), None),
        ("в понедельник", date(2026, 11, 2), None),
        ("в четверг в 11", date(2026, 11, 5), time(11)),
        ("в воскресенье", date(2026, 11, 1), None),
        ("во вторник", date(2026, 11, 3), None),
        ("в среду", date(2026, 11, 4), None),
        ("в следующую среду", date(2026, 11, 4), None),
        ("в следующий понедельник", date(2026, 11, 2), None),
        ("в прошлый четверг", date(2026, 10, 29), None),
        ("15 ноября", date(2026, 11, 15), None),
        ("пятнадцатого ноября", date(2026, 11, 15), None),
        ("двадцать пятого декабря в 18:30", date(2026, 12, 25), time(18, 30)),
        ("1 января", date(2027, 1, 1), None),
        ("15.11", date(2026, 11, 15), None),
        ("15.11.2027", date(2027, 11, 15), None),
        ("2026-12-01", date(2026, 12, 1), None),
        ("через неделю", date(2026, 11, 6), None),
        ("через 3 дня", date(2026, 11, 2), None),
        ("через три дня в 10", date(2026, 11, 2), time(10)),
        ("через месяц", date(2026, 11, 30), None),
    ],
)
def test_day_and_time(text, day, at):
    result = p(text)
    assert (result.day, result.at) == (day, at), text


@pytest.mark.parametrize(
    ("text", "day", "at"),
    [
        ("через два часа", date(2026, 10, 30), time(16)),
        ("через час", date(2026, 10, 30), time(15)),
        ("через полчаса", date(2026, 10, 30), time(14, 30)),
        ("через полтора часа", date(2026, 10, 30), time(15, 30)),
        ("через 15 минут", date(2026, 10, 30), time(14, 15)),
        ("через пару часов", date(2026, 10, 30), time(16)),
        ("через 11 часов", date(2026, 10, 31), time(1)),
    ],
)
def test_relative_offsets(text, day, at):
    result = p(text)
    assert (result.day, result.at) == (day, at), text


def test_time_without_date_rolls_to_tomorrow_when_passed():
    result = p("в 10 утра")
    assert result.day == date(2026, 10, 31)
    assert result.day_inferred


def test_time_without_date_stays_today_when_ahead():
    assert p("в 18:00").day == date(2026, 10, 30)


def test_bare_small_hour_means_afternoon():
    result = p("завтра в 3")
    assert result.at == time(15)
    assert result.meridiem_assumed


def test_bare_evening_hour_today():
    result = p("в 8", datetime(2026, 10, 30, 18, 30, tzinfo=MSK))
    assert (result.day, result.at) == (date(2026, 10, 30), time(20))


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("на этой неделе", date(2026, 10, 26), date(2026, 11, 1)),
        ("на эту неделю", date(2026, 10, 26), date(2026, 11, 1)),
        ("на следующей неделе", date(2026, 11, 2), date(2026, 11, 8)),
        ("на прошлой неделе", date(2026, 10, 19), date(2026, 10, 25)),
        ("в выходные", date(2026, 10, 31), date(2026, 11, 1)),
        ("в этом месяце", date(2026, 10, 1), date(2026, 10, 31)),
        ("в следующем месяце", date(2026, 11, 1), date(2026, 11, 30)),
        ("в декабре", date(2026, 12, 1), date(2026, 12, 31)),
        ("на ближайшие 3 дня", date(2026, 10, 30), date(2026, 11, 1)),
    ],
)
def test_periods(text, start, end):
    assert p(text).period == DateRange(start, end), text


@pytest.mark.parametrize(
    ("text", "day"),
    [
        ("до конца недели", date(2026, 11, 1)),
        ("до конца месяца", date(2026, 10, 31)),
        ("до конца следующей недели", date(2026, 11, 8)),
        ("до конца дня", date(2026, 10, 30)),
        ("до конца года", date(2026, 12, 31)),
        ("до пятницы", date(2026, 11, 6)),
        ("к понедельнику", date(2026, 11, 2)),
        ("к среде", date(2026, 11, 4)),
        ("до завтра", date(2026, 10, 31)),
        ("к 15 ноября", date(2026, 11, 15)),
        ("до 20.11", date(2026, 11, 20)),
    ],
)
def test_deadlines(text, day):
    result = p(text)
    assert result.day == day, text
    assert result.deadline, text


def test_soft_edges_are_not_deadlines():
    result = p("в начале следующей недели")
    assert result.day == date(2026, 11, 2)
    assert not result.deadline


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        ("каждый день", "FREQ=DAILY"),
        ("ежедневно", "FREQ=DAILY"),
        ("каждую неделю", "FREQ=WEEKLY"),
        ("каждый месяц", "FREQ=MONTHLY"),
        ("каждую пятницу", "FREQ=WEEKLY;BYDAY=FR"),
        ("по понедельникам", "FREQ=WEEKLY;BYDAY=MO"),
        ("по вторникам и четвергам", "FREQ=WEEKLY;BYDAY=TU,TH"),
        ("каждый понедельник и среду", "FREQ=WEEKLY;BYDAY=MO,WE"),
        ("по будням", "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR"),
        ("каждые 2 недели", "FREQ=WEEKLY;INTERVAL=2"),
        ("каждое 15 число", "FREQ=MONTHLY;BYMONTHDAY=15"),
    ],
)
def test_recurrence(text, rule):
    assert p(text).recurrence == rule, text


def test_recurrence_with_time_and_first_occurrence():
    result = p("каждую среду в 10 утра")
    assert result.recurrence == "FREQ=WEEKLY;BYDAY=WE"
    assert (result.day, result.at) == (date(2026, 11, 4), time(10))


def test_interval():
    result = p("завтра с 10 до 12:30")
    assert (result.at, result.end_at, result.duration_minutes) == (time(10), time(12, 30), 150)


def test_interval_afternoon_marker():
    result = p("завтра с 2 до 4 дня")
    assert (result.at, result.end_at) == (time(14), time(16))


@pytest.mark.parametrize(
    ("text", "minutes"),
    [("завтра в 10 на час", 60), ("в 12 на 30 минут", 30), ("в 9 на полтора часа", 90)],
)
def test_duration(text, minutes):
    assert p(text).duration_minutes == minutes


def test_duration_does_not_swallow_time():
    assert p("завтра в 10 на час").at == time(10)


def test_reminder_offset():
    assert p("завтра в 10 напомни за 15 минут").remind_before_minutes == 15
    assert p("за час до встречи").remind_before_minutes == 60


@pytest.mark.parametrize(
    ("text", "minutes"),
    [
        ("на час позже", 60),
        ("на 30 минут раньше", -30),
        ("на полчаса вперед", 30),
        ("на 2 часа пораньше", -120),
        ("на день позже", 1440),
    ],
)
def test_shift(text, minutes):
    assert p(text).shift_minutes == minutes, text


def test_past_bias_for_weekday():
    assert p("в четверг", past=True).day == date(2026, 10, 29)


def test_past_bias_for_date():
    assert p("15 декабря", past=True).day == date(2025, 12, 15)


def test_year_boundary():
    now = datetime(2026, 12, 30, 12, 0, tzinfo=MSK)
    assert p("15 января", now).day == date(2027, 1, 15)
    assert p("в следующую пятницу", now).day == date(2027, 1, 8)


def test_no_temporal_expression():
    assert p("купить молоко").empty


@pytest.mark.parametrize(
    ("text", "rest"),
    [
        ("Позвонить Ивану завтра в 10 утра", "Позвонить Ивану"),
        ("купить два билета до конца недели", "купить два билета"),
        ("в пятницу отправить отчет", "отправить отчет"),
        ("через два часа выпить таблетку", "выпить таблетку"),
        ("пятнадцатого ноября сдать проект", "сдать проект"),
    ],
)
def test_spans_preserve_original_words(text, rest):
    assert strip_spans(text, p(text).spans) == rest


def test_ordinal_in_prose_is_not_a_time():
    assert p("сделать в первый раз").at is None
