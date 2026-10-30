"""Очистка текста распознавания и новые правила понимания."""

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pytest

from orbita.nlu.commands import Action, NluContext
from orbita.nlu.rules import RuleInterpreter
from orbita.nlu.temporal import parse_temporal
from orbita.voice.transcript import polish

MSK = ZoneInfo("Europe/Moscow")
NOW = datetime(2026, 10, 30, 14, 0, tzinfo=MSK)
CONTEXT = NluContext(now=NOW, timezone=MSK)


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("Запиши «Купить молоко».", "Запиши Купить молоко"),
        ("Надо добавить. Соль в суп.", "Надо добавить Соль в суп"),
        ("Запланируй созван с заказчиком", "Запланируй созвон с заказчиком"),
        ("Напомню, через два часа выпить таблетку.", "Напомни, через два часа выпить таблетку"),
        ("Переименуют задачу", "Переименуй задачу"),
        ("В пятнадцать нол нол.", "В пятнадцать ноль ноль"),
        ("Встреча в 15.00.", "Встреча в 15.00"),
        ("Завтра в 19. Ужин с Машей.", "Завтра в 19 Ужин с Машей"),
    ],
)
def test_polish(raw, clean):
    assert polish(raw) == clean


def interpret(text):
    return RuleInterpreter().interpret(text, CONTEXT)


def test_wake_up_hour_stays_in_the_morning():
    assert interpret("разбуди меня завтра в семь").when.at == time(7)


def test_ten_in_the_afternoon_means_tomorrow_morning():
    result = parse_temporal("в 10", NOW)
    assert (result.day, result.at) == (date(2026, 10, 31), time(10))


def test_eight_in_the_evening_still_rolls_to_tonight():
    evening = datetime(2026, 10, 30, 18, 30, tzinfo=MSK)
    assert parse_temporal("в 8", evening).at == time(20)


@pytest.mark.parametrize("text", ["посылку забрал", "отчёт я отправил", "купил корм"])
def test_reported_result_completes_task(text):
    assert interpret(text).action is Action.COMPLETE


@pytest.mark.parametrize(
    "text", ["я забыл про отчёт", "встреча была вчера", "когда я купил билеты"]
)
def test_past_tense_alone_does_not_complete(text):
    assert interpret(text).action is not Action.COMPLETE


def test_shift_of_named_meeting_is_update():
    assert interpret("давай созвон с клиентом на час позже").action is Action.UPDATE


def test_undo_variations():
    assert interpret("Отмени последние действия").action is Action.UNDO
    assert interpret("передумал, отмени последнее").action is Action.UNDO


def test_wish_to_look_at_task_is_search():
    assert interpret("хотел посмотреть задачу про аренду офиса").action is Action.SEARCH
