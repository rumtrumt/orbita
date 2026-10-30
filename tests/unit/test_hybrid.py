"""Гибридный интерпретатор: когда верить LLM, а когда нет."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from orbita.llm.client import LlmError
from orbita.nlu.commands import Action, Command, NluContext, Reference
from orbita.nlu.hybrid import HybridInterpreter
from orbita.nlu.llm import LlmInterpreter
from orbita.nlu.rules import RuleInterpreter

MSK = ZoneInfo("Europe/Moscow")
NOW = datetime(2026, 10, 30, 14, 0, tzinfo=MSK)
CONTEXT = NluContext(now=NOW, timezone=MSK, focus_titles=("Полить цветы",))


class StubLlm:
    def __init__(self, command=None, error=False):
        self.command = command
        self.error = error
        self.calls = 0

    def interpret(self, text, context):
        self.calls += 1
        if self.error:
            raise LlmError("нет связи")
        return self.command


def hybrid(llm):
    return HybridInterpreter(RuleInterpreter(), llm)


def test_confident_rules_skip_llm():
    llm = StubLlm()
    command = hybrid(llm).interpret("Напомни завтра в 10 позвонить маме", CONTEXT)
    assert command.action is Action.CREATE_REMINDER
    assert llm.calls == 0


def test_task_with_leading_time_does_not_need_llm():
    llm = StubLlm()
    command = hybrid(llm).interpret("Завтра в 9 утра купить билеты", CONTEXT)
    assert command.action is Action.CREATE_TASK
    assert command.title == "Купить билеты"
    assert llm.calls == 0


def test_llm_decides_uncertain_phrase():
    answer = Command(Action.CREATE_MEETING, title="Стоматолог", source="llm")
    command = hybrid(StubLlm(answer)).interpret("завтра стоматолог в 10", CONTEXT)
    assert command.title == "Стоматолог"
    assert command.when.at is not None


def test_llm_cannot_change_focus_item_without_reference_in_phrase():
    answer = Command(
        Action.UPDATE, new_title="Купить", reference=Reference(pronoun=True), source="llm"
    )
    command = hybrid(StubLlm(answer)).interpret("Завтра стоматолог", CONTEXT)
    assert command.action is not Action.UPDATE


def test_llm_can_change_focus_item_when_phrase_points_to_it():
    answer = Command(Action.UPDATE, reference=Reference(pronoun=True), source="llm")
    command = hybrid(StubLlm(answer)).interpret("эту задачу на потом, на понедельник", CONTEXT)
    assert command.action is Action.UPDATE


def test_llm_unknown_beats_weak_guess():
    command = hybrid(StubLlm(Command(Action.UNKNOWN, source="llm"))).interpret(
        "какая сегодня погода", CONTEXT
    )
    assert command.action is Action.UNKNOWN


def test_llm_failure_falls_back_and_cools_down():
    llm = StubLlm(error=True)
    interpreter = hybrid(llm)
    assert interpreter.interpret("завтра стоматолог в 10", CONTEXT).action is not None
    interpreter.interpret("завтра окулист в 11", CONTEXT)
    assert llm.calls == 1


def test_ellipsis_stays_with_dialog():
    llm = StubLlm()
    assert hybrid(llm).interpret("а на завтра?", CONTEXT).action is Action.UNKNOWN
    assert llm.calls == 0


class FakeClient:
    def __init__(self, payload):
        self.payload = payload

    def complete_json(self, *args, **kwargs):
        return self.payload


BASE = {
    "action": "create_task",
    "title": "Купить билеты",
    "when": None,
    "query": None,
    "kind": None,
    "refers_to_previous": False,
    "ordinal": None,
    "new_title": None,
    "past": False,
}


def test_invented_time_is_ignored():
    client = FakeClient(BASE | {"when": "в пятницу в 18:00"})
    command = LlmInterpreter(client).interpret("купить билеты", CONTEXT)
    assert command.when.empty


def test_lost_time_is_recovered_from_phrase():
    command = LlmInterpreter(FakeClient(BASE)).interpret("завтра купить билеты", CONTEXT)
    assert command.when.day is not None


def test_wake_up_hour_stays_morning_with_llm_quote():
    client = FakeClient(BASE | {"action": "create_reminder", "when": "завтра в семь"})
    command = LlmInterpreter(client).interpret("разбуди меня завтра в семь", CONTEXT)
    assert command.when.at.hour == 7


def test_new_title_requires_rename_request():
    client = FakeClient(BASE | {"action": "update", "new_title": "Купить билеты"})
    command = LlmInterpreter(client).interpret("перенеси её", CONTEXT)
    assert command.new_title is None


def test_rename_keeps_new_title():
    client = FakeClient(BASE | {"action": "update", "new_title": "Отчёт"})
    command = LlmInterpreter(client).interpret("переименуй её в Отчёт", CONTEXT)
    assert command.new_title == "Отчёт"


def test_invalid_action_is_an_error():
    with pytest.raises(LlmError):
        LlmInterpreter(FakeClient(BASE | {"action": "взломать"})).interpret("тест", CONTEXT)
