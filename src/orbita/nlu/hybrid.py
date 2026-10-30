"""Гибридный интерпретатор: правила отвечают сразу, LLM подключается при сомнениях.

Уверенно распознанная реплика не тратит секунды на модель, а её недоступность не ломает
обязательные сценарии: ассистент деградирует до правил, а не падает.
"""

from __future__ import annotations

import time

import structlog

from orbita.llm.client import LlmError
from orbita.nlu.commands import Command, Interpreter, NluContext
from orbita.nlu.morphology import words
from orbita.nlu.rules import clean_utterance, points_at_record
from orbita.nlu.temporal import strip_spans

log = structlog.get_logger(__name__)

_CONFIDENT = 0.85
_COOLDOWN_SECONDS = 60.0


class HybridInterpreter:
    def __init__(self, rules: Interpreter, llm: Interpreter) -> None:
        self._rules = rules
        self._llm = llm
        self._llm_down_until = 0.0

    def interpret(self, text: str, context: NluContext) -> Command:
        fallback = self._rules.interpret(text, context)
        if fallback.confidence >= _CONFIDENT or time.monotonic() < self._llm_down_until:
            return fallback
        # Эллипсис вроде «а на завтра?» разбирает диалог, модели тут нечего добавить.
        remainder = strip_spans(clean_utterance(text), fallback.when.spans)
        if not fallback.when.empty and not [w for w in words(remainder) if len(w) > 2]:
            return fallback
        try:
            command = self._llm.interpret(text, context)
        except LlmError as error:
            # Иначе каждая реплика ждала бы полный таймаут недоступной модели.
            self._llm_down_until = time.monotonic() + _COOLDOWN_SECONDS
            log.warning("nlu.llm_unavailable", error=str(error))
            return fallback
        if self._touches_focus_without_cause(command, text):
            log.info("nlu.llm_overruled", action=command.action.value)
            return fallback
        if command.when.empty:
            command.when = fallback.when
        return command

    @staticmethod
    def _touches_focus_without_cause(command: Command, text: str) -> bool:
        """Модель меняет запись из контекста, хотя в реплике на неё ничто не указывает.

        Такая ошибка хуже непонимания: она молча портит чужие данные. Новая фраза сразу после
        создания записи не должна её переименовывать.
        """
        reference = command.reference
        return bool(reference and reference.pronoun and not points_at_record(text))


class LlmFirstInterpreter:
    """Режим ``llm``: модель решает всё, правила остаются запасным вариантом при сбое."""

    def __init__(self, rules: Interpreter, llm: Interpreter) -> None:
        self._rules = rules
        self._llm = llm

    def interpret(self, text: str, context: NluContext) -> Command:
        try:
            return self._llm.interpret(text, context)
        except LlmError as error:
            log.warning("nlu.llm_unavailable", error=str(error))
            return self._rules.interpret(text, context)
