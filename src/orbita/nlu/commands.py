"""Структурированная команда — общий язык между NLU и диалоговым менеджером."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol
from zoneinfo import ZoneInfo

from orbita.domain.models import ItemKind
from orbita.nlu.temporal import Temporal


class Action(StrEnum):
    CREATE_TASK = "create_task"
    CREATE_REMINDER = "create_reminder"
    CREATE_MEETING = "create_meeting"
    LIST = "list"
    SEARCH = "search"
    UPDATE = "update"
    DELETE = "delete"
    COMPLETE = "complete"
    UNDO = "undo"
    GREET = "greet"
    THANKS = "thanks"
    HELP = "help"
    UNKNOWN = "unknown"


CREATE_ACTIONS = frozenset({Action.CREATE_TASK, Action.CREATE_REMINDER, Action.CREATE_MEETING})


@dataclass(slots=True)
class Reference:
    """На какую запись указывает пользователь."""

    pronoun: bool = False  # «её», «эту задачу», либо объект не назван вовсе
    ordinal: int | None = None  # «вторую»
    query: str | None = None  # «встречу с Иваном»


@dataclass(slots=True)
class Command:
    action: Action
    text: str = ""
    title: str | None = None
    query: str | None = None
    when: Temporal = field(default_factory=Temporal)
    kind: ItemKind | None = None
    reference: Reference | None = None
    new_title: str | None = None
    include_done: bool = False
    only_overdue: bool = False
    bulk: bool = False
    past: bool = False
    confidence: float = 1.0
    source: str = "rules"


@dataclass(frozen=True, slots=True)
class NluContext:
    """Что интерпретатор знает о мире на момент реплики."""

    now: datetime
    timezone: ZoneInfo
    focus_titles: tuple[str, ...] = ()
    pending_question: str | None = None


class Interpreter(Protocol):
    def interpret(self, text: str, context: NluContext) -> Command:
        """Превратить реплику пользователя в команду."""
        ...
