"""Диалоговый менеджер: превращает команды в действия и ведёт разговор.

Здесь живёт всё, что делает ассистента собеседником, а не парсером команд: уточняющие вопросы,
указание на предыдущие записи, разрешение неоднозначностей и конфликтов. Незавершённое действие
хранится в ``DialogState.pending`` и продолжается следующей репликой пользователя.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from orbita.dialog import format as fmt
from orbita.dialog.state import DialogState, draft_from_json, draft_to_json
from orbita.domain.models import Item, ItemKind, ItemStatus, Occurrence, SearchHit
from orbita.domain.recurrence import first_occurrence_on_or_after
from orbita.nlu.commands import (
    CREATE_ACTIONS,
    Action,
    Command,
    Interpreter,
    NluContext,
    Reference,
)
from orbita.nlu.morphology import lemma, words
from orbita.nlu.rules import RuleInterpreter, clean_utterance
from orbita.nlu.temporal import DateRange, Temporal, parse_temporal
from orbita.services.items import ItemDraft, ItemService
from orbita.services.search import SearchFilters, SearchService
from orbita.storage.repositories import DialogRepository

_RE_CANCEL = re.compile(
    r"^(?:нет\s+)?(?:отмена|отмени(?:ть)?|не\s+надо|не\s+нужно|забудь|забей|стоп|хватит"
    r"|ладно\s+не\s+надо|передумал\w*|никак\w*)\b"
)
_RE_YES = re.compile(r"^(?:да|ага|угу|давай|хорошо|ок|окей|конечно|верно|подходит|ладно|пойдет)\b")
_RE_NO = re.compile(r"^(?:нет|не|неа)\b")
_RE_KEEP = re.compile(
    r"\b(?:оставь|оставить|вс[её]\s+равно|в\s+любом\s+случае|пусть|ничего\s+страшного|как\s+есть"
    r"|не\s+страшно|на\s+то\s+же)\b"
)
_RE_ANY_TIME = re.compile(r"\b(?:не\s*важно|без\s+разницы|когда\s+угодно|любое|как\s+хочешь)\b")
_RE_CORRECTION = re.compile(r"^(?:нет|лучше|давай|а\s+лучше|не)\b")
_ORDINAL_LEMMAS = {
    "первый": 1, "второй": 2, "третий": 3, "четвёртый": 4, "четвертый": 4, "пятый": 5,
    "последний": -1, "один": 1, "два": 2, "три": 3,
}  # fmt: skip
_MAX_SPOKEN_ITEMS = 5
_MAX_ATTEMPTS = 2
_DEFAULT_LIST_DAYS = 7
_DEFAULT_REMINDER_TIME = time(9, 0)

_SUGGEST_IDLE = ["Что у меня сегодня?", "Какие планы на неделю?", "Напомни через час позвонить"]


@dataclass(slots=True)
class TurnResult:
    reply: str
    action: str
    items: list[Item] = field(default_factory=list)
    occurrences: list[Occurrence] = field(default_factory=list)
    awaiting: bool = False
    suggestions: list[str] = field(default_factory=list)
    nlu_source: str | None = None
    changed: bool = False


def _normalized(text: str) -> str:
    return clean_utterance(text).lower().replace("ё", "е")


class DialogManager:
    def __init__(
        self,
        interpreter: Interpreter,
        items: ItemService,
        search: SearchService,
        dialogs: DialogRepository,
        timezone: ZoneInfo,
    ) -> None:
        self._interpreter = interpreter
        self._rules = RuleInterpreter()
        self._items = items
        self._search = search
        self._dialogs = dialogs
        self._tz = timezone

    @property
    def _now(self) -> datetime:
        return self._items.now

    @property
    def _today(self) -> date:
        return self._items.today

    def handle(self, user_id: int, text: str) -> TurnResult:
        state = DialogState.from_json(self._dialogs.load(user_id))
        result = self._turn(user_id, state, text)
        self._dialogs.save(user_id, state.to_json(), self._now)
        return result

    def _turn(self, user_id: int, state: DialogState, text: str) -> TurnResult:
        if state.pending is not None:
            resumed = self._resume(user_id, state, text)
            if resumed is not None:
                resumed.nlu_source = "dialog"
                return resumed
        command = self._interpreter.interpret(text, self._context(user_id, state))
        result = self._dispatch(user_id, state, command)
        result.nlu_source = command.source
        return result

    def _context(self, user_id: int, state: DialogState) -> NluContext:
        focus = self._items.get_many(user_id, state.focus[:5])
        return NluContext(
            now=self._now,
            timezone=self._tz,
            focus_titles=tuple(item.title for item in focus),
            pending_question=(state.pending or {}).get("question"),
        )

    def _dispatch(self, user_id: int, state: DialogState, command: Command) -> TurnResult:
        action = command.action
        if action in CREATE_ACTIONS:
            return self._create(user_id, state, command)
        handlers = {
            Action.LIST: self._list,
            Action.SEARCH: self._find,
            Action.UPDATE: self._update,
            Action.DELETE: self._delete,
            Action.COMPLETE: self._complete,
            Action.UNDO: self._undo,
        }
        if handler := handlers.get(action):
            return handler(user_id, state, command)
        return self._conversation(user_id, state, command)

    def _conversation(self, user_id: int, state: DialogState, command: Command) -> TurnResult:
        if command.action is Action.GREET:
            return TurnResult(
                "Здравствуйте! Что запланируем?", "greet", suggestions=list(_SUGGEST_IDLE)
            )
        if command.action is Action.THANKS:
            return TurnResult("Пожалуйста. Обращайтесь!", "thanks")
        if command.action is Action.HELP:
            return TurnResult(
                "Я веду задачи, напоминания и встречи. Скажите, например: «напомни завтра в 10 "
                "позвонить Ивану», «что у меня на неделе» или «найди задачу про договор». Могу "
                "переносить, удалять и отмечать выполненным.",
                "help",
                suggestions=list(_SUGGEST_IDLE),
            )
        return self._follow_up(user_id, state, command)

    def _follow_up(self, user_id: int, state: DialogState, command: Command) -> TurnResult:
        """Реплика без глагола: «а на завтра?», «нет, лучше в 16»."""
        lowered = _normalized(command.text)
        when = command.when
        if not when.empty:
            if state.focus and _RE_CORRECTION.match(lowered):
                retarget = Command(
                    Action.UPDATE, text=command.text, when=when, reference=Reference(pronoun=True)
                )
                return self._update(user_id, state, retarget)
            if state.has_last_list:
                kind = ItemKind(state.last_list_kind) if state.last_list_kind else None
                return self._list(
                    user_id, state, Command(Action.LIST, text=command.text, when=when, kind=kind)
                )
        return TurnResult(
            "Не совсем поняла. Скажите, например, «напомни завтра в 10 позвонить Ивану» "
            "или «что у меня сегодня».",
            "unknown",
            suggestions=list(_SUGGEST_IDLE),
        )

    # Создание -----------------------------------------------------------------------------

    def _create(self, user_id: int, state: DialogState, command: Command) -> TurnResult:
        when = command.when
        kind = {
            Action.CREATE_MEETING: ItemKind.MEETING,
            Action.CREATE_REMINDER: ItemKind.REMINDER,
            Action.CREATE_TASK: ItemKind.TASK,
        }[command.action]
        draft = ItemDraft(
            kind=kind,
            title=command.title or "",
            day=when.day,
            at=when.at,
            duration_minutes=when.duration_minutes,
            deadline=when.deadline,
            recurrence=when.recurrence,
            remind_before_minutes=when.remind_before_minutes,
            source_text=command.text,
        )
        if when.period and draft.day is None and kind is not ItemKind.MEETING:
            # «На следующей неделе сходить к врачу» — точного дня нет, это срок.
            draft.day, draft.deadline = when.period.end, True
        if when.recurrence and draft.day is None:
            draft.day = self._today
        return self._finish_create(
            user_id,
            state,
            draft,
            assumed_pm=when.meridiem_assumed,
            period=when.period,
            day_inferred=when.day_inferred,
        )

    def _finish_create(
        self,
        user_id: int,
        state: DialogState,
        draft: ItemDraft,
        *,
        assumed_pm: bool = False,
        period: DateRange | None = None,
        day_inferred: bool = False,
        skip_conflict: bool = False,
    ) -> TurnResult:
        question = self._missing_slot(draft, period)
        if question is not None:
            slot, text = question
            payload: dict[str, Any] = {"type": "slot", "slot": slot, "question": text}
            payload["draft"] = draft_to_json(draft)
            state.pending = payload
            return TurnResult(text, "clarify", awaiting=True, suggestions=_slot_hints(slot))

        if self._in_past(draft) and not day_inferred:
            text = "Это время уже прошло. Поставить на завтра?"
            state.pending = {
                "type": "past",
                "question": text,
                "draft": draft_to_json(draft),
            }
            return TurnResult(text, "clarify", awaiting=True, suggestions=["Да", "Отмена"])

        if draft.kind is ItemKind.MEETING and draft.day and draft.at and not skip_conflict:
            conflict = self._items.find_conflict(
                user_id, draft.day, draft.at, draft.duration_minutes
            )
            if conflict:
                text = _conflict_question(conflict.occurrence, conflict.suggested_at)
                state.pending = {
                    "type": "conflict",
                    "question": text,
                    "draft": draft_to_json(draft),
                    "suggested": conflict.suggested_at.isoformat()
                    if conflict.suggested_at
                    else None,
                }
                hints = ["Оставь как есть", "Отмена"]
                if conflict.suggested_at:
                    hints.insert(0, f"Давай на {fmt.clock(conflict.suggested_at)}")
                return TurnResult(text, "conflict", awaiting=True, suggestions=hints)

        state.pending = None
        item = self._items.create(user_id, draft)
        state.focus = [item.id] if item.id else []
        state.has_last_list = False
        return TurnResult(
            self._created_text(item, assumed_pm),
            f"created_{item.kind.value}",
            items=[item],
            changed=True,
            suggestions=["Что у меня сегодня?"],
        )

    @staticmethod
    def _missing_slot(draft: ItemDraft, period: DateRange | None) -> tuple[str, str] | None:
        if draft.kind is ItemKind.MEETING:
            if draft.day is None:
                if period:
                    return ("day", "На какой день поставить встречу?")
                return ("day", "На какой день и время поставить встречу?")
            if draft.at is None:
                return ("time", "На какое время поставить встречу?")
            return None
        if not draft.title:
            if draft.kind is ItemKind.REMINDER:
                return ("title", "О чём напомнить?")
            return ("title", "Что нужно сделать?")
        if draft.kind is ItemKind.REMINDER:
            if draft.day is None and draft.at is None:
                return ("day", "Когда напомнить?")
            if draft.at is None:
                return ("time", "Во сколько напомнить?")
        return None

    def _in_past(self, draft: ItemDraft) -> bool:
        if draft.day is None or draft.recurrence:
            return False
        if draft.at is None:
            return draft.day < self._today
        return datetime.combine(draft.day, draft.at, tzinfo=self._tz) <= self._now

    def _created_text(self, item: Item, assumed_pm: bool) -> str:
        when = fmt.item_when(item, self._today)
        if item.kind is ItemKind.MEETING:
            text = f"Поставила: {item.title[:1].lower()}{item.title[1:]}, {when}."
            if item.title == "Встреча":
                text = f"Поставила встречу {when}."
        elif item.kind is ItemKind.REMINDER:
            text = f"Напомню {when}: {item.title}."
        elif item.day is None:
            text = f"Записала задачу: {item.title}."
        else:
            text = f"Записала задачу: {item.title}, {when}."
        if assumed_pm and item.at:
            text += f" Поняла как {fmt.clock(item.at)}, если нужно утро, скажите."
        return text

    # Продолжение незавершённых действий -----------------------------------------------------

    def _resume(self, user_id: int, state: DialogState, text: str) -> TurnResult | None:
        pending = state.pending
        assert pending is not None
        lowered = _normalized(text)
        if _RE_CANCEL.match(lowered):
            state.pending = None
            return TurnResult("Хорошо, отменила.", "cancelled")
        handler = {
            "slot": self._resume_slot,
            "past": self._resume_past,
            "conflict": self._resume_conflict,
            "choose": self._resume_choice,
            "bulk_delete": self._resume_bulk_delete,
            "move": self._resume_move,
        }[pending["type"]]
        return handler(user_id, state, text, lowered)

    def _is_new_command(self, text: str, when: Temporal) -> bool:
        """Реплика — самостоятельная команда, а не ответ на вопрос ассистента."""
        command = self._rules.interpret(text, NluContext(now=self._now, timezone=self._tz))
        if command.action in (Action.UNKNOWN, Action.GREET, Action.THANKS):
            return False
        if command.action in CREATE_ACTIONS:
            return bool(command.title) and command.confidence >= 0.85 and len(words(text)) > 2
        return command.confidence >= 0.85

    def _reask(self, state: DialogState, pending: dict[str, Any]) -> TurnResult | None:
        """Повторить вопрос или, если он не понят дважды, снять его."""
        attempts = int(pending.get("attempts", 0)) + 1
        if attempts >= _MAX_ATTEMPTS:
            state.pending = None
            return None
        pending["attempts"] = attempts
        return TurnResult(
            f"Не расслышала ответ. {pending['question']}",
            "clarify",
            awaiting=True,
            suggestions=["Отмена"],
        )

    def _resume_slot(
        self, user_id: int, state: DialogState, text: str, lowered: str
    ) -> TurnResult | None:
        pending = state.pending
        assert pending is not None
        draft = draft_from_json(pending["draft"])
        slot = pending["slot"]
        when = parse_temporal(text, self._now)
        if self._is_new_command(text, when):
            state.pending = None
            return None

        if slot == "title":
            draft.title = text.strip().rstrip(".!?")
            draft.title = draft.title[:1].upper() + draft.title[1:]
            return self._finish_create(user_id, state, draft)

        if slot == "time" and _RE_ANY_TIME.search(lowered):
            draft.at = _DEFAULT_REMINDER_TIME
            return self._finish_create(user_id, state, draft)

        if when.day is None and when.at is None:
            return self._reask(state, pending)

        draft.at = when.at or draft.at
        if slot == "day" or not when.day_inferred:
            draft.day = when.day or draft.day
        if draft.day is None and when.at is not None:
            draft.day = self._today
        # На вопрос «на какое время» день уже назван: «в 8» не должно уводить его на завтра.
        return self._finish_create(
            user_id,
            state,
            draft,
            assumed_pm=when.meridiem_assumed,
            day_inferred=slot == "day" and when.day_inferred,
        )

    def _resume_past(
        self, user_id: int, state: DialogState, text: str, lowered: str
    ) -> TurnResult | None:
        pending = state.pending
        assert pending is not None
        draft = draft_from_json(pending["draft"])
        when = parse_temporal(text, self._now)
        if self._is_new_command(text, when):
            state.pending = None
            return None
        if when.day or when.at:
            draft.day = when.day or draft.day
            draft.at = when.at or draft.at
            return self._finish_create(user_id, state, draft, day_inferred=when.day_inferred)
        if _RE_YES.match(lowered):
            draft.day = self._today + timedelta(days=1)
            return self._finish_create(user_id, state, draft)
        if _RE_NO.match(lowered):
            state.pending = None
            return TurnResult("Хорошо, не ставлю.", "cancelled")
        return self._reask(state, pending)

    def _resume_conflict(
        self, user_id: int, state: DialogState, text: str, lowered: str
    ) -> TurnResult | None:
        pending = state.pending
        assert pending is not None
        draft = draft_from_json(pending["draft"])
        target_id = pending.get("target")
        suggested = pending.get("suggested")
        when = parse_temporal(text, self._now)

        def commit(skip: bool) -> TurnResult:
            if target_id is None:
                return self._finish_create(user_id, state, draft, skip_conflict=skip)
            state.pending = None
            item = self._items.get(user_id, target_id)
            return self._apply_move(user_id, state, item, draft.day, draft.at, skip_conflict=skip)

        if _RE_KEEP.search(lowered):
            return commit(skip=True)
        if self._is_new_command(text, when):
            state.pending = None
            return None
        if when.at is not None or when.day is not None:
            draft.at = when.at or draft.at
            if not when.day_inferred:
                draft.day = when.day or draft.day
            return commit(skip=False)
        if _RE_YES.match(lowered):
            if suggested:
                draft.at = time.fromisoformat(suggested)
                return commit(skip=False)
            return commit(skip=True)
        if _RE_NO.match(lowered):
            state.pending = None
            return TurnResult("Хорошо, не ставлю.", "cancelled")
        return self._reask(state, pending)

    def _resume_move(
        self, user_id: int, state: DialogState, text: str, lowered: str
    ) -> TurnResult | None:
        pending = state.pending
        assert pending is not None
        when = parse_temporal(text, self._now)
        if when.empty:
            if self._is_new_command(text, when):
                state.pending = None
                return None
            return self._reask(state, pending)
        state.pending = None
        item = self._items.get(user_id, pending["target"])
        return self._move(user_id, state, item, when)

    def _resume_choice(
        self, user_id: int, state: DialogState, text: str, lowered: str
    ) -> TurnResult | None:
        pending = state.pending
        assert pending is not None
        options = self._items.get_many(user_id, pending["options"])
        chosen = _pick_option(lowered, options)
        if chosen is None:
            if self._is_new_command(text, Temporal()):
                state.pending = None
                return None
            return self._reask(state, pending)
        state.pending = None
        state.focus = [chosen.id] if chosen.id else []
        command = Command(
            Action(pending["action"]),
            text=pending.get("text", ""),
            when=parse_temporal(pending.get("text", ""), self._now),
            new_title=pending.get("new_title"),
        )
        return self._act_on(user_id, state, chosen, command)

    def _resume_bulk_delete(
        self, user_id: int, state: DialogState, text: str, lowered: str
    ) -> TurnResult | None:
        pending = state.pending
        assert pending is not None
        if _RE_YES.match(lowered):
            state.pending = None
            items = self._items.get_many(user_id, pending["options"])
            for item in items:
                self._items.delete(item)
            state.focus = []
            label = fmt.count_words(len(items), "запись", "записи", "записей")
            return TurnResult(f"Удалила {label}.", "deleted", changed=True)
        state.pending = None
        return TurnResult("Хорошо, ничего не удаляю.", "cancelled")

    # Просмотр и поиск -------------------------------------------------------------------------

    def _list(self, user_id: int, state: DialogState, command: Command) -> TurnResult:
        state.has_last_list = True
        state.last_list_kind = command.kind.value if command.kind else None
        if command.include_done:
            items = self._items.done(user_id, command.kind)
            state.focus = [item.id for item in items if item.id]
            return TurnResult(_done_text(items), "list", items=items)
        if command.only_overdue:
            items = self._items.overdue(user_id, command.kind)
            state.focus = [item.id for item in items if item.id]
            return TurnResult(_overdue_text(items, self._today), "list", items=items)

        period = self._list_period(command.when)
        agenda = self._items.agenda(user_id, period.start, period.end, kind=command.kind)
        ordered = list(agenda.overdue)
        seen = {item.id for item in ordered}
        for occurrence in agenda.occurrences:
            if occurrence.item.id not in seen:
                seen.add(occurrence.item.id)
                ordered.append(occurrence.item)
        state.focus = [item.id for item in ordered if item.id]
        text = _agenda_text(agenda.occurrences, agenda.overdue, period, command.kind, self._today)
        return TurnResult(
            text,
            "list",
            items=ordered,
            occurrences=agenda.occurrences,
            suggestions=_after_list_hints(period, self._today),
        )

    def _list_period(self, when: Temporal) -> DateRange:
        if when.period:
            return when.period
        if when.day:
            if when.deadline and when.day >= self._today:
                return DateRange(self._today, when.day)
            return DateRange(when.day, when.day)
        return DateRange(self._today, self._today + timedelta(days=_DEFAULT_LIST_DAYS - 1))

    def _find(self, user_id: int, state: DialogState, command: Command) -> TurnResult:
        query = (command.query or "").strip()
        when = command.when
        if not query and command.kind is None and when.empty:
            text = "Что найти?"
            return TurnResult(text, "clarify", suggestions=["Найди задачу про отчёт"])

        period = when.period or (DateRange(when.day, when.day) if when.day else None)
        filters = SearchFilters(
            kind=command.kind,
            period=period,
            before=self._today if command.past else None,
        )
        hits = self._search.search(user_id, query, filters)
        past_missing = False
        if not hits and command.past:
            hits = self._search.search(
                user_id, query, SearchFilters(kind=command.kind, period=period)
            )
            past_missing = bool(hits)
        items = [hit.item for hit in hits]
        state.focus = [item.id for item in items if item.id]
        state.has_last_list = False
        if not hits:
            subject = f"про «{query}»" if query else "по этому запросу"
            return TurnResult(
                f"Не нашла ничего {subject}.", "search", suggestions=list(_SUGGEST_IDLE)
            )
        return TurnResult(
            self._found_text(hits, command, past_missing),
            "search",
            items=items,
        )

    def _found_text(self, hits: list[SearchHit], command: Command, past_missing: bool) -> str:
        if len(hits) == 1:
            item = hits[0].item
            return self._single_found(item, command, past_missing)
        lines = [f"{hit.item.title} ({fmt.item_when(hit.item, self._today)})" for hit in hits]
        head = "Нашла " + fmt.count_words(len(hits), "запись", "записи", "записей")
        return f"{head}: " + "; ".join(lines) + "."

    def _single_found(self, item: Item, command: Command, past_missing: bool) -> str:
        when = fmt.item_when(self._upcoming(item), self._today)
        status = " Она уже выполнена." if item.status is ItemStatus.DONE else ""
        if command.past and item.day and item.day <= self._today and not item.recurrence:
            verb = "была" if item.kind is ItemKind.MEETING else "стояла"
            return f"{item.title} {verb} {when}.{status}"
        if past_missing:
            return f"Прошедших записей нет, ближайшая: {item.title}, {when}.{status}"
        return f"{item.title}: {when}.{status}"

    def _upcoming(self, item: Item) -> Item:
        if item.recurrence and item.day and item.day < self._today:
            following = first_occurrence_on_or_after(item, self._today)
            return item.copy(day=following or item.day)
        return item

    # Действия над существующими записями -------------------------------------------------------

    def _candidates(self, user_id: int, state: DialogState, command: Command) -> list[Item]:
        reference = command.reference or Reference(pronoun=True)
        focus = self._items.get_many(user_id, state.focus)
        if reference.ordinal is not None:
            index = reference.ordinal
            position = len(focus) - 1 if index == -1 else index - 1
            return [focus[position]] if 0 <= position < len(focus) else []
        if reference.pronoun or not reference.query:
            if command.kind is not None:
                focus = [item for item in focus if item.kind is command.kind] or focus
            return focus
        return self._lookup(user_id, reference.query, command.kind)

    def _lookup(self, user_id: int, query: str, kind: ItemKind | None) -> list[Item]:
        hits = self._search.search(user_id, query, SearchFilters(kind=kind))
        pending = [hit for hit in hits if hit.item.status is ItemStatus.PENDING]
        hits = pending or hits
        if len(hits) >= 2 and hits[0].score - hits[1].score >= 0.2:
            return [hits[0].item]
        return [hit.item for hit in hits]

    def _with_target(self, user_id: int, state: DialogState, command: Command) -> TurnResult:
        candidates = self._candidates(user_id, state, command)
        if not candidates:
            reference = command.reference
            if reference and reference.query:
                return TurnResult(f"Не нашла запись «{reference.query}».", "not_found")
            return TurnResult(
                "Уточните, какую запись вы имеете в виду. Например, «удали задачу про отчёт».",
                "not_found",
            )
        if len(candidates) > 1:
            return self._ask_choice(state, command, candidates)
        state.focus = [candidates[0].id] if candidates[0].id else []
        return self._act_on(user_id, state, candidates[0], command)

    def _ask_choice(self, state: DialogState, command: Command, items: list[Item]) -> TurnResult:
        items = items[:5]
        verb = {
            Action.DELETE: "удалить",
            Action.UPDATE: "изменить",
            Action.COMPLETE: "отметить",
        }[command.action]
        lines = [
            f"{index}) {item.title}, {fmt.item_when(item, self._today)}"
            for index, item in enumerate(items, 1)
        ]
        text = f"Какую запись {verb}? " + "; ".join(lines) + "."
        state.pending = {
            "type": "choose",
            "question": text,
            "action": command.action.value,
            "options": [item.id for item in items],
            "text": command.text,
            "new_title": command.new_title,
        }
        hints = ["Первую", "Вторую", "Отмена"]
        return TurnResult(text, "clarify", items=items, awaiting=True, suggestions=hints)

    def _act_on(self, user_id: int, state: DialogState, item: Item, command: Command) -> TurnResult:
        if command.action is Action.DELETE:
            self._items.delete(item)
            state.focus = []
            return TurnResult(
                f"Удалила: {item.title}. Если передумаете, скажите «отмени».",
                "deleted",
                changed=True,
                suggestions=["Отмени"],
            )
        if command.action is Action.COMPLETE:
            updated = self._items.complete(item)
            if updated.status is ItemStatus.DONE:
                text = f"Отметила выполненной: {item.title}."
            else:
                moment = fmt.item_when(updated.copy(recurrence=None), self._today)
                text = f"Отметила: {item.title}. Следующий раз {moment}."
            return TurnResult(text, "completed", items=[updated], changed=True)
        if command.new_title:
            updated = self._items.update(item, title=command.new_title)
            return TurnResult(
                f"Переименовала в «{updated.title}».", "updated", items=[updated], changed=True
            )
        return self._move(user_id, state, item, command.when)

    def _delete(self, user_id: int, state: DialogState, command: Command) -> TurnResult:
        when = command.when
        if command.bulk and (when.period or when.day):
            period = self._list_period(when)
            agenda = self._items.agenda(user_id, period.start, period.end, kind=command.kind)
            items = list({o.item.id: o.item for o in agenda.occurrences}.values())
            if not items:
                return TurnResult("Удалять нечего: на этот период записей нет.", "not_found")
            text = (
                f"Удалить {fmt.count_words(len(items), 'запись', 'записи', 'записей')} "
                f"{fmt.human_period(period, self._today)}?"
            )
            state.pending = {
                "type": "bulk_delete",
                "question": text,
                "options": [item.id for item in items],
            }
            return TurnResult(
                text, "clarify", items=items, awaiting=True, suggestions=["Да", "Нет"]
            )
        return self._with_target(user_id, state, command)

    def _complete(self, user_id: int, state: DialogState, command: Command) -> TurnResult:
        return self._with_target(user_id, state, command)

    def _update(self, user_id: int, state: DialogState, command: Command) -> TurnResult:
        return self._with_target(user_id, state, command)

    def _move(self, user_id: int, state: DialogState, item: Item, when: Temporal) -> TurnResult:
        if when.shift_minutes is not None:
            return self._shift(user_id, state, item, when.shift_minutes)
        if when.period and when.day is None:
            return self._apply_move(
                user_id, state, item, _same_weekday_in(item.day, when.period), item.at
            )
        if when.day is None and when.at is None:
            text = "На когда перенести?"
            state.pending = {
                "type": "move",
                "question": text,
                "target": item.id,
            }
            return TurnResult(
                text, "clarify", items=[item], awaiting=True, suggestions=["На завтра", "Отмена"]
            )
        # «Перенеси на 16:00» оставляет день: парсер подставил бы ближайший, а не день записи.
        day = item.day if when.day_inferred and item.day else when.day or item.day
        return self._apply_move(user_id, state, item, day, when.at or item.at)

    def _shift(self, user_id: int, state: DialogState, item: Item, minutes: int) -> TurnResult:
        if item.day is None:
            return TurnResult(
                f"У записи «{item.title}» нет даты, сдвигать нечего. "
                "Скажите, на когда её поставить.",
                "not_found",
            )
        moment = datetime.combine(item.day, item.at or time.min) + timedelta(minutes=minutes)
        return self._apply_move(
            user_id, state, item, moment.date(), moment.time() if item.at else None
        )

    def _apply_move(
        self,
        user_id: int,
        state: DialogState,
        item: Item,
        day: date | None,
        at: time | None,
        *,
        skip_conflict: bool = False,
    ) -> TurnResult:
        if item.kind is ItemKind.MEETING and day and at and not skip_conflict:
            conflict = self._items.find_conflict(user_id, day, at, item.duration_minutes, item.id)
            if conflict:
                text = _conflict_question(conflict.occurrence, conflict.suggested_at)
                draft = ItemDraft(kind=item.kind, title=item.title, day=day, at=at)
                state.pending = {
                    "type": "conflict",
                    "question": text,
                    "draft": draft_to_json(draft),
                    "target": item.id,
                    "suggested": conflict.suggested_at.isoformat()
                    if conflict.suggested_at
                    else None,
                }
                hints = ["Оставь как есть", "Отмена"]
                if conflict.suggested_at:
                    hints.insert(0, f"Давай на {fmt.clock(conflict.suggested_at)}")
                return TurnResult(text, "conflict", items=[item], awaiting=True, suggestions=hints)
        updated = self._items.update(item, day=day, at=at)
        state.focus = [updated.id] if updated.id else []
        note = " Перенесла всю серию." if updated.recurrence else ""
        when = fmt.item_when(updated.copy(recurrence=None), self._today)
        return TurnResult(
            f"Перенесла: {updated.title}, {when}.{note}",
            "updated",
            items=[updated],
            changed=True,
            suggestions=["Отмени"],
        )

    def _undo(self, user_id: int, state: DialogState, command: Command) -> TurnResult:
        result = self._items.undo(user_id)
        if result is None:
            return TurnResult("Отменять пока нечего.", "undo")
        item, action = result
        text = {
            "create": f"Отменила создание: {item.title}.",
            "delete": f"Вернула: {item.title}.",
            "update": f"Вернула как было: {item.title}.",
        }[action]
        state.focus = [] if action == "create" or item.id is None else [item.id]
        items = [] if action == "create" else [item]
        return TurnResult(text, "undo", items=items, changed=True)


def _conflict_question(occurrence: Occurrence, suggested: time | None) -> str:
    at = fmt.clock(occurrence.at) if occurrence.at else "это время"
    text = f"В {at} уже стоит: {occurrence.item.title}."
    if suggested:
        return f"{text} Поставить на {fmt.clock(suggested)} или оставить как есть?"
    return f"{text} Всё равно поставить?"


def _slot_hints(slot: str) -> list[str]:
    return {
        "time": ["В 10:00", "В 15:00", "Отмена"],
        "day": ["Завтра", "В понедельник", "Отмена"],
    }.get(slot, ["Отмена"])


def _pick_option(lowered: str, options: list[Item]) -> Item | None:
    tokens = words(lowered)
    for token in tokens:
        number = _ORDINAL_LEMMAS.get(lemma(token))
        if number is not None:
            position = len(options) - 1 if number == -1 else number - 1
            return options[position] if 0 <= position < len(options) else None
        if token.isdigit() and 0 < int(token) <= len(options):
            return options[int(token) - 1]
    answer = {lemma(token) for token in tokens if len(token) > 2}
    scored = [(len(answer & {lemma(t) for t in words(item.title)}), item) for item in options]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    if scored and scored[0][0] and (len(scored) == 1 or scored[0][0] > scored[1][0]):
        return scored[0][1]
    return None


def _same_weekday_in(day: date | None, period: DateRange) -> date:
    if day is None:
        return period.start
    offset = (day.weekday() - period.start.weekday()) % 7
    return min(period.start + timedelta(days=offset), period.end)


def _subject(kind: ItemKind | None) -> tuple[str, str, str]:
    return {
        ItemKind.MEETING: ("встреча", "встречи", "встреч"),
        ItemKind.TASK: ("задача", "задачи", "задач"),
        ItemKind.REMINDER: ("напоминание", "напоминания", "напоминаний"),
        None: ("запись", "записи", "записей"),
    }[kind]


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]


def _agenda_text(
    occurrences: list[Occurrence],
    overdue: list[Item],
    period: DateRange,
    kind: ItemKind | None,
    today: date,
) -> str:
    label = fmt.human_period(period, today)
    subject = _subject(kind)
    single_day = period.start == period.end
    if not occurrences:
        empty = (
            f"{_capitalize(label)} {subject[2]} нет."
            if kind
            else f"{_capitalize(label)} планов нет."
        )
        return f"{empty} {_overdue_note(overdue)}".strip()

    lines = [
        fmt.occurrence_line(occurrence, not single_day, today)
        for occurrence in occurrences[:_MAX_SPOKEN_ITEMS]
    ]
    count = fmt.count_words(len(occurrences), *subject)
    head = f"{_capitalize(label)}, {count}: " if "," in label else f"{_capitalize(label)} {count}: "
    text = head + "; ".join(lines)
    rest = len(occurrences) - _MAX_SPOKEN_ITEMS
    if rest > 0:
        text += f" и ещё {rest}"
    return f"{text}. {_overdue_note(overdue)}".strip()


def _overdue_note(overdue: list[Item]) -> str:
    if not overdue:
        return ""
    if len(overdue) == 1:
        return f"Ещё просрочено: {overdue[0].title}."
    return f"Ещё просрочено: {fmt.count_words(len(overdue), 'запись', 'записи', 'записей')}."


def _overdue_text(items: list[Item], today: date) -> str:
    if not items:
        return "Просроченных задач нет."
    lines = [f"{item.title} ({fmt.item_when(item, today)})" for item in items[:_MAX_SPOKEN_ITEMS]]
    head = fmt.count_words(
        len(items), "просроченная запись", "просроченные записи", "просроченных записей"
    )
    return f"{_capitalize(head)}: " + "; ".join(lines) + "."


def _done_text(items: list[Item]) -> str:
    if not items:
        return "Выполненных записей пока нет."
    lines = [item.title for item in items[:_MAX_SPOKEN_ITEMS]]
    head = fmt.count_words(
        len(items), "выполненная запись", "выполненные записи", "выполненных записей"
    )
    return f"{_capitalize(head)}: " + "; ".join(lines) + "."


def _after_list_hints(period: DateRange, today: date) -> list[str]:
    if period.start == today:
        return ["А на завтра?", "Какие планы на неделю?"]
    return ["Что у меня сегодня?"]
