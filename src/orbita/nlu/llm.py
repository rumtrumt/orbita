"""Интерпретатор на LLM.

Модель извлекает смысл: действие, название, предмет поиска — и дословно цитирует выражение
времени. Дату по этой цитате вычисляет parse_temporal, поэтому модель не может ошибиться в
календаре, а её ответ не требует проверки арифметики.
"""

from __future__ import annotations

import re
from typing import Any

from orbita.domain.models import ItemKind
from orbita.llm.client import LlmClient, LlmError
from orbita.nlu.commands import Action, Command, NluContext, Reference
from orbita.nlu.morphology import words
from orbita.nlu.rules import clean_utterance
from orbita.nlu.temporal import parse_temporal, strip_spans

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": [action.value for action in Action]},
        "title": {"type": ["string", "null"]},
        "when": {"type": ["string", "null"]},
        "query": {"type": ["string", "null"]},
        "kind": {"type": ["string", "null"], "enum": ["task", "reminder", "meeting", None]},
        "refers_to_previous": {"type": "boolean"},
        "ordinal": {"type": ["integer", "null"]},
        "new_title": {"type": ["string", "null"]},
        "past": {"type": "boolean"},
    },
    "required": [
        "action", "title", "when", "query", "kind", "refers_to_previous", "ordinal", "new_title",
        "past",
    ],
}  # fmt: skip

_SYSTEM = """\
Ты — модуль понимания речи голосового ассистента для задач и встреч. Определи намерение реплики и \
верни JSON. Не отвечай пользователю и не вычисляй даты.

action:
- create_task — дело, которое нужно сделать ("в пятницу отправить отчёт", "надо купить билеты")
- create_reminder — просьба напомнить ("напомни завтра позвонить")
- create_meeting — событие с другими людьми или визит ("в 11 встреча с дизайнером", "завтра стоматолог в 10")
- list — вопрос о планах за период ("что у меня сегодня", "я завтра свободен?", "как у меня со временем в среду")
- search — поиск конкретной записи по содержанию ("найди задачу про договор", "когда встреча с Иваном?", \
"была какая-то задача с налогами"). Вопрос о том, что пользователь записывал или о чём договаривался \
раньше, всегда search, даже со словом "напомни" или "созвониться"
- update — просьба перенести или переименовать уже существующую запись ("с Иваном давай в четверг")
- delete — убрать или отменить существующую запись ("снимай встречу", "про билеты можно забыть")
- complete — пользователь сообщает, что дело сделано
- undo — отменить последнее действие ассистента, без указания записи
- greet, thanks, help — приветствие, благодарность, вопрос о возможностях
- unknown — не относится к задачам и календарю

Поля:
- title: суть записи без слов-команд и без времени, с заглавной буквы. Для встречи: "Встреча с Иваном".
- when: выражение времени ДОСЛОВНО из реплики, например "завтра в 10 утра", "до конца недели", "каждую пятницу". Нет времени — null.
- query: что искать, только значимые слова без "найди", "задача", "встреча". Для search.
- kind: тип записи, если пользователь его назвал; иначе null.
- refers_to_previous: true, если сказано "её", "это", "ту задачу" либо объект не назван.
- ordinal: номер из списка: "вторую" → 2, "последнюю" → -1.
- new_title: новое название при переименовании.
- past: true, если речь о прошлом ("когда была встреча").

Примеры:
"завтра стоматолог в 10" → {"action":"create_meeting","title":"Стоматолог","when":"завтра в 10","query":null,"kind":"meeting","refers_to_previous":false,"ordinal":null,"new_title":null,"past":false}
"надо бы маме цветы купить к восьмому марта" → {"action":"create_task","title":"Купить маме цветы","when":"к восьмому марта","query":null,"kind":"task","refers_to_previous":false,"ordinal":null,"new_title":null,"past":false}
"что там по документам на подпись" → {"action":"search","title":null,"when":null,"query":"документы на подпись","kind":null,"refers_to_previous":false,"ordinal":null,"new_title":null,"past":false}
"сдвинь это на час позже" → {"action":"update","title":null,"when":"на час позже","query":null,"kind":null,"refers_to_previous":true,"ordinal":null,"new_title":null,"past":false}
"мы с Олегом вообще созванивались в этом месяце" → {"action":"search","title":null,"when":"в этом месяце","query":"Олег","kind":"meeting","refers_to_previous":false,"ordinal":null,"new_title":null,"past":true}
"забрал документы из банка" → {"action":"complete","title":null,"when":null,"query":"забрать документы из банка","kind":null,"refers_to_previous":false,"ordinal":null,"new_title":null,"past":false}
"послезавтра к десяти в паспортный стол" → {"action":"create_meeting","title":"Паспортный стол","when":"послезавтра к десяти","query":null,"kind":"meeting","refers_to_previous":false,"ordinal":null,"new_title":null,"past":false}
"какая погода" → {"action":"unknown","title":null,"when":null,"query":null,"kind":null,"refers_to_previous":false,"ordinal":null,"new_title":null,"past":false}
"""


# Переименование легко спутать с новой записью, поэтому нужен явный признак в реплике.
_RE_RENAME = re.compile(r"переимен|назов|назвать|название", re.IGNORECASE)


class LlmInterpreter:
    def __init__(self, client: LlmClient) -> None:
        self._client = client

    def interpret(self, text: str, context: NluContext) -> Command:
        cleaned = clean_utterance(text)
        user = self._prompt(cleaned, context)
        data = self._client.complete_json(_SYSTEM, user, _SCHEMA, "command")
        return self._to_command(text, cleaned, data, context)

    @staticmethod
    def _prompt(text: str, context: NluContext) -> str:
        lines = []
        if context.focus_titles:
            lines.append("Последние обсуждаемые записи: " + "; ".join(context.focus_titles))
        lines.append(f"Реплика: {text}")
        return "\n".join(lines)

    @staticmethod
    def _to_command(text: str, cleaned: str, data: dict[str, Any], context: NluContext) -> Command:
        try:
            action = Action(data["action"])
            kind = ItemKind(data["kind"]) if data.get("kind") else None
        except (KeyError, ValueError) as error:
            raise LlmError(f"Недопустимое значение в ответе: {data}") from error

        past = bool(data.get("past"))
        when_text = data.get("when") or ""
        # Выражение времени, которого нет в реплике, считаем выдумкой модели.
        quoted = when_text and set(words(when_text)) <= set(words(cleaned))
        # Модель часто теряет время в вопросах, поэтому без цитаты разбираем всю реплику.
        when = parse_temporal(when_text if quoted else cleaned, context.now, past=past)

        reference = None
        if action in (Action.UPDATE, Action.DELETE, Action.COMPLETE):
            ordinal = data.get("ordinal")
            query = (data.get("query") or data.get("title") or "").strip()
            if ordinal:
                reference = Reference(ordinal=ordinal)
            elif query:
                # Названный объект важнее флага: «созвон с клиентом на час позже».
                reference = Reference(query=query)
            else:
                reference = Reference(pronoun=True)

        title = (data.get("title") or "").strip() or None
        if title:
            # Модель иногда оставляет время в названии: «Делать зарядку в 8 утра».
            title = strip_spans(title, parse_temporal(title, context.now).spans) or title
            title = title[:1].upper() + title[1:]
        return Command(
            action=action,
            text=text,
            title=title,
            query=(data.get("query") or "").strip() or None,
            when=when,
            kind=kind,
            reference=reference,
            new_title=data.get("new_title") if _RE_RENAME.search(cleaned) else None,
            past=past,
            confidence=0.9,
            source="llm",
        )
