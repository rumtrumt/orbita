"""Интерпретатор на правилах.

Работает без сети и GPU, отвечает за миллисекунды и детерминирован, поэтому закрывает все
обязательные сценарии сам. LLM подключается поверх него для формулировок, которые правила
не уверенно распознают (см. nlu/hybrid.py). Решения принимаются по леммам: так одно правило
покрывает «поставь», «поставьте» и «поставить».
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from orbita.domain.models import ItemKind
from orbita.nlu.commands import Action, Command, NluContext, Reference
from orbita.nlu.morphology import analyzer, lemma, words
from orbita.nlu.temporal import Temporal, parse_temporal, strip_spans

_WAKE = re.compile(r"^\s*(?:эй|ок|окей|слушай)?[,\s]*орбита[,\s]+", re.IGNORECASE)
_FILLERS = re.compile(
    r"\b(?:пожалуйста|пожалуй|слушай|послушай|ну|короче|кстати|будь добр\w*|будьте добры)\b[,]?",
    re.IGNORECASE,
)

_MEETING_LEMMAS = {
    "встреча", "созвон", "совещание", "собеседование", "митинг", "планёрка", "планерка",
    "интервью", "переговоры", "стендап", "конференция", "звонок", "приём", "прием", "обед",
    "ужин", "собрание", "презентация", "демо", "ретро", "синк",
}  # fmt: skip
# Сами по себе не делают запись встречей: «подготовить презентацию» — задача.
_WEAK_MEETING_LEMMAS = {"презентация", "звонок", "обед", "ужин", "приём", "прием", "демо"}
_MEETING_VERBS = {"встречаться", "встретиться", "созвониться", "созваниваться", "увидеться"}
_CREATE_LEMMAS = {
    "поставить", "добавить", "создать", "запланировать", "назначить", "записать", "занести",
    "внести", "запиши", "организовать",
}  # fmt: skip
_NEED_LEMMAS = {"нужно", "надо", "необходимо", "должен", "нужен", "хотеть", "требоваться", "пора"}
_TASK_NOUNS = {"задача", "задачка", "дело", "таск", "пункт", "туду"}
_ANY_ITEM_NOUNS = _TASK_NOUNS | {"встреча", "напоминание", "запись", "событие", "мероприятие"}
_LIST_NOUNS = {
    "план", "расписание", "календарь", "дело", "задача", "встреча", "напоминание", "событие",
    "повестка", "список", "мероприятие", "запись",
}  # fmt: skip
_SHOW_LEMMAS = {
    "показать",
    "покажи",
    "рассказать",
    "прочитать",
    "перечислить",
    "открыть",
    "вывести",
}
_DELETE_LEMMAS = {"удалить", "убрать", "стереть", "снести", "вычеркнуть", "выкинуть", "удаль"}
_UPDATE_LEMMAS = {
    "перенести", "передвинуть", "сдвинуть", "подвинуть", "переставить", "перепланировать",
    "поменять", "изменить", "сменить", "переименовать", "исправить", "заменить", "отложить",
    "перекинуть",
}  # fmt: skip
_DONE_LEMMAS = {"выполнить", "сделать", "закончить", "завершить", "закрыть", "готовый"}
_REFERENCE_WORDS = {"ее", "её", "его", "их", "эту", "эта", "это", "этот", "ту", "тот", "последнюю"}
_ORDINALS = {
    "первый": 1, "второй": 2, "третий": 3, "четвертый": 4, "четвёртый": 4, "пятый": 5,
    "последний": -1,
}  # fmt: skip

_RE_PAST = re.compile(
    r"\b(был|была|были|было|ставил\w*|создавал\w*|добавлял\w*|записывал\w*|планировал\w*)\b"
)
_RE_UNDO = re.compile(
    r"^(?:отмени(?:ть)?|откати(?:ть)?|верни(?:те)?)(?:\s+(?:это|последнее|как было|назад|обратно"
    r"|действие|последнее действие|изменения?|все как было))*\s*$"
)
_RE_GREET = re.compile(r"^(?:привет\w*|здравствуй\w*|добр\w+ (?:утро|день|вечер)|хай|салют)\b")
_RE_THANKS = re.compile(r"^(?:спасибо|благодарю|спс|отлично спасибо|супер)\b")
_RE_HELP = re.compile(r"\b(?:что ты умеешь|как тобой пользоваться|что ты можешь|помощь|справка)\b")
_RE_LIST_QUESTION = re.compile(
    r"\b(?:что\s+(?:у\s+меня|там|сегодня|завтра|по\s+план\w+|запланирован\w*|намечен\w*"
    r"|предстоит|на\s+\S+)|как\w+\s+(?:план\w*|дел\w*|задач\w*|встреч\w*|напоминан\w*)"
    r"|чем\s+я\s+занят\w*|я(?:\s+\S+)?\s+(?:свобод\w+|занят\w*)|мо[ие]\s+(?:план\w*|дел\w*|задач\w*"
    r"|встреч\w*)|на\s+сегодня\s+план\w*|есть\s+(?:ли\s+)?(?:у\s+меня\s+)?(?:какие-нибудь\s+)?"
    r"(?:что-нибудь|что-то|план\w*|дел\w*)|сколько\s+(?:у\s+меня\s+)?(?:задач|встреч|дел))"
)
_RE_EXISTS = re.compile(r"\b(?:есть\s+ли|есть\s+у\s+меня|будет\s+ли|будут\s+ли)\b")
_RE_ABOUT = re.compile(r"\b(?:про|насчет|касательно|по\s+поводу|связанн\w+\s+с)\b")
_RE_SEARCH_VERB = re.compile(
    r"\b(?:найди\w*|найти|поищи\w*|ищи|искать|отыщи|разыщи|вспомни"
    r"|напомни,?\s+(?:когда|что\s+я|про\s+что)"
    r"|где\s+(?:у\s+меня\s+)?(?:задач|встреч|запис))"
)
_RE_WHEN_QUESTION = re.compile(r"^(?:а\s+)?когда\b")
_RE_OVERDUE = re.compile(r"\bпросроч\w+|\bне\s+сделал\w*|\bхвост\w*")
_RE_DONE_FILTER = re.compile(r"\b(?:выполненн\w+|сделанн\w+|завершенн\w+|закрыт\w+)\b")
_RE_BULK = re.compile(r"\bвсе\b|\bвсю\b")
_RE_ALREADY_DONE = re.compile(r"^(?:а\s+)?(?:я\s+)?уже\s+(\S+)(.*)$")
_RE_MARK_DONE = re.compile(
    r"\b(?:отмет\w+|пометь|пометить)\b.*\b(?:выполненн\w+|сделанн\w+|готов\w+|завершенн\w+)\b"
    r"|\b(?:выполнен\w?|сделан\w?|готово|закончил\w*|завершил\w*|сделал\w*|выполнил\w*)\b"
    r"|^(?:заверши|закрой|закрыть|завершить)\b"
)
_RE_RENAME = re.compile(
    r"\b(?:переименуй\w*|измени\w*\s+название|поменяй\w*\s+название|назови)\b.*?\b(?:в|на)\s+(.+)$"
)
_RE_STRIP_REMINDER = re.compile(
    r"^(?:(?:а\s+)?(?:ты\s+)?(?:мне\s+)?"
    r"(?:напомни(?:ть|те)?|напоминай(?:те)?|поставь\s+напоминание|создай\s+напоминание|добавь\s+напоминание"
    r"|сделай\s+напоминание|установи\s+напоминание|напоминание|не\s+дай\s+(?:мне\s+)?забыть"
    r"|не\s+забудь|не\s+забыть)(?:\s+мне|\s+нам)?)(?![а-яё])"
    r"(?:\s*[,:]?\s*(?:о\s+том|про\s+то|о|об|обо|про|что(?:\s+(?:мне\s+)?(?:нужно|надо))?"
    r"|чтобы(?:\s+я)?))*\s*",
    re.IGNORECASE,
)
_RE_STRIP_TASK = re.compile(
    r"^(?:(?:а\s+)?(?:мне\s+|у\s+меня\s+|я\s+)?"
    r"(?:(?:поставь|поставить|добавь|добавить|создай|создать|запиши|записать|заведи|внеси|занеси"
    r"|запланируй|запланировать)(?:\s+мне|\s+в\s+\S+)?"
    r"(?:\s+(?:новую\s+)?(?:задачу|задачку|дело|таск|пункт|напоминание))?"
    r"|(?:новая\s+)?(?:задача|задачка|дело|таск)|(?:мне\s+)?(?:нужно|надо|необходимо|пора)"
    r"|должен(?:а)?|хочу|требуется)"
    r")(?![а-яё])(?:\s*[,:\-]?\s*(?:о\s+том|что(?:\s+(?:нужно|надо))?|чтобы|на|про|по)\b)?\s*",
    re.IGNORECASE,
)
_RE_NEED_WORD = re.compile(r"\b(?:нужно|надо|необходимо|пора|должен|должна|хочу|требуется)\b", re.I)
_RE_QUERY_NOISE = re.compile(
    r"\b(?:когда-то|как-то|где-то|что-то|какую-то|какая-то|какой-то|найди\w*|найти|поищи\w*|ищи"
    r"|искать|отыщи|разыщи|вспомни|покажи|покажите|показать|посмотри|скажи|подскажи|напомни|когда"
    r"|где|есть\s+ли|был\w*|будет|будут|ли|у\s+меня|меня|мне|мой|моя|мою|мои|моей|я|ты|там|тут"
    r"|какая|какую|какой|какие|вроде|кажется|ставил\w*|создавал\w*|добавлял\w*"
    r"|записывал\w*|планировал\w*|назначал\w*|запись|записи|ее|его|ее|а|и|же|уже|еще|все)\b",
    re.IGNORECASE,
)
_RE_LEADING_ABOUT = re.compile(r"^(?:про|о|об|обо|по|насчет|с|со|на|для)\s+(?=\S)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class _Signals:
    """Предвычисленные признаки реплики."""

    raw: str
    lower: str
    tokens: tuple[str, ...]
    lemmas: frozenset[str]
    when: Temporal

    def has(self, vocabulary: set[str]) -> bool:
        return bool(self.lemmas & vocabulary)


def clean_utterance(text: str) -> str:
    text = _WAKE.sub("", text)
    text = _FILLERS.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip(" \t\n,.!…")


def _lower(text: str) -> str:
    return text.lower().replace("ё", "е").strip(" ?!.,…")


def _capitalize_names(title: str) -> str:
    def fix(match: re.Match[str]) -> str:
        word = match.group(0)
        if word[0].isupper():
            return word
        parsed = analyzer().parse(word)[0]
        return word.capitalize() if "Name" in parsed.tag and parsed.score > 0.3 else word

    return re.sub(r"[а-яёА-ЯЁ]+", fix, title)


def _sentence_case(title: str) -> str:
    title = _capitalize_names(title.strip(" ,.;:!?-—"))
    return title[:1].upper() + title[1:]


def _meeting_title(residual: str) -> str:
    """Название встречи: от существительного-события до конца, в именительном падеже."""
    tokens = residual.split()
    for index, token in enumerate(tokens):
        word = lemma(token.strip(",.")).replace("ё", "е")
        if word in _MEETING_LEMMAS:
            head = word.replace("планерка", "планёрка").replace("прием", "приём")
            return _sentence_case(" ".join([head, *tokens[index + 1 :]]))
        if word in _MEETING_VERBS:
            rest = tokens[index + 1 :]
            noun = "созвон" if word.startswith("созв") else "встреча"
            return _sentence_case(" ".join([noun, *rest]))
    return _sentence_case(_strip_create_prefix(residual)) or "Встреча"


def _strip_create_prefix(text: str) -> str:
    text = re.sub(
        r"^(?:(?:а\s+)?(?:поставь|поставить|добавь|добавить|создай|создать|запиши|запланируй"
        r"|назначь|назначить|организуй|у\s+меня)(?:\s+мне)?(?![а-яё])\s*)+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def _task_title(residual: str) -> str:
    title = residual
    for _ in range(3):
        match = _RE_STRIP_TASK.match(title)
        if match is None or not match.group(0):
            break
        title = title[match.end() :]
        # После «надо» глагол уже относится к делу: «надо создать презентацию».
        if _RE_NEED_WORD.search(match.group(0)):
            break
    return _sentence_case(title)


def _reminder_title(residual: str) -> str:
    title = _RE_STRIP_REMINDER.sub("", residual, count=1)
    # Предлог мог остаться после вырезания выражения времени: «напомни завтра о встрече».
    title = re.sub(
        r"^(?:мне\s+)?(?:о\s+том,?\s+что|про\s+то,?\s+что|что(?:\s+(?:мне\s+)?(?:нужно|надо))?"
        r"|чтобы(?:\s+я)?|о|об|обо|про)\s+",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(r"^(?:нужно|надо)\s+", "", title, flags=re.IGNORECASE)
    head, *rest = title.split() or [""]
    if lemma(head).replace("ё", "е") in _MEETING_LEMMAS:
        title = " ".join([lemma(head), *rest])
    return _sentence_case(title)


def _extract_query(residual: str) -> str:
    query = _RE_QUERY_NOISE.sub(" ", residual.lower().replace("ё", "е"))
    query = re.sub(r"[?!.,]", " ", query)
    query = re.sub(r"\s+", " ", query).strip()
    previous = None
    while previous != query:
        previous = query
        query = _RE_LEADING_ABOUT.sub("", query)
    return query


def _detect_kind(signals: _Signals) -> ItemKind | None:
    lemmas = signals.lemmas
    if lemmas & {"встреча", "созвон", "совещание", "собеседование", "митинг", "планерка"}:
        return ItemKind.MEETING
    if "напоминание" in lemmas:
        return ItemKind.REMINDER
    if lemmas & _TASK_NOUNS:
        return ItemKind.TASK
    return None


def _starts_with_infinitive(residual: str) -> bool:
    """Глагол ищется в тексте без времени: «Завтра в 9 утра купить билеты»."""
    for token in words(residual)[:3]:
        parsed = analyzer().parse(token)[0]
        if "INFN" in parsed.tag and parsed.score > 0.5:
            return True
    return False


def _unknown_verb(tokens: tuple[str, ...]) -> bool:
    """В реплике есть глагол вне словаря: значит, правила могли не понять её смысл."""
    known = _CREATE_LEMMAS | _MEETING_VERBS | _NEED_LEMMAS | {"быть", "напомнить"}
    for token in tokens:
        parsed = analyzer().parse(token)[0]
        if parsed.tag.POS == "VERB" and str(parsed.normal_form) not in known:
            return True
    return False


def points_at_record(text: str) -> bool:
    """В реплике есть указание на существующую запись: «её», «эту», «последнюю»."""
    tokens = words(text)
    return any(t in _REFERENCE_WORDS or lemma(t) in _ORDINALS for t in tokens)


def _reference(signals: _Signals, query: str) -> Reference:
    ordinal = next(
        (value for token in signals.tokens if (value := _ORDINALS.get(lemma(token)))), None
    )
    if ordinal is not None:
        return Reference(ordinal=ordinal)
    meaningful = [
        w for w in words(query) if lemma(w) not in _ANY_ITEM_NOUNS and w not in _REFERENCE_WORDS
    ]
    return Reference(query=query) if meaningful else Reference(pronoun=True)


class RuleInterpreter:
    def interpret(self, text: str, context: NluContext) -> Command:
        cleaned = clean_utterance(text)
        lower = _lower(cleaned)
        past = bool(_RE_PAST.search(lower))
        when = parse_temporal(cleaned, context.now, past=past)
        tokens = tuple(words(cleaned))
        signals = _Signals(
            raw=cleaned,
            lower=lower,
            tokens=tokens,
            lemmas=frozenset(lemma(t) for t in tokens),
            when=when,
        )
        residual = strip_spans(cleaned, when.spans)
        command = self._classify(signals, residual)
        command.text = text
        command.when = when
        command.past = command.past or past
        return command

    def _classify(self, s: _Signals, residual: str) -> Command:
        if not s.lower:
            return Command(Action.UNKNOWN, confidence=0.0)
        for check in (self._smalltalk, self._undo, self._complete, self._delete, self._update):
            if (command := check(s, residual)) is not None:
                return command
        for check in (self._search, self._list, self._create):
            if (command := check(s, residual)) is not None:
                return command
        return Command(Action.UNKNOWN, confidence=0.1)

    def _smalltalk(self, s: _Signals, residual: str) -> Command | None:
        if _RE_HELP.search(s.lower):
            return Command(Action.HELP, confidence=0.95)
        if _RE_GREET.match(s.lower) and len(s.tokens) <= 4:
            return Command(Action.GREET, confidence=0.95)
        if _RE_THANKS.match(s.lower) and len(s.tokens) <= 3:
            return Command(Action.THANKS, confidence=0.95)
        return None

    def _undo(self, s: _Signals, residual: str) -> Command | None:
        return Command(Action.UNDO, confidence=0.95) if _RE_UNDO.match(s.lower) else None

    def _delete(self, s: _Signals, residual: str) -> Command | None:
        is_delete = s.has(_DELETE_LEMMAS) or s.lower.startswith(("отмени ", "отменить "))
        if not is_delete or "напомни" in s.tokens:
            return None
        query = _extract_query(
            re.sub(
                r"\b(?:удали\w*|убери\w*|убрать|отмени\w*|стереть|сотри\w*|снеси\w*|вычеркни\w*)\b",
                " ",
                residual,
                flags=re.IGNORECASE,
            )
        )
        bulk = bool(_RE_BULK.search(s.lower))
        return Command(
            Action.DELETE,
            kind=_detect_kind(s),
            reference=_reference(s, query),
            bulk=bulk,
            confidence=0.9,
        )

    def _complete(self, s: _Signals, residual: str) -> Command | None:
        already = _RE_ALREADY_DONE.match(s.lower)
        if (
            already
            and "VERB" in analyzer().parse(already.group(1))[0].tag
            and "past" in (analyzer().parse(already.group(1))[0].tag)
        ):
            query = _extract_query(f"{lemma(already.group(1))}{already.group(2)}")
            return Command(
                Action.COMPLETE, reference=Reference(query=query), kind=None, confidence=0.75
            )
        if not _RE_MARK_DONE.search(s.lower) or (
            _RE_DONE_FILTER.search(s.lower)
            and (_RE_LIST_QUESTION.search(s.lower) or s.has(_SHOW_LEMMAS))
        ):
            return None
        query = _extract_query(
            re.sub(
                r"\b(?:отмет\w+|пометь\w*|заверши\w*|закрой\w*|закрыть|как|выполненн\w+|выполнен\w?"
                r"|сделанн\w+|сделан\w?|сделал\w*|выполнил\w*|закончил\w*|завершил\w*|завершенн\w+"
                r"|готов\w*)\b",
                " ",
                residual,
                flags=re.IGNORECASE,
            )
        )
        return Command(
            Action.COMPLETE, kind=_detect_kind(s), reference=_reference(s, query), confidence=0.9
        )

    def _update(self, s: _Signals, residual: str) -> Command | None:
        rename = _RE_RENAME.search(s.raw.lower())
        if rename:
            original = s.raw[rename.start(1) : rename.end(1)]
            query = _extract_query(s.raw[: rename.start(1)])
            query = re.sub(
                r"\b(?:переименуй\w*|название|измени\w*|поменяй\w*|назови|в|на)\b", " ", query
            )
            return Command(
                Action.UPDATE,
                kind=_detect_kind(s),
                new_title=_sentence_case(original),
                reference=_reference(s, re.sub(r"\s+", " ", query).strip()),
                confidence=0.9,
            )
        move_by_pronoun = bool(
            re.match(r"^(?:а\s+)?(?:лучше\s+|давай\s+|сделай\s+|поставь\s+)", s.lower)
            and s.tokens
            and any(t in _REFERENCE_WORDS for t in s.tokens)
            and not s.when.empty
        )
        if not s.has(_UPDATE_LEMMAS) and not move_by_pronoun:
            return None
        query = _extract_query(
            re.sub(
                r"\b(?:перенеси\w*|перенести|передвинь\w*|сдвинь\w*|подвинь\w*|переставь\w*"
                r"|поменяй\w*|измени\w*|смени\w*|исправь\w*|замени\w*|отложи\w*|перекинь\w*"
                r"|лучше|давай|сделай|поставь|время|дату|дата)\b",
                " ",
                residual,
                flags=re.IGNORECASE,
            )
        )
        return Command(
            Action.UPDATE, kind=_detect_kind(s), reference=_reference(s, query), confidence=0.9
        )

    def _search(self, s: _Signals, residual: str) -> Command | None:
        has_search_verb = bool(_RE_SEARCH_VERB.search(s.lower))
        when_question = bool(_RE_WHEN_QUESTION.match(s.lower))
        about = bool(_RE_ABOUT.search(s.lower))
        exists_with_person = bool(
            _RE_EXISTS.search(s.lower) and re.search(r"\b(?:с|со|по)\s+\w", residual.lower())
        )
        show_about = s.has(_SHOW_LEMMAS) and about
        if not (has_search_verb or when_question or exists_with_person or show_about):
            return None
        kind = _detect_kind(s)
        query = _extract_query(residual)
        if lemma(query) in _ANY_ITEM_NOUNS:
            query = ""
        else:
            query = " ".join(
                w for w in query.split() if lemma(w) not in _ANY_ITEM_NOUNS or kind is None
            )
            query = _RE_LEADING_ABOUT.sub("", query)
        return Command(
            Action.SEARCH,
            query=query,
            kind=kind,
            confidence=0.9,
            past=when_question and "был" in s.lower,
        )

    def _list(self, s: _Signals, residual: str) -> Command | None:
        question = bool(_RE_LIST_QUESTION.search(s.lower))
        show = s.has(_SHOW_LEMMAS) and s.has(_LIST_NOUNS)
        exists = bool(_RE_EXISTS.search(s.lower)) and s.has(_LIST_NOUNS)
        overdue = bool(_RE_OVERDUE.search(s.lower)) and (
            question or s.has(_LIST_NOUNS | _SHOW_LEMMAS)
        )
        bare_noun = s.has(_LIST_NOUNS) and len(s.tokens) <= 5 and not s.has(_CREATE_LEMMAS)
        bare_noun = bare_noun and not s.has(_NEED_LEMMAS) and not s.when.empty
        if not (question or show or exists or overdue or bare_noun):
            return None
        return Command(
            Action.LIST,
            kind=_detect_kind(s),
            include_done=bool(_RE_DONE_FILTER.search(s.lower)),
            only_overdue=overdue,
            confidence=0.9 if question or show or exists else 0.75,
        )

    def _create(self, s: _Signals, residual: str) -> Command | None:
        reminder = bool(s.lemmas & {"напомнить", "напоминать", "напоминание"})
        reminder = (
            reminder or "не забудь" in s.lower or "не дай" in s.lower or "не забыть" in s.lower
        )
        if reminder:
            return Command(Action.CREATE_REMINDER, title=_reminder_title(residual), confidence=0.95)

        meeting_lemma = s.lemmas & _MEETING_LEMMAS
        strong = meeting_lemma - _WEAK_MEETING_LEMMAS
        meeting_verb = s.has(_MEETING_VERBS)
        task_words = s.has(_TASK_NOUNS)
        need = s.has(_NEED_LEMMAS)
        weak_with_person = bool(meeting_lemma) and bool(
            re.search(r"\b(?:обед|ужин|звонок|созвон)\w*\s+(?:с|со)\s+\w", s.lower)
            or re.search(r"\bпри[её]м\w*\s+у\s+\w", s.lower)
        )
        if not task_words and not need and (strong or meeting_verb or weak_with_person):
            confidence = 0.7 if _unknown_verb(s.tokens) else 0.95
            return Command(
                Action.CREATE_MEETING, title=_meeting_title(residual), confidence=confidence
            )
        if task_words and (strong or meeting_verb) and not s.has(_CREATE_LEMMAS):
            return Command(Action.CREATE_MEETING, title=_meeting_title(residual), confidence=0.7)

        if s.has(_CREATE_LEMMAS) or need:
            return Command(Action.CREATE_TASK, title=_task_title(residual), confidence=0.9)
        if task_words:
            # Слово «задача» без глагола создания: это может быть и вопрос, и ссылка на запись.
            return Command(Action.CREATE_TASK, title=_task_title(residual), confidence=0.6)
        if _starts_with_infinitive(residual):
            confidence = 0.85 if not s.when.empty else 0.65
            return Command(Action.CREATE_TASK, title=_task_title(residual), confidence=confidence)
        content = [w for w in words(residual) if len(w) > 2]
        if not s.when.empty and len(content) >= 1 and len(residual.split()) >= 2:
            # «Завтра стоматолог», «в пятницу отчёт» — похоже на запись, но уверенности мало.
            return Command(Action.CREATE_TASK, title=_task_title(residual), confidence=0.55)
        return None
