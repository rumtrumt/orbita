"""Подготовка ответа к озвучиванию.

Синтезаторы читают «15:00» и «31 октября» плохо или с ошибкой в падеже, поэтому числа
разворачиваются в слова здесь, а текст на экране остаётся компактным.
"""

from __future__ import annotations

import re

from num2words import num2words

_ORDINAL_GENITIVE = (
    "", "первого", "второго", "третьего", "четвёртого", "пятого", "шестого", "седьмого",
    "восьмого", "девятого", "десятого", "одиннадцатого", "двенадцатого", "тринадцатого",
    "четырнадцатого", "пятнадцатого", "шестнадцатого", "семнадцатого", "восемнадцатого",
    "девятнадцатого", "двадцатого", "двадцать первого", "двадцать второго", "двадцать третьего",
    "двадцать четвёртого", "двадцать пятого", "двадцать шестого", "двадцать седьмого",
    "двадцать восьмого", "двадцать девятого", "тридцатого", "тридцать первого",
)  # fmt: skip
_MONTHS = "января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря"
_DATE = re.compile(rf"\b(\d{{1,2}}) ({_MONTHS})(?: (\d{{4}}))?")
_CLOCK = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_NUMBER = re.compile(r"\b\d+\b")
_LIST_MARK = re.compile(r"\b(\d)\)")


def _words(value: int) -> str:
    return str(num2words(value, lang="ru"))


def _year(value: int) -> str:
    return str(num2words(value, lang="ru", to="ordinal")).removesuffix("й") + "го года"


def _spoken_date(match: re.Match[str]) -> str:
    day = int(match.group(1))
    head = _ORDINAL_GENITIVE[day] if 0 < day <= 31 else _words(day)
    tail = f" {_year(int(match.group(3)))}" if match.group(3) else ""
    return f"{head} {match.group(2)}{tail}"


def _spoken_clock(match: re.Match[str]) -> str:
    hour, minute = int(match.group(1)), int(match.group(2))
    if minute == 0:
        return f"{_words(hour)} ноль ноль" if hour else "полночь"
    if minute < 10:
        return f"{_words(hour)} ноль {_words(minute)}"
    return f"{_words(hour)} {_words(minute)}"


def to_speech(text: str) -> str:
    spoken = _LIST_MARK.sub(lambda m: f"{_words(int(m.group(1)))}:", text)
    spoken = _DATE.sub(_spoken_date, spoken)
    spoken = _CLOCK.sub(_spoken_clock, spoken)
    spoken = _NUMBER.sub(lambda m: _words(int(m.group(0))), spoken)
    spoken = spoken.replace(" — ", ", ").replace("«", "").replace("»", "")
    return re.sub(r"\s+", " ", spoken).strip()
