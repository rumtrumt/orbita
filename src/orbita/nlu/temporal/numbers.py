"""Перевод числительных в цифры с сохранением карты смещений.

Whisper пишет и «в 10», и «в десять», поэтому парсер времени работает с нормализованной
строкой, а карта позволяет вырезать найденные выражения из исходного текста — название
задачи должно остаться таким, как его произнёс пользователь («купить два билета»).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from orbita.nlu.morphology import analyzer

_CARDINALS = {
    "ноль": 0,
    "один": 1,
    "два": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
    "одиннадцать": 11,
    "двенадцать": 12,
    "тринадцать": 13,
    "четырнадцать": 14,
    "пятнадцать": 15,
    "шестнадцать": 16,
    "семнадцать": 17,
    "восемнадцать": 18,
    "девятнадцать": 19,
    "двадцать": 20,
    "тридцать": 30,
    "сорок": 40,
    "пятьдесят": 50,
}
_ORDINALS = {
    "первый": 1,
    "второй": 2,
    "третий": 3,
    "четвертый": 4,
    "пятый": 5,
    "шестой": 6,
    "седьмой": 7,
    "восьмой": 8,
    "девятый": 9,
    "десятый": 10,
    "одиннадцатый": 11,
    "двенадцатый": 12,
    "тринадцатый": 13,
    "четырнадцатый": 14,
    "пятнадцатый": 15,
    "шестнадцатый": 16,
    "семнадцатый": 17,
    "восемнадцатый": 18,
    "девятнадцатый": 19,
    "двадцатый": 20,
    "тридцатый": 30,
}
# Только родительный/именительный среднего рода: «пятого», «пятое». Формы вроде «первый»
# не трогаем, иначе «в первый раз» превратится во время «в 1».
_ORDINAL_ENDINGS = ("ого", "ое", "ье", "ому", "ьего", "ьему")
_TOKEN = re.compile(r"[а-я]+|\d+")


@dataclass(frozen=True, slots=True)
class Normalized:
    """Текст и для каждого его символа — границы породившего фрагмента исходной строки."""

    text: str
    starts: tuple[int, ...]
    ends: tuple[int, ...]

    def original_span(self, start: int, end: int) -> tuple[int, int]:
        return (self.starts[start], self.ends[end - 1])


def _classify(word: str) -> tuple[str, int] | None:
    for parsed in analyzer().parse(word):
        form = str(parsed.normal_form).replace("ё", "е")
        pos = str(parsed.tag.POS)
        if form in _CARDINALS and pos == "NUMR":
            return ("cardinal", _CARDINALS[form])
        if form in _ORDINALS and word.endswith(_ORDINAL_ENDINGS):
            return ("ordinal", _ORDINALS[form])
    return None


def normalize(text: str) -> Normalized:
    lowered = text.lower().replace("ё", "е")
    tokens = [(m.start(), m.end(), m.group(0)) for m in _TOKEN.finditer(lowered)]
    classes = [_classify(t[2]) if t[2][0].isalpha() else None for t in tokens]

    def gap_is_space(a: int, b: int) -> bool:
        return lowered[tokens[a][1] : tokens[b][0]] == " "

    replacements: list[tuple[int, int, str]] = []
    i = 0
    while i < len(tokens):
        cls = classes[i]
        if cls is None:
            i += 1
            continue
        kind, value = cls
        start, end = tokens[i][0], tokens[i][1]
        following = classes[i + 1] if i + 1 < len(tokens) and gap_is_space(i, i + 1) else None
        if value == 0 and kind == "cardinal":
            # «ноль ноль» → 00, «ноль пять» → 05
            if following is not None and following[0] == "cardinal" and following[1] < 10:
                replacements.append((start, tokens[i + 1][1], f"{following[1]:02d}"))
                i += 2
            else:
                replacements.append((start, end, "0"))
                i += 1
            continue
        if value >= 20 and value % 10 == 0 and following is not None and 0 < following[1] < 10:
            total = value + following[1]
            suffix = "-го" if following[0] == "ordinal" else ""
            replacements.append((start, tokens[i + 1][1], f"{total}{suffix}"))
            i += 2
            continue
        replacements.append((start, end, f"{value}-го" if kind == "ordinal" else str(value)))
        i += 1

    out: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    cursor = 0

    def copy_until(stop: int) -> None:
        out.append(lowered[cursor:stop])
        starts.extend(range(cursor, stop))
        ends.extend(range(cursor + 1, stop + 1))

    for start, end, replacement in replacements:
        copy_until(start)
        out.append(replacement)
        starts.extend([start] * len(replacement))
        ends.extend([end] * len(replacement))
        cursor = end
    copy_until(len(lowered))
    return Normalized("".join(out), tuple(starts), tuple(ends))
