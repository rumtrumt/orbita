"""Обёртка над pymorphy3: лемматизация с кэшем и склонение для ответов."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import pymorphy3

_WORD = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


@lru_cache(maxsize=1)
def analyzer() -> Any:
    return pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=50_000)
def lemma(word: str) -> str:
    return str(analyzer().parse(word.lower())[0].normal_form).replace("ё", "е")


def words(text: str) -> list[str]:
    return [m.group(0).lower().replace("ё", "е") for m in _WORD.finditer(text)]


def lemmas(text: str) -> list[str]:
    return [lemma(w) for w in words(text)]


def to_nominative(word: str) -> str:
    """«встречу» → «встреча»; для незнакомых слов возвращает исходную форму."""
    parsed = analyzer().parse(word)[0]
    inflected = parsed.inflect({"nomn", parsed.tag.number or "sing"})
    return str(inflected.word) if inflected else word
