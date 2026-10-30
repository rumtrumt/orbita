"""Приведение текста Whisper к виду, удобному для разбора.

Whisper оформляет речь как прозу: ставит кавычки и точки посреди команды, а редкие для него слова
заменяет созвучными. Ошибки вроде «созван» систематические, поэтому чинятся словарём, а не правилами
в каждом месте разбора.
"""

from __future__ import annotations

import re

_CONFUSIONS = {
    "созван": "созвон",
    "созвана": "созвона",
    "созване": "созвоне",
    "созваном": "созвоном",
    "планерка": "планёрка",
}
# Глагол в начале фразы Whisper часто слышит в другом лице: «Напомню», «Переименуют».
_LEADING_VERBS = {
    "напомню": "напомни",
    "напомнит": "напомни",
    "переименую": "переименуй",
    "переименуют": "переименуй",
    "удалю": "удали",
    "удалит": "удали",
    "перенесу": "перенеси",
    "перенесут": "перенеси",
    "добавлю": "добавь",
    "запланирую": "запланируй",
    "запишу": "запиши",
    "отменю": "отмени",
}
_QUOTES = re.compile(r"[«»\"„“”]")
_INNER_PERIOD = re.compile(r"(?<=[а-яёa-z0-9])\.\s+(?=[а-яёА-ЯЁ])")
_TIME_WORDS = re.compile(r"\bнол\b", re.IGNORECASE)
_WORD = re.compile(r"[а-яёА-ЯЁ]+")


def _keep_case(original: str, replacement: str) -> str:
    return replacement.capitalize() if original[0].isupper() else replacement


def polish(text: str) -> str:
    cleaned = _QUOTES.sub("", text)
    # Одна команда не делится на предложения: «Надо добавить. Соль в суп».
    cleaned = _INNER_PERIOD.sub(" ", cleaned).strip().rstrip(".")
    cleaned = _TIME_WORDS.sub("ноль", cleaned)
    cleaned = _WORD.sub(
        lambda m: (
            _keep_case(m[0], _CONFUSIONS[m[0].lower()]) if m[0].lower() in _CONFUSIONS else m[0]
        ),
        cleaned,
    )
    head = _WORD.match(cleaned)
    if head and head[0].lower() in _LEADING_VERBS:
        word = _keep_case(head[0], _LEADING_VERBS[head[0].lower()])
        cleaned = word + cleaned[head.end() :]
    return cleaned
