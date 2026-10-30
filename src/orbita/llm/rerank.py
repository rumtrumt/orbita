"""Смысловой отбор кандидатов поиска."""

from __future__ import annotations

from typing import Any

from orbita.llm.client import LlmClient

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"matches": {"type": "array", "items": {"type": "integer"}}},
    "required": ["matches"],
}
_SYSTEM = """\
Пользователь ищет свою запись. Верни номера записей, которые действительно относятся к запросу по \
смыслу: синоним, частный случай, то же событие другими словами. Сомнительное и лишь отдалённо \
похожее не включай. Если подходящих нет, верни пустой список.

Запрос: зубной врач. Записи: 1. Записаться к стоматологу; 2. Купить молоко → {"matches":[1]}
Запрос: рецепт супа. Записи: 1. Подготовить презентацию; 2. Позвонить Ивану → {"matches":[]}
"""


class LlmReranker:
    def __init__(self, client: LlmClient) -> None:
        self._client = client

    def select(self, query: str, titles: list[str]) -> list[int]:
        numbered = "; ".join(f"{index}. {title}" for index, title in enumerate(titles, 1))
        data = self._client.complete_json(
            _SYSTEM, f"Запрос: {query}. Записи: {numbered}", _SCHEMA, "matches"
        )
        return [n - 1 for n in data.get("matches", []) if 0 < n <= len(titles)]
