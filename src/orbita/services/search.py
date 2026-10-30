"""Гибридный поиск по записям.

Лексическая часть точна на именах и опечатках распознавания («Иваном» → «Иван»), семантическая
ловит близкий смысл («документ на подпись» → «договор»). Записей у одного человека тысячи, а
не миллионы, поэтому полный перебор в памяти проще и надёжнее внешнего векторного индекса.

Эмбеддинги хорошо ранжируют, но абсолютная близость не отделяет слабое совпадение от шума:
«поездка» и «билеты в Казань» дают 0.37, а «рецепт борща» и «презентация» 0.46. Поэтому средняя
зона решается вторым шагом: LLM судит, подходят ли кандидаты по смыслу. Без LLM средняя зона
отбрасывается, и поиск теряет полноту, но не начинает выдавать мусор.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Protocol

import numpy as np
import structlog
from rapidfuzz import fuzz

from orbita.domain.models import Item, ItemKind, SearchHit
from orbita.llm.client import LlmError
from orbita.nlu.morphology import lemmas
from orbita.nlu.temporal import DateRange
from orbita.services.embeddings import Embedder
from orbita.storage.repositories import EmbeddingRepository, ItemRepository

log = structlog.get_logger(__name__)

_STOP_LEMMAS = {
    "и", "в", "на", "с", "со", "по", "к", "у", "о", "об", "про", "для", "от", "до", "за", "из",
    "это", "тот", "мой", "я", "что", "когда", "задача", "встреча", "напоминание", "запись",
    "дело", "событие",
}  # fmt: skip
_FUZZY_MIN = 86.0
# Пороги подобраны на tests/integration/test_search_quality.py.
_SEMANTIC_SURE = 0.7  # выше — близко по смыслу без дополнительной проверки
_SEMANTIC_PLAUSIBLE = 0.53  # столько достаточно, если проверить некому
_SEMANTIC_CANDIDATE = 0.3  # ниже — не показываем даже LLM
_MAX_CANDIDATES = 5
_MAX_BLIND_CANDIDATES = 30  # без эмбеддингов LLM смотрит последние записи
_LEXICAL_MIN = 0.45
_LEXICAL_WEIGHT = 0.65


class Reranker(Protocol):
    def select(self, query: str, titles: list[str]) -> list[int]:
        """Индексы названий, подходящих к запросу по смыслу."""
        ...


@dataclass(frozen=True, slots=True)
class SearchFilters:
    kind: ItemKind | None = None
    period: DateRange | None = None
    before: date | None = None


def _content_lemmas(text: str) -> list[str]:
    return [token for token in lemmas(text) if token not in _STOP_LEMMAS and len(token) > 1]


def _lexical_scores(query: list[str], items: list[Item]) -> dict[int, float]:
    """Доля слов запроса, найденных в записи, взвешенная по их редкости среди записей."""
    documents = {item.id: set(_content_lemmas(item.title)) for item in items if item.id}
    total = max(len(documents), 1)
    frequency = {
        word: sum(any(_token_score(word, t) > 0 for t in doc) for doc in documents.values())
        for word in query
    }
    weights = {word: math.log(1 + total / (1 + frequency[word])) + 0.1 for word in query}
    weight_sum = sum(weights.values()) or 1.0
    scores: dict[int, float] = {}
    for item_id, document in documents.items():
        matched = sum(
            weights[word] * max((_token_score(word, t) for t in document), default=0.0)
            for word in query
        )
        if matched:
            scores[item_id] = matched / weight_sum
    return scores


def _token_score(query_word: str, document_word: str) -> float:
    if query_word == document_word:
        return 1.0
    if len(query_word) >= 4 and len(document_word) >= 4:
        similarity = fuzz.ratio(query_word, document_word)
        if similarity >= _FUZZY_MIN:
            return similarity / 100
    return 0.0


class SearchService:
    def __init__(
        self,
        items: ItemRepository,
        embeddings: EmbeddingRepository,
        embedder: Embedder | None,
        reranker: Reranker | None = None,
    ) -> None:
        self._items = items
        self._embeddings = embeddings
        self._embedder = embedder
        self._reranker = reranker

    def index(self, item: Item) -> None:
        """Обновить вектор записи. Сбой модели не должен мешать сохранению задачи."""
        if self._embedder is None or item.id is None:
            return
        try:
            vector = self._embedder.embed([item.title])[0]
        except Exception:
            log.exception("search.index_failed", item_id=item.id)
            return
        self._embeddings.upsert(
            item.id, self._embedder.model_name, item.title, vector.astype(np.float32).tobytes()
        )

    def _semantic_scores(self, user_id: int, query: str, items: list[Item]) -> dict[int, float]:
        if self._embedder is None:
            return {}
        try:
            stored = self._embeddings.for_user(user_id, self._embedder.model_name)
            for item in items:
                # Запись могла быть создана до включения поиска или переименована.
                if item.id is not None and stored.get(item.id, ("",))[0] != item.title:
                    self.index(item)
            stored = self._embeddings.for_user(user_id, self._embedder.model_name)
            ids = [item.id for item in items if item.id in stored]
            if not ids:
                return {}
            matrix = np.vstack(
                [np.frombuffer(stored[item_id][1], dtype=np.float32) for item_id in ids]
            )
            query_vector = self._embedder.embed([query])[0]
            return dict(zip(ids, (matrix @ query_vector).tolist(), strict=True))
        except Exception:
            log.exception("search.semantic_failed")
            return {}

    def search(
        self,
        user_id: int,
        query: str,
        filters: SearchFilters | None = None,
        limit: int = 5,
    ) -> list[SearchHit]:
        filters = filters or SearchFilters()
        items = [item for item in self._items.live(user_id) if _passes(item, filters)]
        if not items:
            return []
        if not query.strip():
            items.sort(key=lambda item: item.created_at or item.updated_at, reverse=True)  # type: ignore[arg-type,return-value]
            return [SearchHit(item, 1.0, "filter") for item in items[:limit]]

        query_lemmas = _content_lemmas(query)
        lexical = _lexical_scores(query_lemmas, items) if query_lemmas else {}
        semantic = self._semantic_scores(user_id, query, items)

        hits: list[SearchHit] = []
        uncertain: list[tuple[Item, float]] = []
        for item in items:
            assert item.id is not None
            lex = lexical.get(item.id, 0.0)
            sem = semantic.get(item.id, 0.0)
            if lex >= _LEXICAL_MIN:
                score = _LEXICAL_WEIGHT * lex + (1 - _LEXICAL_WEIGHT) * max(sem, 0.0) + 0.2
                hits.append(SearchHit(item, score, "lexical"))
            elif sem >= _SEMANTIC_SURE:
                hits.append(SearchHit(item, sem, "semantic"))
            elif sem >= _SEMANTIC_CANDIDATE:
                uncertain.append((item, sem))
        if not hits and self._embedder is None:
            newest = sorted(items, key=lambda item: item.id or 0, reverse=True)
            uncertain = [(item, 0.0) for item in newest[:_MAX_BLIND_CANDIDATES]]
        if not hits:
            hits = self._judge(query, uncertain)
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]

    def _judge(self, query: str, uncertain: list[tuple[Item, float]]) -> list[SearchHit]:
        """Решить судьбу записей из средней зоны близости."""
        uncertain.sort(key=lambda pair: pair[1], reverse=True)
        limit = _MAX_CANDIDATES if self._embedder else _MAX_BLIND_CANDIDATES
        candidates = uncertain[:limit]
        if not candidates:
            return []
        if self._reranker is not None:
            try:
                chosen = self._reranker.select(query, [item.title for item, _ in candidates])
                return [SearchHit(candidates[i][0], candidates[i][1], "llm") for i in chosen]
            except LlmError as error:
                log.warning("search.rerank_failed", error=str(error))
        return [
            SearchHit(item, score, "semantic")
            for item, score in candidates
            if score >= _SEMANTIC_PLAUSIBLE
        ]


def _passes(item: Item, filters: SearchFilters) -> bool:
    if filters.kind is not None and item.kind is not filters.kind:
        return False
    if filters.period is not None and (item.day is None or not filters.period.contains(item.day)):
        return False
    return not (filters.before is not None and (item.day is None or item.day > filters.before))
