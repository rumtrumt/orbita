"""Качество поиска на настоящей модели эмбеддингов.

Здесь же проверяется порог: близкий смысл находится, а посторонние запросы — нет.
"""

from __future__ import annotations

import pytest

from orbita.config import Settings
from orbita.container import build_llm_client
from orbita.llm.client import LlmError
from orbita.llm.rerank import LlmReranker
from orbita.services.embeddings import FastEmbedder
from tests.conftest import MSK, Assistant  # noqa: F401

pytestmark = [pytest.mark.voice, pytest.mark.llm]

SEED = [
    "Завтра подписать договор с подрядчиком",
    "В понедельник подготовить презентацию для клиента",
    "Встреча с Иваном в среду в 12",
    "В пятницу отправить квартальный отчёт",
    "Завтра купить билеты в Казань",
    "В субботу записаться к стоматологу",
    "В воскресенье оплатить коммунальные услуги",
    "Во вторник забрать ребёнка из школы",
]
FOUND = [
    ("Найди задачу про документы на подпись", "Подписать договор"),
    ("Найди что-то про слайды для заказчика", "презентацию"),
    ("Что у меня было про поездку", "билеты"),
    ("Найди задачу про зубного врача", "стоматолог"),
    ("Найди про счета за квартиру", "коммунальные"),
    ("Поищи финансовую отчётность", "отчёт"),
]
MISSING = [
    "Найди задачу про слона",
    "Найди что-то про ремонт машины",
    "Поищи рецепт борща",
]


@pytest.fixture(scope="module")
def embedder():
    settings = Settings(_env_file=None)
    return FastEmbedder(settings.embedding_model, settings.models_dir / "embeddings")


@pytest.fixture
def searcher(assistant, embedder) -> Assistant:
    assistant.embedder = embedder
    assistant.reranker = LlmReranker(build_llm_client(Settings(_env_file=None)))
    try:
        assistant.reranker.select("проверка", ["Запись"])
    except LlmError:
        pytest.skip("LLM недоступна")
    for phrase in SEED:
        assistant.say(phrase)
        assistant.say("отмена")
    return assistant


@pytest.mark.parametrize(("query", "expected"), FOUND)
def test_close_meaning_is_found(searcher, query, expected):
    reply = searcher.say(query)
    assert expected.lower() in reply.reply.lower(), reply.reply


@pytest.mark.parametrize("query", MISSING)
def test_unrelated_query_finds_nothing(searcher, query):
    reply = searcher.say(query)
    assert "Не нашла" in reply.reply, reply.reply


@pytest.mark.parametrize("query", MISSING)
def test_without_llm_unrelated_query_still_finds_nothing(searcher, query):
    searcher.reranker = None
    assert "Не нашла" in searcher.say(query).reply
