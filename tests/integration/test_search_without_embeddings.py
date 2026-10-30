"""Поиск на лёгком сервере, где нет модели эмбеддингов."""

from orbita.llm.client import LlmError


class StubReranker:
    def __init__(self, answer=None, error=False):
        self.answer = answer or []
        self.error = error
        self.seen = []

    def select(self, query, titles):
        self.seen = titles
        if self.error:
            raise LlmError("нет связи")
        return self.answer


def seed(assistant):
    assistant.say("Завтра купить молоко")
    assistant.say("Завтра записаться к стоматологу")


def test_llm_judges_all_records_when_no_words_match(assistant):
    seed(assistant)
    assistant.reranker = StubReranker([0])  # новые записи идут первыми
    reply = assistant.say("Найди задачу про зубного врача")
    assert "Записаться к стоматологу" in reply.reply
    assert len(assistant.reranker.seen) == 2


def test_word_match_does_not_call_llm(assistant):
    seed(assistant)
    assistant.reranker = StubReranker([0])
    assert "Купить молоко" in assistant.say("Найди задачу про молоко").reply
    assert assistant.reranker.seen == []


def test_llm_failure_means_nothing_found(assistant):
    seed(assistant)
    assistant.reranker = StubReranker(error=True)
    assert "Не нашла" in assistant.say("Найди задачу про зубного врача").reply
