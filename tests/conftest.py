from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from orbita.clock import FixedClock
from orbita.dialog.manager import DialogManager, TurnResult
from orbita.domain.models import User
from orbita.nlu.rules import RuleInterpreter
from orbita.services.items import ItemService
from orbita.services.search import SearchService
from orbita.storage.db import Database, make_engine, upgrade_database
from orbita.storage.repositories import (
    DialogRepository,
    EmbeddingRepository,
    ItemRepository,
    UserRepository,
)

MSK = ZoneInfo("Europe/Moscow")
NOW = datetime(2026, 10, 30, 14, 0, tzinfo=MSK)


class HashEmbedder:
    """Детерминированная заглушка: тесты логики не должны скачивать модель."""

    model_name = "hash-test"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")
            vector = np.random.default_rng(seed).standard_normal(16)
            vectors.append(vector / np.linalg.norm(vector))
        return np.array(vectors, dtype=np.float32)


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock(NOW)


@pytest.fixture
def database(tmp_path) -> Database:
    url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    upgrade_database(url)
    return Database(make_engine(url))


@dataclass
class Assistant:
    """Тестовый собеседник: каждая реплика идёт в своей транзакции, как в приложении."""

    database: Database
    clock: FixedClock
    user_id: int
    embedder: object | None = None
    reranker: object | None = None

    def _services(self, session):
        items_repo = ItemRepository(session)
        search = SearchService(
            items_repo, EmbeddingRepository(session), self.embedder, self.reranker
        )
        items = ItemService(items_repo, self.clock, MSK, on_saved=search.index)
        return items, search

    def say(self, text: str) -> TurnResult:
        with self.database.session() as session:
            items, search = self._services(session)
            manager = DialogManager(
                RuleInterpreter(), items, search, DialogRepository(session), MSK
            )
            return manager.handle(self.user_id, text)

    def items(self):
        with self.database.session() as session:
            return ItemRepository(session).live(self.user_id)

    def titles(self) -> list[str]:
        return sorted(item.title for item in self.items())


@pytest.fixture
def assistant(database, clock) -> Assistant:
    with database.session() as session:
        user = UserRepository(session).add(User(name="Тест", calendar_token="cal"), "token", NOW)
    return Assistant(database, clock, user.id)
