"""Векторные представления текста для семантического поиска."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import structlog

log = structlog.get_logger(__name__)


class Embedder(Protocol):
    @property
    def model_name(self) -> str: ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """Нормированные векторы, по строке на текст."""
        ...


class FastEmbedder:
    """Мультиязычная sentence-transformers модель через ONNX: без PyTorch и GPU."""

    def __init__(self, model_name: str, cache_dir: Path) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding  # noqa: PLC0415 — тяжёлый импорт

                log.info("embeddings.loading", model=self._model_name)
                self._model = TextEmbedding(self._model_name, cache_dir=str(self._cache_dir))
            return self._model

    def warm_up(self) -> None:
        self.embed(["прогрев"])

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(list(self._load().embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-9, None)
