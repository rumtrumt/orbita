"""Шина событий для доставки напоминаний и обновлений в открытые вкладки по SSE."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any


class EventBus:
    """Публикация безопасна из рабочих потоков, подписчики живут в цикле событий."""

    def __init__(self) -> None:
        self._subscribers: dict[int, set[asyncio.Queue[dict[str, Any]]]] = defaultdict(set)
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self, user_id: int) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        self._subscribers[user_id].add(queue)
        return queue

    def unsubscribe(self, user_id: int, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers[user_id].discard(queue)

    def publish(self, user_id: int, event: dict[str, Any]) -> None:
        if self._loop is None:
            return
        for queue in tuple(self._subscribers[user_id]):
            self._loop.call_soon_threadsafe(queue.put_nowait, event)
