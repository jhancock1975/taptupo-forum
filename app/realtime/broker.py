"""In-process pub/sub broker keyed by ``thread_id``.

A minimal asyncio-based broker: subscribers get their own bounded
``asyncio.Queue``; publishers fan out a single event to every queue
subscribed to that thread. If a queue is full we drop the message for
that subscriber rather than back-pressuring the publisher — slow
clients must not stall agent reply writes.

Scope is intentionally single-process. Multi-worker deployments would
swap this for Redis pub/sub or a similar cross-process channel.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

_log = structlog.get_logger(__name__)

_DEFAULT_MAXSIZE = 64


class Broker:
    """Topic-keyed fan-out of JSON-serialisable events."""

    def __init__(self, *, queue_maxsize: int = _DEFAULT_MAXSIZE) -> None:
        self._subs: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}
        self._maxsize = queue_maxsize
        self._lock = asyncio.Lock()

    async def subscribe(self, topic: str) -> asyncio.Queue[dict[str, Any]]:
        """Return a fresh queue that will receive events published to ``topic``."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._maxsize)
        async with self._lock:
            self._subs.setdefault(topic, set()).add(queue)
        return queue

    async def unsubscribe(self, topic: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Remove ``queue`` from the subscriber set for ``topic``."""
        async with self._lock:
            subs = self._subs.get(topic)
            if subs is None:
                return
            subs.discard(queue)
            if not subs:
                del self._subs[topic]

    async def publish(self, topic: str, event: dict[str, Any]) -> None:
        """Deliver ``event`` to every subscriber of ``topic`` without blocking."""
        async with self._lock:
            subs = list(self._subs.get(topic, ()))
        for queue in subs:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                _log.warning("realtime.queue_full", topic=topic)
