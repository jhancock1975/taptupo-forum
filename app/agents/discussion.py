"""Discussion engine: dispatches new posts to agents for possible reply.

For each incoming post we walk the agent roster and, for those that
choose to reply (``decide_to_respond``), schedule a response after a
random jitter (30-300s by default) so conversation feels organic.

Jitter is configurable so unit tests can run with zero delay. The engine
is transport-agnostic: it just calls the repository to create posts and
bump thread activity. Websocket broadcast happens elsewhere by watching
those writes.
"""

from __future__ import annotations

import asyncio
import random
from datetime import UTC, datetime
from typing import Protocol

import structlog

from app.db.interface import RepositoryInterface
from app.models import Post, User
from app.realtime.broker import Broker

_log = structlog.get_logger(__name__)


class _Agent(Protocol):
    user: User

    async def decide_to_respond(self, post_text: str) -> bool: ...
    async def generate_response(self, post_text: str) -> str: ...


class DiscussionEngine:
    """Coordinates agent responses to forum activity."""

    def __init__(
        self,
        *,
        repository: RepositoryInterface,
        agents: list[_Agent],
        min_delay: float = 30.0,
        max_delay: float = 300.0,
        rng: random.Random | None = None,
        broker: Broker | None = None,
    ) -> None:
        self._repo = repository
        self._agents = agents
        self._min_delay = min_delay
        self._max_delay = max_delay
        self._rng = rng or random.Random()  # noqa: S311  # nosec B311 — non-crypto use
        self._broker = broker

    async def on_new_post(self, post: Post) -> None:
        """Possibly-schedule replies from each agent to ``post``."""
        tasks = [asyncio.create_task(self._respond_maybe(post, agent)) for agent in self._agents]
        # Wait for all decisions so tests see posts without needing to
        # manage loop lifetimes. In production, callers may prefer to
        # schedule on_new_post as a fire-and-forget task.
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _respond_maybe(self, post: Post, agent: _Agent) -> None:
        if agent.user.user_id == post.author_id:
            return
        try:
            if not await agent.decide_to_respond(post.content):
                return
            await asyncio.sleep(self._random_delay())
            reply_text = await agent.generate_response(post.content)
            reply = Post(
                thread_id=post.thread_id,
                author_id=agent.user.user_id,
                content=reply_text,
                created_at=datetime.now(UTC),
            )
            await self._repo.create_post(reply)
            await self._repo.update_thread_activity(post.thread_id, reply.created_at)
            if self._broker is not None:
                await self._broker.publish(
                    post.thread_id,
                    {"type": "post.created", "post": reply.model_dump(mode="json")},
                )
        except Exception as exc:
            _log.warning(
                "discussion.agent_failed",
                agent=agent.user.username,
                error=str(exc),
            )

    def _random_delay(self) -> float:
        if self._max_delay <= self._min_delay:
            return self._min_delay
        return self._rng.uniform(self._min_delay, self._max_delay)
