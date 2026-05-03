from __future__ import annotations

import asyncio
import random

import structlog

from app.agents.base_agent import BaseAgent
from app.config import settings
from app.db.interface import RepositoryInterface
from app.models.schemas import Post, User
from app.routes.websocket import ConnectionManager

logger = structlog.get_logger()


class DiscussionEngine:
    def __init__(
        self,
        repo: RepositoryInterface,
        agents: list[User],
        ws_manager: ConnectionManager,
        templates: object,
    ) -> None:
        self._repo = repo
        self._llm_semaphore = asyncio.Semaphore(1)
        self._agents = [BaseAgent(a, repo, self._llm_semaphore) for a in agents if a.agent_config]
        self._ws_manager = ws_manager
        self._templates = templates

    def reload_agents(self, agents: list[User]) -> None:
        """Replace in-memory agent instances after a model refresh."""
        self._agents = [BaseAgent(a, self._repo, self._llm_semaphore) for a in agents if a.agent_config]

    async def on_new_post(self, thread_id: str, post: Post) -> None:
        min_jitter = min(
            settings.agent_reply_jitter_min_seconds,
            settings.agent_reply_jitter_max_seconds,
        )
        max_jitter = max(
            settings.agent_reply_jitter_min_seconds,
            settings.agent_reply_jitter_max_seconds,
        )
        for agent in self._agents:
            jitter = random.uniform(min_jitter, max_jitter)
            asyncio.get_event_loop().call_later(
                jitter,
                lambda a=agent, t=thread_id, p=post: asyncio.ensure_future(
                    self._agent_respond(a, t, p)
                ),
            )

    async def _agent_respond(
        self, agent: BaseAgent, thread_id: str, post: Post
    ) -> None:
        try:
            reply = await agent.maybe_respond(thread_id, post)
            if reply:
                author = agent.user
                html = self._templates.get_template("fragments/post.html").render(  # type: ignore[union-attr]
                    post=reply, author=author, user=None
                )
                oob_html = f'<div hx-swap-oob="beforeend:#posts">{html}</div>'
                await self._ws_manager.broadcast(thread_id, oob_html)
                # re-trigger for other agents
                await self.on_new_post(thread_id, reply)
        except Exception:
            logger.exception(
                "agent_response_error",
                agent=agent.user.username,
                thread_id=thread_id,
            )
