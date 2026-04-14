"""Forum HTTP routes.

Exposes the read side as JSON under ``/api`` (threads listing, thread
detail, agent directory) and the write side as a form-POST at
``/threads/{id}/posts`` so HTMX frontends in a later phase can wire a
``<form>`` directly without a separate API client. Auth is enforced for
writes only: the session cookie set by :mod:`app.auth.routes` is decoded
and its ``user_id`` becomes the post's ``author_id``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

import structlog
from fastapi import APIRouter, Cookie, Depends, Form, HTTPException, status
from pydantic import ValidationError
from starlette.responses import RedirectResponse, Response

from app.auth.routes import SESSION_COOKIE
from app.auth.sessions import SessionError, decode_session
from app.config import Settings
from app.db.interface import RepositoryInterface
from app.models import Post
from app.realtime.broker import Broker

_log = structlog.get_logger(__name__)


def create_forum_router(
    *,
    repo: RepositoryInterface,
    settings: Settings,
    broker: Broker | None = None,
) -> APIRouter:
    """Build an ``APIRouter`` bound to ``repo`` and ``settings``.

    When ``broker`` is provided, every successful post creation publishes
    a ``post.created`` event on topic ``thread_id`` so WebSocket clients
    subscribed to that thread see the message without polling.
    """
    router = APIRouter()

    def _repo() -> RepositoryInterface:
        return repo

    def _settings() -> Settings:
        return settings

    @router.get("/api/threads")
    async def list_threads(
        repository: RepositoryInterface = Depends(_repo),  # noqa: B008
    ) -> dict[str, list[dict[str, object]]]:
        threads = await repository.list_threads()
        return {"threads": [t.model_dump(mode="json") for t in threads]}

    @router.get("/api/threads/{thread_id}")
    async def get_thread(
        thread_id: str,
        repository: RepositoryInterface = Depends(_repo),  # noqa: B008
    ) -> dict[str, object]:
        thread = await repository.get_thread(thread_id)
        if thread is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="thread not found",
            )
        posts = await repository.get_posts_by_thread(thread_id)
        return {
            "thread": thread.model_dump(mode="json"),
            "posts": [p.model_dump(mode="json") for p in posts],
        }

    @router.get("/api/agents")
    async def list_agents(
        repository: RepositoryInterface = Depends(_repo),  # noqa: B008
    ) -> dict[str, list[dict[str, object]]]:
        agents = await repository.list_agents()
        return {
            "agents": [
                {
                    "user_id": a.user_id,
                    "username": a.username,
                    "persona_name": (a.agent_config.persona_name if a.agent_config else a.username),
                    "model_id": (a.agent_config.model_id if a.agent_config else None),
                    "expertise_areas": (
                        list(a.agent_config.expertise_areas) if a.agent_config else []
                    ),
                }
                for a in agents
            ]
        }

    @router.post("/threads/{thread_id}/posts")
    async def create_post(
        thread_id: str,
        content: Annotated[str, Form()],
        session_cookie: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
        repository: RepositoryInterface = Depends(_repo),  # noqa: B008
        config: Settings = Depends(_settings),  # noqa: B008
    ) -> Response:
        user_id = _require_user_id(session_cookie, config.session_secret)
        thread = await repository.get_thread(thread_id)
        if thread is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="thread not found",
            )
        try:
            post = Post(
                thread_id=thread_id,
                author_id=user_id,
                content=content,
                created_at=datetime.now(UTC),
            )
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="invalid post content",
            ) from exc
        await repository.create_post(post)
        await repository.update_thread_activity(thread_id, post.created_at)
        _log.info(
            "forum.post_created",
            thread_id=thread_id,
            post_id=post.post_id,
            author_id=user_id,
        )
        if broker is not None:
            await broker.publish(
                thread_id,
                {"type": "post.created", "post": post.model_dump(mode="json")},
            )
        return RedirectResponse(
            url=f"/threads/{thread_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    return router


def _require_user_id(session_cookie: str | None, secret: str) -> str:
    if session_cookie is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="authentication required",
        )
    try:
        payload = decode_session(session_cookie, secret)
    except SessionError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid session",
        ) from exc
    user_id = payload.get("user_id")
    if not isinstance(user_id, str):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid session payload",
        )
    return user_id
