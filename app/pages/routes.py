"""HTML page routes rendered via Jinja2.

These live alongside — and complement — the JSON routes in
:mod:`app.forum.routes`. Pages read the current user from the session
cookie when present so the nav can render differently for logged-in
visitors, but none of them require auth (posting does; the form in
``thread.html`` POSTs to the auth-gated JSON route).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, status
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse

from app.auth.routes import SESSION_COOKIE
from app.auth.sessions import SessionError, decode_session
from app.config import Settings
from app.db.interface import RepositoryInterface

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


def create_pages_router(*, repo: RepositoryInterface, settings: Settings) -> APIRouter:
    """Build an ``APIRouter`` for server-rendered pages."""
    router = APIRouter()
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    def _repo() -> RepositoryInterface:
        return repo

    def _settings() -> Settings:
        return settings

    def _current_user(
        cookie: str | None,
        secret: str,
    ) -> dict[str, Any] | None:
        if cookie is None:
            return None
        try:
            return decode_session(cookie, secret)
        except SessionError:
            return None

    @router.get("/", response_class=HTMLResponse)
    async def index(
        request: Request,
        session_cookie: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
        repository: RepositoryInterface = Depends(_repo),  # noqa: B008
        config: Settings = Depends(_settings),  # noqa: B008
    ) -> HTMLResponse:
        threads = await repository.list_threads()
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "threads": threads,
                "current_user": _current_user(session_cookie, config.session_secret),
            },
        )

    @router.get("/threads/{thread_id}", response_class=HTMLResponse)
    async def thread_detail(
        thread_id: str,
        request: Request,
        session_cookie: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
        repository: RepositoryInterface = Depends(_repo),  # noqa: B008
        config: Settings = Depends(_settings),  # noqa: B008
    ) -> HTMLResponse:
        thread = await repository.get_thread(thread_id)
        if thread is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="thread not found",
            )
        posts = await repository.get_posts_by_thread(thread_id)
        return templates.TemplateResponse(
            request,
            "thread.html",
            {
                "thread": thread,
                "posts": posts,
                "current_user": _current_user(session_cookie, config.session_secret),
            },
        )

    @router.get("/agents", response_class=HTMLResponse)
    async def agents(
        request: Request,
        session_cookie: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
        repository: RepositoryInterface = Depends(_repo),  # noqa: B008
        config: Settings = Depends(_settings),  # noqa: B008
    ) -> HTMLResponse:
        agent_users = await repository.list_agents()
        return templates.TemplateResponse(
            request,
            "agents.html",
            {
                "agents": agent_users,
                "current_user": _current_user(session_cookie, config.session_secret),
            },
        )

    return router
