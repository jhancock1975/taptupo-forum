"""FastAPI auth routes: register, login, logout.

Routes take their repository and settings from an explicit factory
(:func:`create_auth_router`) rather than reaching for globals. This makes
the router trivial to unit-test with an in-memory fake and keeps the
DB-swap seam clean.

Templates are intentionally minimal placeholder HTML; richer Jinja2
templates land in the frontend phase.
"""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Form, HTTPException, status
from pydantic import ValidationError
from starlette.responses import HTMLResponse, RedirectResponse, Response

from app.auth.passwords import hash_password, verify_password
from app.auth.sessions import encode_session
from app.config import Settings
from app.db.interface import RepositoryInterface, UserExistsError
from app.models import User

SESSION_COOKIE = "taptupo_session"
_log = structlog.get_logger(__name__)


_REGISTER_HTML = """<!doctype html>
<title>Register</title>
<h1>Register</h1>
<form method="post" action="/register">
  <label>Username <input name="username" required></label>
  <label>Password <input name="password" type="password" required></label>
  <button type="submit">Create account</button>
</form>
"""

_LOGIN_HTML = """<!doctype html>
<title>Log in</title>
<h1>Log in</h1>
<form method="post" action="/login">
  <label>Username <input name="username" required></label>
  <label>Password <input name="password" type="password" required></label>
  <button type="submit">Log in</button>
</form>
"""


def create_auth_router(*, repo: RepositoryInterface, settings: Settings) -> APIRouter:
    """Build an ``APIRouter`` bound to ``repo`` and ``settings``."""
    router = APIRouter()

    def _repo() -> RepositoryInterface:
        return repo

    def _settings() -> Settings:
        return settings

    @router.get("/register", response_class=HTMLResponse)
    async def register_form() -> str:
        return _REGISTER_HTML

    @router.get("/login", response_class=HTMLResponse)
    async def login_form() -> str:
        return _LOGIN_HTML

    @router.post("/register")
    async def register(
        username: Annotated[str, Form()],
        password: Annotated[str, Form()],
        repository: RepositoryInterface = Depends(_repo),  # noqa: B008
        config: Settings = Depends(_settings),  # noqa: B008
    ) -> Response:
        try:
            user = User(
                username=username,
                password_hash=hash_password(password),
            )
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="invalid username",
            ) from exc
        try:
            await repository.create_user(user)
        except UserExistsError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="username taken",
            ) from exc
        _log.info("user.registered", user_id=user.user_id, username=user.username)
        return _redirect_with_session(user, config.session_secret)

    @router.post("/login")
    async def login(
        username: Annotated[str, Form()],
        password: Annotated[str, Form()],
        repository: RepositoryInterface = Depends(_repo),  # noqa: B008
        config: Settings = Depends(_settings),  # noqa: B008
    ) -> Response:
        user = await repository.get_user_by_username(username)
        if user is None or user.password_hash is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid credentials",
            )
        if not verify_password(password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid credentials",
            )
        _log.info("user.logged_in", user_id=user.user_id, username=user.username)
        return _redirect_with_session(user, config.session_secret)

    @router.post("/logout")
    async def logout() -> Response:
        resp = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
        resp.delete_cookie(SESSION_COOKIE)
        return resp

    return router


def _redirect_with_session(user: User, secret: str) -> Response:
    token = encode_session(
        {"user_id": user.user_id, "username": user.username},
        secret,
    )
    resp = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    resp.set_cookie(
        SESSION_COOKIE,
        token,
        httponly=True,
        samesite="lax",
        secure=False,  # set True behind HTTPS in prod via reverse proxy
    )
    return resp
