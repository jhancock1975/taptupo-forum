from __future__ import annotations

import re

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.auth.utils import create_access_token, hash_password, verify_password
from app.models.schemas import User

router = APIRouter(prefix="/auth", tags=["auth"])

USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]{3,30}$")


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    templates = request.app.state.templates
    return templates.TemplateResponse(request, "login.html", {"error": None})


@router.post("/login", response_class=HTMLResponse, response_model=None)
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
) -> HTMLResponse | RedirectResponse:
    templates = request.app.state.templates
    repo = request.app.state.repo

    user = await repo.get_user_by_username(username)
    if (
        not user
        or not user.password_hash
        or not verify_password(password, user.password_hash)
    ):
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": "Invalid username or password"},
            status_code=401,
        )

    token = create_access_token(user.user_id, user.username)
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=86400,
    )
    return response


@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request) -> HTMLResponse:
    templates = request.app.state.templates
    return templates.TemplateResponse(request, "register.html", {"error": None})


@router.post("/register", response_class=HTMLResponse, response_model=None)
async def register(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    password_confirm: str = Form(...),
) -> HTMLResponse | RedirectResponse:
    templates = request.app.state.templates
    repo = request.app.state.repo

    if not USERNAME_RE.match(username):
        return templates.TemplateResponse(
            request,
            "register.html",
            {"error": "Username must be 3-30 alphanumeric characters or underscores"},
            status_code=400,
        )

    if len(password) < 8:
        return templates.TemplateResponse(
            request,
            "register.html",
            {"error": "Password must be at least 8 characters"},
            status_code=400,
        )

    if password != password_confirm:
        return templates.TemplateResponse(
            request,
            "register.html",
            {"error": "Passwords do not match"},
            status_code=400,
        )

    existing = await repo.get_user_by_username(username)
    if existing:
        return templates.TemplateResponse(
            request,
            "register.html",
            {"error": "Username already taken"},
            status_code=409,
        )

    user = User(
        username=username,
        password_hash=hash_password(password),
    )
    await repo.create_user(user)

    token = create_access_token(user.user_id, user.username)
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=86400,
    )
    return response


@router.post("/logout")
async def logout() -> RedirectResponse:
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie("access_token")
    return response
