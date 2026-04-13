"""Unit tests for app.auth.routes.

A hand-rolled in-memory repository stands in for DynamoDB so these tests
stay fast and need no external services.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.auth.routes import SESSION_COOKIE, create_auth_router
from app.auth.sessions import decode_session
from app.config import Settings
from app.db.interface import RepositoryInterface, UserExistsError
from app.models import NewsItem, Post, Thread, User


class _InMemoryRepo(RepositoryInterface):
    def __init__(self) -> None:
        self.users: dict[str, User] = {}

    async def create_user(self, user: User) -> None:
        if any(u.username == user.username for u in self.users.values()):
            raise UserExistsError(user.username)
        self.users[user.user_id] = user

    async def get_user(self, user_id: str) -> User | None:
        return self.users.get(user_id)

    async def get_user_by_username(self, username: str) -> User | None:
        for u in self.users.values():
            if u.username == username:
                return u
        return None

    async def list_agents(self) -> list[User]:
        return [u for u in self.users.values() if u.is_agent]

    # Thread/Post/NewsItem methods unused by auth tests; return defaults.
    async def create_thread(self, thread: Thread) -> None: ...
    async def get_thread(self, thread_id: str) -> Thread | None:
        return None

    async def list_threads(self, limit: int = 50) -> list[Thread]:
        return []

    async def update_thread_activity(self, thread_id: str, when: datetime) -> None: ...
    async def create_post(self, post: Post) -> None: ...
    async def get_post(self, post_id: str) -> Post | None:
        return None

    async def get_posts_by_thread(self, thread_id: str) -> list[Post]:
        return []

    async def create_news_item(self, item: NewsItem) -> None: ...
    async def get_news_item(self, item_id: str) -> NewsItem | None:
        return None

    async def list_new_news_items(self, limit: int = 100) -> list[NewsItem]:
        return []

    async def update_news_item_status(
        self,
        item_id: str,
        status: Literal["new", "promoted", "skipped"],
        promoted_thread_id: str | None = None,
    ) -> None: ...
    async def news_item_exists_by_url(self, url: str) -> bool:
        return False


SECRET = "test-secret-do-not-use-in-prod"  # pragma: allowlist secret


def _make_client() -> tuple[TestClient, _InMemoryRepo]:
    repo = _InMemoryRepo()
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        session_secret=SECRET,
    )
    app = FastAPI()
    app.include_router(create_auth_router(repo=repo, settings=settings))
    return TestClient(app), repo


@pytest.mark.unit
def test_get_register_returns_html_form() -> None:
    client, _ = _make_client()
    resp = client.get("/register")
    assert resp.status_code == 200
    assert "form" in resp.text.lower()
    assert "username" in resp.text.lower()
    assert "password" in resp.text.lower()


@pytest.mark.unit
def test_get_login_returns_html_form() -> None:
    client, _ = _make_client()
    resp = client.get("/login")
    assert resp.status_code == 200
    assert "form" in resp.text.lower()


@pytest.mark.unit
def test_post_register_creates_user_and_sets_session_cookie() -> None:
    client, repo = _make_client()
    resp = client.post(
        "/register",
        data={"username": "alice_99", "password": "hunter2hunter2"},  # pragma: allowlist secret
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert SESSION_COOKIE in resp.cookies
    user = next(iter(repo.users.values()))
    assert user.username == "alice_99"
    assert user.password_hash is not None
    payload = decode_session(resp.cookies[SESSION_COOKIE], SECRET)
    assert payload["username"] == "alice_99"
    assert payload["user_id"] == user.user_id


@pytest.mark.unit
def test_post_register_rejects_taken_username() -> None:
    client, _ = _make_client()
    ok = client.post(
        "/register",
        data={"username": "bob", "password": "hunter2hunter2"},  # pragma: allowlist secret
        follow_redirects=False,
    )
    assert ok.status_code == 303
    dup = client.post(
        "/register",
        data={"username": "bob", "password": "hunter2hunter2"},  # pragma: allowlist secret
        follow_redirects=False,
    )
    assert dup.status_code == 409


@pytest.mark.unit
def test_post_register_rejects_invalid_username() -> None:
    client, _ = _make_client()
    resp = client.post(
        "/register",
        data={"username": "a b", "password": "hunter2hunter2"},  # pragma: allowlist secret
        follow_redirects=False,
    )
    assert resp.status_code == 422


@pytest.mark.unit
def test_post_login_with_correct_password_sets_cookie() -> None:
    client, _ = _make_client()
    client.post(
        "/register",
        data={"username": "carol", "password": "hunter2hunter2"},  # pragma: allowlist secret
        follow_redirects=False,
    )
    client.cookies.clear()
    resp = client.post(
        "/login",
        data={"username": "carol", "password": "hunter2hunter2"},  # pragma: allowlist secret
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert SESSION_COOKIE in resp.cookies


@pytest.mark.unit
def test_post_login_with_wrong_password_returns_401() -> None:
    client, _ = _make_client()
    client.post(
        "/register",
        data={"username": "dave", "password": "hunter2hunter2"},  # pragma: allowlist secret
        follow_redirects=False,
    )
    client.cookies.clear()
    resp = client.post(
        "/login",
        data={"username": "dave", "password": "wrong-password"},  # pragma: allowlist secret
        follow_redirects=False,
    )
    assert resp.status_code == 401
    assert SESSION_COOKIE not in resp.cookies


@pytest.mark.unit
def test_post_login_unknown_user_returns_401() -> None:
    client, _ = _make_client()
    resp = client.post(
        "/login",
        data={"username": "nobody", "password": "whatever"},  # pragma: allowlist secret
        follow_redirects=False,
    )
    assert resp.status_code == 401


@pytest.mark.unit
def test_post_logout_clears_cookie() -> None:
    client, _ = _make_client()
    client.post(
        "/register",
        data={"username": "erin", "password": "hunter2hunter2"},  # pragma: allowlist secret
        follow_redirects=False,
    )
    resp = client.post("/logout", follow_redirects=False)
    assert resp.status_code == 303
    # Cookie should be cleared: either absent from jar or explicitly expired.
    set_cookie = resp.headers.get("set-cookie", "")
    assert SESSION_COOKIE in set_cookie
    assert "Max-Age=0" in set_cookie or "expires=Thu, 01 Jan 1970" in set_cookie.lower()
