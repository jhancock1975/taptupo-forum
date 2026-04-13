"""Property test: /register rejects any username not matching the model regex.

The canonical username regex is ``^[A-Za-z0-9_]{3,32}$``. Any string that
violates that must produce a 422, never a 5xx and never a created user.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from app.auth.routes import create_auth_router
from app.config import Settings
from app.db.interface import RepositoryInterface, UserExistsError
from app.models import NewsItem, Post, Thread, User

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,32}$")


class _InMemoryRepo(RepositoryInterface):
    def __init__(self) -> None:
        self._users: dict[str, User] = {}

    async def create_user(self, user: User) -> None:
        if any(u.username == user.username for u in self._users.values()):
            raise UserExistsError(user.username)
        self._users[user.user_id] = user

    async def get_user(self, user_id: str) -> User | None:
        return self._users.get(user_id)

    async def get_user_by_username(self, username: str) -> User | None:
        for u in self._users.values():
            if u.username == username:
                return u
        return None

    async def list_agents(self) -> list[User]:
        return []

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


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(
        create_auth_router(
            repo=_InMemoryRepo(),
            settings=Settings(
                _env_file=None,  # type: ignore[call-arg]
                session_secret="test-secret",  # pragma: allowlist secret
            ),
        )
    )
    return TestClient(app)


@pytest.mark.hypothesis
@given(username=st.text(min_size=0, max_size=64).filter(lambda s: _USERNAME_RE.match(s) is None))
@settings(max_examples=100, deadline=None)
def test_register_rejects_invalid_usernames(username: str) -> None:
    client = _client()
    resp = client.post(
        "/register",
        data={
            "username": username,
            "password": "hunter2hunter2",  # pragma: allowlist secret
        },
        follow_redirects=False,
    )
    assert resp.status_code == 422, f"expected 422 for {username!r}, got {resp.status_code}"
