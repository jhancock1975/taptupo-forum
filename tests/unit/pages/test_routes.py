"""Unit tests for app.pages.routes (Jinja2 HTML pages)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.auth.routes import SESSION_COOKIE
from app.auth.sessions import encode_session
from app.config import Settings
from app.db.interface import RepositoryInterface
from app.models import AgentConfig, NewsItem, Post, Thread, User
from app.pages.routes import create_pages_router

SECRET = "test-secret-pages"  # pragma: allowlist secret


class _Repo(RepositoryInterface):
    def __init__(self) -> None:
        self.threads: dict[str, Thread] = {}
        self.posts: list[Post] = []
        self.users: dict[str, User] = {}

    async def create_user(self, user: User) -> None:
        self.users[user.user_id] = user

    async def get_user(self, user_id: str) -> User | None:
        return self.users.get(user_id)

    async def get_user_by_username(self, username: str) -> User | None:
        return None

    async def list_agents(self) -> list[User]:
        return [u for u in self.users.values() if u.is_agent]

    async def create_thread(self, thread: Thread) -> None:
        self.threads[thread.thread_id] = thread

    async def get_thread(self, thread_id: str) -> Thread | None:
        return self.threads.get(thread_id)

    async def list_threads(self, limit: int = 50) -> list[Thread]:
        return list(self.threads.values())

    async def update_thread_activity(self, thread_id: str, when: datetime) -> None: ...

    async def create_post(self, post: Post) -> None:
        self.posts.append(post)

    async def get_post(self, post_id: str) -> Post | None:
        return None

    async def get_posts_by_thread(self, thread_id: str) -> list[Post]:
        return [p for p in self.posts if p.thread_id == thread_id]

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


def _client() -> tuple[TestClient, _Repo]:
    repo = _Repo()
    settings = Settings(_env_file=None, session_secret=SECRET)  # type: ignore[call-arg]
    app = FastAPI()
    app.include_router(create_pages_router(repo=repo, settings=settings))
    return TestClient(app), repo


@pytest.mark.unit
def test_index_renders_empty_state_when_no_threads() -> None:
    client, _ = _client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert "No threads yet" in resp.text
    assert "taptupo" in resp.text.lower()


@pytest.mark.unit
def test_index_lists_threads() -> None:
    client, repo = _client()
    repo.threads["t-1"] = Thread(
        thread_id="t-1",
        title="hello world",
        source_type="human",
        created_by="u-1",
    )
    resp = client.get("/")
    assert resp.status_code == 200
    assert "hello world" in resp.text
    assert "/threads/t-1" in resp.text


@pytest.mark.unit
def test_thread_detail_renders_posts_and_reply_prompt_for_anonymous() -> None:
    client, repo = _client()
    repo.threads["t-1"] = Thread(
        thread_id="t-1",
        title="a topic",
        source_type="human",
        created_by="u-1",
    )
    repo.posts.append(
        Post(
            thread_id="t-1",
            author_id="u-alice",
            content="first reply",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
    )
    resp = client.get("/threads/t-1")
    assert resp.status_code == 200
    assert "a topic" in resp.text
    assert "first reply" in resp.text
    assert "Log in" in resp.text  # anonymous prompt
    # WebSocket wiring present
    assert 'ws-connect="/ws/threads/t-1"' in resp.text


@pytest.mark.unit
def test_thread_detail_shows_reply_form_for_authenticated_user() -> None:
    client, repo = _client()
    repo.threads["t-1"] = Thread(
        thread_id="t-1",
        title="topic",
        source_type="human",
        created_by="u-1",
    )
    token = encode_session({"user_id": "u-alice", "username": "alice"}, SECRET)
    client.cookies.set(SESSION_COOKIE, token)
    resp = client.get("/threads/t-1")
    assert resp.status_code == 200
    assert "<textarea" in resp.text
    assert "alice" in resp.text


@pytest.mark.unit
def test_thread_detail_404_when_missing() -> None:
    client, _ = _client()
    resp = client.get("/threads/nope")
    assert resp.status_code == 404


@pytest.mark.unit
def test_agents_page_lists_agents() -> None:
    client, repo = _client()
    repo.users["u-a"] = User(
        user_id="u-a",
        username="bot_one",
        is_agent=True,
        agent_config=AgentConfig(
            model_id="free/x",
            persona_name="Doctor Forum",
            expertise_areas=["ml"],
            personality_traits=["calm"],
            response_probability=0.5,
            system_prompt="...",
        ),
    )
    resp = client.get("/agents")
    assert resp.status_code == 200
    assert "Doctor Forum" in resp.text
    assert "bot_one" in resp.text
    assert "ml" in resp.text
