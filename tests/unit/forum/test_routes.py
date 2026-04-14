"""Unit tests for app.forum.routes."""

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
from app.forum.routes import create_forum_router
from app.models import AgentConfig, NewsItem, Post, Thread, User

SECRET = "test-secret-forum"  # pragma: allowlist secret


class _InMemoryRepo(RepositoryInterface):
    def __init__(self) -> None:
        self.users: dict[str, User] = {}
        self.threads: dict[str, Thread] = {}
        self.posts: list[Post] = []
        self.activity: list[tuple[str, datetime]] = []

    async def create_user(self, user: User) -> None:
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

    async def create_thread(self, thread: Thread) -> None:
        self.threads[thread.thread_id] = thread

    async def get_thread(self, thread_id: str) -> Thread | None:
        return self.threads.get(thread_id)

    async def list_threads(self, limit: int = 50) -> list[Thread]:
        return sorted(
            self.threads.values(),
            key=lambda t: t.last_activity_at,
            reverse=True,
        )[:limit]

    async def update_thread_activity(self, thread_id: str, when: datetime) -> None:
        self.activity.append((thread_id, when))
        if thread_id in self.threads:
            self.threads[thread_id].last_activity_at = when

    async def create_post(self, post: Post) -> None:
        self.posts.append(post)

    async def get_post(self, post_id: str) -> Post | None:
        for p in self.posts:
            if p.post_id == post_id:
                return p
        return None

    async def get_posts_by_thread(self, thread_id: str) -> list[Post]:
        return sorted(
            [p for p in self.posts if p.thread_id == thread_id],
            key=lambda p: p.created_at,
        )

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


def _make_client() -> tuple[TestClient, _InMemoryRepo]:
    repo = _InMemoryRepo()
    settings = Settings(_env_file=None, session_secret=SECRET)  # type: ignore[call-arg]
    app = FastAPI()
    app.include_router(create_forum_router(repo=repo, settings=settings))
    return TestClient(app), repo


def _thread(title: str = "First", thread_id: str = "t-1") -> Thread:
    return Thread(
        thread_id=thread_id,
        title=title,
        source_type="human",
        created_by="u-creator",
    )


def _human(user_id: str = "u-alice", username: str = "alice") -> User:
    return User(
        user_id=user_id,
        username=username,
        password_hash="$argon2id$fake",  # pragma: allowlist secret
    )


def _agent(user_id: str = "u-agent", username: str = "agent_one") -> User:
    return User(
        user_id=user_id,
        username=username,
        is_agent=True,
        agent_config=AgentConfig(
            model_id="free/model-x",
            persona_name="Professor X",
            expertise_areas=["ml", "ai"],
            personality_traits=["curious"],
            response_probability=0.5,
            system_prompt="You are Professor X.",
        ),
    )


@pytest.mark.unit
def test_list_threads_returns_threads_newest_first() -> None:
    client, repo = _make_client()
    older = _thread(title="old", thread_id="t-old")
    older.last_activity_at = datetime(2020, 1, 1, tzinfo=UTC)
    newer = _thread(title="new", thread_id="t-new")
    newer.last_activity_at = datetime(2025, 1, 1, tzinfo=UTC)
    repo.threads[older.thread_id] = older
    repo.threads[newer.thread_id] = newer

    resp = client.get("/api/threads")
    assert resp.status_code == 200
    titles = [t["title"] for t in resp.json()["threads"]]
    assert titles == ["new", "old"]


@pytest.mark.unit
def test_get_thread_returns_thread_and_posts() -> None:
    client, repo = _make_client()
    th = _thread(thread_id="t-42")
    repo.threads[th.thread_id] = th
    repo.posts.append(
        Post(
            thread_id="t-42",
            author_id="u-alice",
            content="hi",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
    )

    resp = client.get("/api/threads/t-42")
    assert resp.status_code == 200
    body = resp.json()
    assert body["thread"]["thread_id"] == "t-42"
    assert len(body["posts"]) == 1
    assert body["posts"][0]["content"] == "hi"


@pytest.mark.unit
def test_get_thread_missing_returns_404() -> None:
    client, _ = _make_client()
    resp = client.get("/api/threads/nope")
    assert resp.status_code == 404


@pytest.mark.unit
def test_list_agents_returns_only_agent_users() -> None:
    client, repo = _make_client()
    repo.users["u-alice"] = _human()
    agent = _agent()
    repo.users[agent.user_id] = agent

    resp = client.get("/api/agents")
    assert resp.status_code == 200
    agents = resp.json()["agents"]
    assert len(agents) == 1
    assert agents[0]["username"] == "agent_one"
    assert agents[0]["persona_name"] == "Professor X"
    assert agents[0]["model_id"] == "free/model-x"
    assert agents[0]["expertise_areas"] == ["ml", "ai"]


@pytest.mark.unit
def test_post_to_thread_requires_authentication() -> None:
    client, repo = _make_client()
    repo.threads["t-1"] = _thread()
    resp = client.post(
        "/threads/t-1/posts",
        data={"content": "hello"},
        follow_redirects=False,
    )
    assert resp.status_code == 401


@pytest.mark.unit
def test_post_to_thread_with_invalid_session_returns_401() -> None:
    client, repo = _make_client()
    repo.threads["t-1"] = _thread()
    client.cookies.set(SESSION_COOKIE, "garbage-token")
    resp = client.post(
        "/threads/t-1/posts",
        data={"content": "hello"},
        follow_redirects=False,
    )
    assert resp.status_code == 401


@pytest.mark.unit
def test_post_to_missing_thread_returns_404() -> None:
    client, _ = _make_client()
    token = encode_session({"user_id": "u-alice", "username": "alice"}, SECRET)
    client.cookies.set(SESSION_COOKIE, token)
    resp = client.post(
        "/threads/nope/posts",
        data={"content": "hi"},
        follow_redirects=False,
    )
    assert resp.status_code == 404


@pytest.mark.unit
def test_post_to_thread_creates_post_and_bumps_activity() -> None:
    client, repo = _make_client()
    repo.threads["t-9"] = _thread(thread_id="t-9")
    token = encode_session({"user_id": "u-alice", "username": "alice"}, SECRET)
    client.cookies.set(SESSION_COOKIE, token)

    resp = client.post(
        "/threads/t-9/posts",
        data={"content": "a new reply"},
        follow_redirects=False,
    )

    assert resp.status_code == 303
    assert resp.headers["location"] == "/threads/t-9"
    assert len(repo.posts) == 1
    assert repo.posts[0].content == "a new reply"
    assert repo.posts[0].author_id == "u-alice"
    assert repo.posts[0].thread_id == "t-9"
    assert [tid for tid, _ in repo.activity] == ["t-9"]


@pytest.mark.unit
def test_post_to_thread_with_empty_content_returns_422() -> None:
    client, repo = _make_client()
    repo.threads["t-1"] = _thread()
    token = encode_session({"user_id": "u-alice", "username": "alice"}, SECRET)
    client.cookies.set(SESSION_COOKIE, token)
    resp = client.post(
        "/threads/t-1/posts",
        data={"content": ""},
        follow_redirects=False,
    )
    # Empty form field may be rejected at the FastAPI layer (422) or by
    # Pydantic (422) — both acceptable.
    assert resp.status_code == 422
