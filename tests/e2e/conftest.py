"""Fixtures for Playwright end-to-end tests.

Each test gets a fresh ``TestApp`` - an in-memory FastAPI instance
served by uvicorn on a random local port. The repository is a simple
dict-backed fake so the test can seed users, threads and posts directly
without touching DynamoDB.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from datetime import datetime
from typing import Literal

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.auth.routes import create_auth_router
from app.config import Settings
from app.db.interface import RepositoryInterface, UserExistsError
from app.forum.routes import create_forum_router
from app.main import _STATIC_DIR  # reuse project's static dir
from app.models import NewsItem, Post, Thread, User
from app.pages.routes import create_pages_router
from app.realtime.broker import Broker
from app.realtime.websocket import create_websocket_router


class InMemoryRepo(RepositoryInterface):
    """A minimal fake repository used only by e2e tests."""

    def __init__(self) -> None:
        self.users: dict[str, User] = {}
        self.threads: dict[str, Thread] = {}
        self.posts: list[Post] = []

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

    async def create_thread(self, thread: Thread) -> None:
        self.threads[thread.thread_id] = thread

    async def get_thread(self, thread_id: str) -> Thread | None:
        return self.threads.get(thread_id)

    async def list_threads(self, limit: int = 50) -> list[Thread]:
        return sorted(self.threads.values(), key=lambda t: t.last_activity_at, reverse=True)[:limit]

    async def update_thread_activity(self, thread_id: str, when: datetime) -> None:
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


def _build_app(repo: RepositoryInterface, settings: Settings, broker: Broker) -> FastAPI:
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
    app.include_router(create_auth_router(repo=repo, settings=settings))
    app.include_router(create_forum_router(repo=repo, settings=settings, broker=broker))
    app.include_router(create_pages_router(repo=repo, settings=settings))
    app.include_router(create_websocket_router(broker=broker))
    return app


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def live_app() -> Iterator[tuple[str, InMemoryRepo]]:
    """Spawn the FastAPI app on a background thread; yield (base_url, repo)."""
    repo = InMemoryRepo()
    broker = Broker()
    settings = Settings(_env_file=None, session_secret="e2e-secret")  # type: ignore[call-arg]  # pragma: allowlist secret
    app = _build_app(repo, settings, broker)

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait briefly for the server to bind.
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("uvicorn did not start")

    try:
        yield f"http://127.0.0.1:{port}", repo
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)
