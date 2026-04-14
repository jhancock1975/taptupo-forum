"""Unit tests for the discussion engine."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

import pytest

from app.agents.discussion import DiscussionEngine
from app.db.interface import RepositoryInterface
from app.models import AgentConfig, NewsItem, Post, Thread, User


class _FakeRepo(RepositoryInterface):
    def __init__(self) -> None:
        self.posts: list[Post] = []
        self.thread_activity: list[tuple[str, datetime]] = []

    async def create_post(self, post: Post) -> None:
        self.posts.append(post)

    async def update_thread_activity(self, thread_id: str, when: datetime) -> None:
        self.thread_activity.append((thread_id, when))

    # Unused
    async def create_user(self, user: User) -> None: ...
    async def get_user(self, user_id: str) -> User | None:
        return None

    async def get_user_by_username(self, username: str) -> User | None:
        return None

    async def list_agents(self) -> list[User]:
        return []

    async def create_thread(self, thread: Thread) -> None: ...
    async def get_thread(self, thread_id: str) -> Thread | None:
        return None

    async def list_threads(self, limit: int = 50) -> list[Thread]:
        return []

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


class _FakeAgent:
    def __init__(
        self,
        username: str,
        *,
        will_respond: bool,
        reply: str = "hello from agent",
    ) -> None:
        self.user = User(
            username=username,
            is_agent=True,
            agent_config=AgentConfig(
                model_id="m",
                persona_name=username,
                expertise_areas=["ml"],
                personality_traits=["x"],
                response_probability=1.0,
                system_prompt="s",
            ),
        )
        self._will_respond = will_respond
        self._reply = reply

    async def decide_to_respond(self, post_text: str) -> bool:
        return self._will_respond

    async def generate_response(self, post_text: str) -> str:
        return self._reply


def _post(author_id: str = "u-alice", thread_id: str = "t1") -> Post:
    return Post(
        thread_id=thread_id,
        author_id=author_id,
        content="Something about ml",
        created_at=datetime.now(UTC),
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_creates_replies_only_for_willing_agents() -> None:
    repo = _FakeRepo()
    agents = [
        _FakeAgent("agent_a", will_respond=True, reply="A says hi"),
        _FakeAgent("agent_b", will_respond=False),
        _FakeAgent("agent_c", will_respond=True, reply="C says yo"),
    ]
    engine = DiscussionEngine(
        repository=repo,
        agents=agents,
        min_delay=0.0,
        max_delay=0.0,
    )
    await engine.on_new_post(_post())
    contents = {p.content for p in repo.posts}
    assert contents == {"A says hi", "C says yo"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_does_not_reply_to_self() -> None:
    repo = _FakeRepo()
    agents = [_FakeAgent("agent_a", will_respond=True, reply="loop")]
    engine = DiscussionEngine(
        repository=repo,
        agents=agents,
        min_delay=0.0,
        max_delay=0.0,
    )
    await engine.on_new_post(_post(author_id=agents[0].user.user_id))
    assert repo.posts == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_bumps_thread_activity_when_reply_posted() -> None:
    repo = _FakeRepo()
    agents = [_FakeAgent("agent_a", will_respond=True)]
    engine = DiscussionEngine(
        repository=repo,
        agents=agents,
        min_delay=0.0,
        max_delay=0.0,
    )
    await engine.on_new_post(_post(thread_id="t42"))
    assert [tid for tid, _ in repo.thread_activity] == ["t42"]
