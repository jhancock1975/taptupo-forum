"""Unit tests for app.agents.news_agent."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import pytest

from app.agents.news_agent import NewsAgent
from app.db.interface import RepositoryInterface
from app.models import NewsItem, Post, Thread, User


class _FakeRepo(RepositoryInterface):
    def __init__(self, items: list[NewsItem]) -> None:
        self._items = items
        self.threads: list[Thread] = []
        self.statuses: list[tuple[str, str, str | None]] = []

    async def list_new_news_items(self, limit: int = 100) -> list[NewsItem]:
        return list(self._items)

    async def create_thread(self, thread: Thread) -> None:
        self.threads.append(thread)

    async def update_news_item_status(
        self,
        item_id: str,
        status: Literal["new", "promoted", "skipped"],
        promoted_thread_id: str | None = None,
    ) -> None:
        self.statuses.append((item_id, status, promoted_thread_id))

    # Unused
    async def create_user(self, user: User) -> None: ...
    async def get_user(self, user_id: str) -> User | None:
        return None

    async def get_user_by_username(self, username: str) -> User | None:
        return None

    async def list_agents(self) -> list[User]:
        return []

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

    async def news_item_exists_by_url(self, url: str) -> bool:
        return False


class _FakeLLM:
    """Returns YES iff the post contains the word "important"."""

    async def ainvoke(self, prompt: str) -> object:
        class _R:
            content = "YES" if "important" in prompt.lower() else "NO"

        return _R()


def _item(title: str, source: str = "guardian") -> NewsItem:
    return NewsItem(
        source=source,  # type: ignore[arg-type]
        title=title,
        url=f"https://example.com/{title.replace(' ', '-')}",
        raw_content=f"content of {title}",
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_news_agent_promotes_relevant_and_skips_others() -> None:
    items = [_item("boring weather"), _item("important finding")]
    repo = _FakeRepo(items)
    agent = NewsAgent(repository=repo, llm=_FakeLLM(), creator_user_id="u-news-agent")
    await agent.run_once()

    assert len(repo.threads) == 1
    assert repo.threads[0].title == "important finding"
    assert repo.threads[0].source_type == "guardian"
    assert repo.threads[0].source_url == items[1].url
    assert repo.threads[0].created_by == "u-news-agent"
    # Both items had their statuses updated.
    by_id = {iid: (st, tid) for iid, st, tid in repo.statuses}
    assert by_id[items[0].item_id] == ("skipped", None)
    promoted_status, thread_id = by_id[items[1].item_id]
    assert promoted_status == "promoted"
    assert thread_id == repo.threads[0].thread_id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_news_agent_no_op_with_no_items() -> None:
    repo = _FakeRepo([])
    agent = NewsAgent(repository=repo, llm=_FakeLLM(), creator_user_id="u")
    await agent.run_once()
    assert repo.threads == []
    assert repo.statuses == []
