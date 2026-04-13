"""Unit tests for the news aggregator scheduler."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import pytest

from app.db.interface import RepositoryInterface
from app.models import NewsItem, Post, Thread, User
from app.news.aggregator import NewsAggregator


class _FakeRepo(RepositoryInterface):
    def __init__(self, *, existing_urls: set[str] | None = None) -> None:
        self._existing_urls = existing_urls or set()
        self.created: list[NewsItem] = []

    async def news_item_exists_by_url(self, url: str) -> bool:
        return url in self._existing_urls

    async def create_news_item(self, item: NewsItem) -> None:
        self._existing_urls.add(item.url)
        self.created.append(item)

    # Unused methods below.
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

    async def update_thread_activity(self, thread_id: str, when: datetime) -> None: ...
    async def create_post(self, post: Post) -> None: ...
    async def get_post(self, post_id: str) -> Post | None:
        return None

    async def get_posts_by_thread(self, thread_id: str) -> list[Post]:
        return []

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


class _FakeFetcher:
    def __init__(self, source: str, items: list[NewsItem]) -> None:
        self.source_name = source
        self._items = items
        self.calls = 0

    async def fetch(self) -> list[NewsItem]:
        self.calls += 1
        return list(self._items)


class _ExplodingFetcher:
    source_name = "broken"

    async def fetch(self) -> list[NewsItem]:
        raise RuntimeError("boom")


def _item(url: str, source: str = "guardian") -> NewsItem:
    return NewsItem(source=source, title="T", url=url, raw_content="")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_aggregator_persists_new_items() -> None:
    repo = _FakeRepo()
    agg = NewsAggregator(
        repository=repo,
        fetchers=[_FakeFetcher("guardian", [_item("https://a"), _item("https://b")])],
    )
    await agg.run_once()
    assert {i.url for i in repo.created} == {"https://a", "https://b"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_aggregator_dedupes_against_existing_urls() -> None:
    repo = _FakeRepo(existing_urls={"https://a"})
    agg = NewsAggregator(
        repository=repo,
        fetchers=[_FakeFetcher("guardian", [_item("https://a"), _item("https://b")])],
    )
    await agg.run_once()
    assert [i.url for i in repo.created] == ["https://b"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_aggregator_dedupes_within_single_batch() -> None:
    repo = _FakeRepo()
    agg = NewsAggregator(
        repository=repo,
        fetchers=[
            _FakeFetcher("guardian", [_item("https://a")]),
            _FakeFetcher("hackernews", [_item("https://a", source="hackernews")]),
        ],
    )
    await agg.run_once()
    assert [i.url for i in repo.created] == ["https://a"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_aggregator_continues_past_failing_fetcher() -> None:
    repo = _FakeRepo()
    agg = NewsAggregator(
        repository=repo,
        fetchers=[
            _ExplodingFetcher(),
            _FakeFetcher("guardian", [_item("https://a")]),
        ],
    )
    await agg.run_once()
    assert [i.url for i in repo.created] == ["https://a"]
