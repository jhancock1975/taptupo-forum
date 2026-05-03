from __future__ import annotations

from typing import Any

import httpx
import pytest

from app.models.schemas import NewsItem
from app.news import hackernews
from app.news.aggregator import NewsAggregator
from app.news.hackernews import HackerNewsFetcher
from app.news.interface import NewsFetcher


class AggregatorRepo:
    def __init__(self, existing_urls: set[str] | None = None) -> None:
        self.existing_urls = existing_urls or set()
        self.created: list[NewsItem] = []

    async def get_news_item_by_url(self, url: str) -> NewsItem | None:
        if url in self.existing_urls:
            return NewsItem(source="hn", title="existing", url=url)
        return None

    async def create_news_item(self, item: NewsItem) -> NewsItem:
        self.created.append(item)
        return item


class FakeFetcher:
    def __init__(
        self,
        source_name: str,
        items: list[NewsItem] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.source_name = source_name
        self.items = items or []
        self.error = error

    async def fetch(self) -> list[NewsItem]:
        if self.error:
            raise self.error
        return self.items


@pytest.mark.anyio
async def test_news_fetcher_protocol_accepts_fetcher_shape() -> None:
    fetcher = FakeFetcher("fake")

    assert isinstance(fetcher, NewsFetcher)


@pytest.mark.anyio
async def test_aggregator_creates_only_new_items_and_continues_after_errors() -> None:
    repo = AggregatorRepo(existing_urls={"https://existing"})
    aggregator = NewsAggregator(repo)  # type: ignore[arg-type]
    aggregator._fetchers = []
    new_item = NewsItem(source="fake", title="New", url="https://new")
    existing = NewsItem(source="fake", title="Existing", url="https://existing")

    aggregator.register_fetcher(FakeFetcher("good", [new_item, existing]))
    aggregator.register_fetcher(FakeFetcher("bad", error=RuntimeError("boom")))

    assert await aggregator.fetch_all() == 1
    assert repo.created == [new_item]


class FakeHNResponse:
    def __init__(self, payload: Any, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error

    def raise_for_status(self) -> None:
        if self.error:
            raise self.error

    def json(self) -> Any:
        return self.payload


class FakeHNClient:
    top_response = FakeHNResponse([1, 2, 3, 4])
    item_responses: dict[int, FakeHNResponse] = {}
    requested_urls: list[str] = []

    def __init__(self, timeout: int) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> FakeHNClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def get(self, url: str) -> FakeHNResponse:
        self.requested_urls.append(url)
        if url.endswith("/topstories.json"):
            return self.top_response
        story_id = int(url.rsplit("/", 1)[1].split(".", 1)[0])
        return self.item_responses[story_id]


@pytest.mark.anyio
async def test_hackernews_fetcher_builds_items_and_skips_bad_stories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeHNClient.requested_urls = []
    FakeHNClient.top_response = FakeHNResponse([1, 2, 3, 4])
    FakeHNClient.item_responses = {
        1: FakeHNResponse({"type": "story", "title": "Story", "url": "https://story", "text": "body"}),
        2: FakeHNResponse({"type": "story", "title": "Fallback"}),
        3: FakeHNResponse({"type": "job", "title": "Job"}),
        4: FakeHNResponse({}, error=httpx.HTTPError("item failed")),
    }
    monkeypatch.setattr(hackernews.httpx, "AsyncClient", FakeHNClient)

    items = await HackerNewsFetcher().fetch()

    assert [item.title for item in items] == ["Story", "Fallback"]
    assert items[0].url == "https://story"
    assert items[0].raw_content == "body"
    assert items[1].url == "https://news.ycombinator.com/item?id=2"
    assert all(item.source == "hackernews" for item in items)
    assert FakeHNClient.requested_urls[0].endswith("/topstories.json")


@pytest.mark.anyio
async def test_hackernews_fetcher_returns_empty_list_when_top_stories_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeHNClient.top_response = FakeHNResponse({}, error=httpx.HTTPError("top failed"))
    FakeHNClient.item_responses = {}
    monkeypatch.setattr(hackernews.httpx, "AsyncClient", FakeHNClient)

    assert await HackerNewsFetcher().fetch() == []
