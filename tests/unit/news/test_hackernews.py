"""Unit tests for the HackerNews fetcher."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.news.hackernews import HackerNewsFetcher

FIX = Path(__file__).resolve().parents[2] / "fixtures" / "hackernews"


def _transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/topstories.json"):
            return httpx.Response(200, json=json.loads((FIX / "topstories.json").read_text()))
        # /v0/item/{id}.json
        item_id = path.rsplit("/", 1)[-1].replace(".json", "")
        fixture = FIX / f"item_{item_id}.json"
        if fixture.exists():
            return httpx.Response(200, json=json.loads(fixture.read_text()))
        return httpx.Response(404, json={})

    return httpx.MockTransport(handler)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hackernews_fetch_filters_to_stories_with_urls() -> None:
    fetcher = HackerNewsFetcher(transport=_transport(), limit=5)
    items = await fetcher.fetch()
    # 11003 is a job post -> excluded; only two stories remain.
    assert len(items) == 2
    assert {i.source for i in items} == {"hackernews"}
    titles = {i.title for i in items}
    assert "New language model beats GPT-5" in titles


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hackernews_fetch_returns_empty_on_topstories_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={})

    fetcher = HackerNewsFetcher(transport=httpx.MockTransport(handler))
    assert await fetcher.fetch() == []


@pytest.mark.unit
def test_hackernews_source_name() -> None:
    assert HackerNewsFetcher().source_name == "hackernews"
