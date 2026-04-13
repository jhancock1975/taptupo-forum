"""Unit tests for the Reddit fetcher."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.news.reddit import RedditFetcher

FIX = Path(__file__).resolve().parents[2] / "fixtures" / "reddit" / "sample.json"


def _transport(payload: dict[str, object], *, status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reddit_fetch_parses_self_and_link_posts() -> None:
    payload = json.loads(FIX.read_text())
    fetcher = RedditFetcher(subreddits=["MachineLearning"], transport=_transport(payload))
    items = await fetcher.fetch()
    assert len(items) == 2
    assert all(i.source == "reddit" for i in items)
    urls = {i.url for i in items}
    assert "https://www.reddit.com/r/MachineLearning/comments/abc/xyz" in urls
    assert "https://arxiv.org/abs/2604.00003" in urls


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reddit_fetch_skips_failed_subreddits_but_continues() -> None:
    payload = json.loads(FIX.read_text())

    def handler(request: httpx.Request) -> httpx.Response:
        if "fails" in request.url.path:
            return httpx.Response(500, json={})
        return httpx.Response(200, json=payload)

    fetcher = RedditFetcher(
        subreddits=["fails", "MachineLearning"],
        transport=httpx.MockTransport(handler),
    )
    items = await fetcher.fetch()
    assert len(items) == 2


@pytest.mark.unit
def test_reddit_source_name() -> None:
    assert RedditFetcher().source_name == "reddit"
