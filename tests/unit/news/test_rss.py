"""Unit tests for the RSS fetcher."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from app.news.rss import RSSFetcher

FIX = Path(__file__).resolve().parents[2] / "fixtures" / "rss" / "sample.xml"


def _transport(body: str, *, status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=body)

    return httpx.MockTransport(handler)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rss_fetch_parses_rss20_feed() -> None:
    fetcher = RSSFetcher(
        feed_urls=["https://example.com/feed"], transport=_transport(FIX.read_text())
    )
    items = await fetcher.fetch()
    assert len(items) == 2
    assert all(i.source == "rss" for i in items)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rss_fetch_skips_failed_feeds() -> None:
    body = FIX.read_text()

    def handler(request: httpx.Request) -> httpx.Response:
        if "broken" in str(request.url):
            return httpx.Response(500)
        return httpx.Response(200, text=body)

    fetcher = RSSFetcher(
        feed_urls=["https://example.com/broken", "https://example.com/ok"],
        transport=httpx.MockTransport(handler),
    )
    items = await fetcher.fetch()
    assert len(items) == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rss_fetch_handles_garbage_xml() -> None:
    fetcher = RSSFetcher(feed_urls=["https://x"], transport=_transport("<<<not xml"))
    assert await fetcher.fetch() == []


@pytest.mark.unit
def test_rss_source_name() -> None:
    assert RSSFetcher(feed_urls=[]).source_name == "rss"
