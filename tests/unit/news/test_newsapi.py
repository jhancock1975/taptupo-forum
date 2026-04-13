"""Unit tests for the NewsAPI fetcher."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.news.newsapi import NewsAPIFetcher

FIX = Path(__file__).resolve().parents[2] / "fixtures" / "newsapi" / "sample.json"


def _transport(payload: dict[str, object], *, status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_newsapi_fetch_parses_articles() -> None:
    fetcher = NewsAPIFetcher(api_key="x", transport=_transport(json.loads(FIX.read_text())))
    items = await fetcher.fetch()
    assert len(items) == 2
    assert {i.source for i in items} == {"newsapi"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_newsapi_fetch_returns_empty_on_error() -> None:
    fetcher = NewsAPIFetcher(api_key="x", transport=_transport({}, status=429))
    assert await fetcher.fetch() == []


@pytest.mark.unit
def test_newsapi_source_name() -> None:
    assert NewsAPIFetcher(api_key="x").source_name == "newsapi"
