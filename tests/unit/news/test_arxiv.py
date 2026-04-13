"""Unit tests for the arXiv fetcher."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from app.news.arxiv import ArxivFetcher

FIX = Path(__file__).resolve().parents[2] / "fixtures" / "arxiv" / "sample.atom"


def _transport(body: str, *, status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=body, headers={"content-type": "application/atom+xml"})

    return httpx.MockTransport(handler)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_arxiv_fetch_parses_atom_feed() -> None:
    body = FIX.read_text()
    fetcher = ArxivFetcher(transport=_transport(body))
    items = await fetcher.fetch()
    assert len(items) == 2
    assert {i.source for i in items} == {"arxiv"}
    titles = {i.title for i in items}
    assert "Scaling laws for emergent reasoning" in titles


@pytest.mark.unit
@pytest.mark.asyncio
async def test_arxiv_fetch_returns_empty_on_http_error() -> None:
    fetcher = ArxivFetcher(transport=_transport("", status=503))
    assert await fetcher.fetch() == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_arxiv_fetch_returns_empty_on_garbage_body() -> None:
    fetcher = ArxivFetcher(transport=_transport("not xml at all"))
    assert await fetcher.fetch() == []


@pytest.mark.unit
def test_arxiv_source_name() -> None:
    assert ArxivFetcher().source_name == "arxiv"
