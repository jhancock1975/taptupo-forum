"""Unit tests for the Guardian fetcher."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.news.guardian import GuardianFetcher

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "guardian" / "sample.json"


def _mock_transport(payload: dict[str, object], *, status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_guardian_fetch_parses_sample_fixture() -> None:
    payload = json.loads(FIXTURE.read_text())
    transport = _mock_transport(payload)
    fetcher = GuardianFetcher(api_key="test-key", transport=transport)

    items = await fetcher.fetch()

    assert len(items) == 2
    assert {i.source for i in items} == {"guardian"}
    titles = {i.title for i in items}
    assert "Large language models reshape science publishing" in titles
    assert all(i.url.startswith("https://www.theguardian.com/") for i in items)
    assert all(i.status == "new" for i in items)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_guardian_fetch_returns_empty_list_on_http_error() -> None:
    transport = _mock_transport({}, status=500)
    fetcher = GuardianFetcher(api_key="test-key", transport=transport)
    assert await fetcher.fetch() == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_guardian_fetch_returns_empty_list_on_transport_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    transport = httpx.MockTransport(handler)
    fetcher = GuardianFetcher(api_key="test-key", transport=transport)
    assert await fetcher.fetch() == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_guardian_fetch_skips_items_missing_required_fields() -> None:
    payload = {
        "response": {
            "results": [
                {"webTitle": "ok", "webUrl": "https://x/y"},
                {"webTitle": "", "webUrl": "https://x/z"},
                {"webUrl": "https://x/w"},
            ]
        }
    }
    transport = _mock_transport(payload)
    fetcher = GuardianFetcher(api_key="test-key", transport=transport)
    items = await fetcher.fetch()
    assert len(items) == 1
    assert items[0].title == "ok"


@pytest.mark.unit
def test_guardian_source_name_matches_literal() -> None:
    assert GuardianFetcher(api_key="x").source_name == "guardian"
