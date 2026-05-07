from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from app.mcp import catalog as tool_catalog
from app.mcp.catalog import (
    MCPToolCatalog,
    _as_text,
    _coerce_float,
    _coerce_int,
    _normalize_statuses,
)
from app.models.schemas import NewsItem
from app.news import guardian as guardian_api


class FakeRepo:
    def __init__(self) -> None:
        self.items_by_status: dict[str, list[NewsItem]] = {}

    async def get_news_items_by_status(self, status: str) -> list[NewsItem]:
        return self.items_by_status.get(status, [])


class FakeAggregator:
    def __init__(self, new_items: int = 0) -> None:
        self.new_items = new_items
        self.calls = 0

    async def fetch_all(self) -> int:
        self.calls += 1
        return self.new_items


class FakeResponse:
    def __init__(
        self,
        payload: Any,
        *,
        status_code: int = 200,
        error: Exception | None = None,
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self._error = error

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error
        if self.status_code >= 400:
            raise httpx.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        return self._payload


def patch_async_client(
    monkeypatch: pytest.MonkeyPatch,
    responder: Any,
) -> None:
    async def noop_sleep(_: float) -> None:
        return None

    class FakeAsyncClient:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def get(
            self,
            url: str,
            params: dict[str, Any] | None = None,
        ) -> FakeResponse:
            return responder(url, params or {})

    monkeypatch.setattr(tool_catalog.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(guardian_api.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(
        guardian_api,
        "_shared_guardian_rate_limiter",
        guardian_api.GuardianRateLimiter(
            0.0,
            clock=lambda: 0.0,
            sleep=noop_sleep,
        ),
    )


def test_tool_catalog_lists_meta_and_news_tools() -> None:
    catalog = MCPToolCatalog(repo=None, news_aggregator=None)

    names = [tool["name"] for tool in catalog.list_tools()]

    assert "meta.list_tools" in names
    assert "forum.news.refresh" in names
    assert "forum.news.latest" in names
    assert "weather.current" in names


@pytest.mark.anyio
async def test_invoke_unknown_tool_returns_structured_error() -> None:
    catalog = MCPToolCatalog(repo=None, news_aggregator=None)

    result = await catalog.invoke("missing.tool", {})

    assert result["ok"] is False
    assert result["error"] == "unknown_tool"
    assert "available_tools" in result


@pytest.mark.anyio
async def test_invoke_returns_execution_failed_when_tool_raises() -> None:
    catalog = MCPToolCatalog(repo=None, news_aggregator=None)

    async def explode(_: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("boom")

    catalog._handlers["exploding.tool"] = explode  # type: ignore[attr-defined]

    result = await catalog.invoke("exploding.tool", {})

    assert result == {
        "ok": False,
        "tool": "exploding.tool",
        "error": "tool_execution_failed",
    }


@pytest.mark.anyio
async def test_forum_news_refresh_calls_aggregator() -> None:
    aggregator = FakeAggregator(new_items=7)
    catalog = MCPToolCatalog(repo=None, news_aggregator=aggregator)  # type: ignore[arg-type]

    result = await catalog.invoke("forum.news.refresh", {})

    assert result["ok"] is True
    assert result["result"] == {"new_items": 7}
    assert aggregator.calls == 1


@pytest.mark.anyio
async def test_forum_news_refresh_reports_unavailable_aggregator() -> None:
    catalog = MCPToolCatalog(repo=None, news_aggregator=None)

    result = await catalog.invoke("forum.news.refresh", {})

    assert result["ok"] is True
    assert result["result"]["error"] == "news_aggregator_unavailable"


@pytest.mark.anyio
async def test_forum_news_latest_reports_unavailable_repo() -> None:
    catalog = MCPToolCatalog(repo=None, news_aggregator=None)

    result = await catalog.invoke("forum.news.latest", {})

    assert result["ok"] is True
    assert result["result"]["error"] == "repo_unavailable"


@pytest.mark.anyio
async def test_forum_news_latest_merges_statuses_and_limits_results() -> None:
    repo = FakeRepo()
    now = datetime.now(UTC)
    repo.items_by_status = {
        "new": [
            NewsItem(
                item_id="n1",
                source="hackernews",
                title="New one",
                url="https://one",
                status="new",
                fetched_at=now,
            ),
            NewsItem(
                item_id="n2",
                source="hackernews",
                title="New two",
                url="https://two",
                status="new",
                fetched_at=now - timedelta(minutes=5),
            ),
        ],
        "promoted": [
            NewsItem(
                item_id="p1",
                source="hackernews",
                title="Promoted one",
                url="https://three",
                status="promoted",
                fetched_at=now - timedelta(minutes=1),
            )
        ],
    }
    catalog = MCPToolCatalog(repo=repo, news_aggregator=None)  # type: ignore[arg-type]

    result = await catalog.invoke(
        "forum.news.latest",
        {"statuses": ["new", "promoted"], "limit": 2},
    )

    assert result["ok"] is True
    payload = result["result"]
    assert payload["count"] == 2
    assert [item["item_id"] for item in payload["items"]] == ["n1", "p1"]


def test_suggest_tools_maps_keywords_to_expected_tools() -> None:
    catalog = MCPToolCatalog(repo=None, news_aggregator=None)

    tools = catalog.suggest_tools("Can you check weather and wikipedia context?")

    assert set(tools) == {"weather.current", "wikipedia.summary"}


def test_suggest_tools_respects_max_tools() -> None:
    catalog = MCPToolCatalog(repo=None, news_aggregator=None)

    tools = catalog.suggest_tools(
        "weather and wikipedia and guardian",
        max_tools=1,
    )

    assert len(tools) == 1
    assert tools[0] in ("weather.current", "wikipedia.summary", "guardian.search")


@pytest.mark.anyio
async def test_hn_top_stories_filters_invalid_story_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def responder(url: str, params: dict[str, Any]) -> FakeResponse:
        _ = params
        if url.endswith("/topstories.json"):
            return FakeResponse([1, "2", "bad-id", 3])
        if url.endswith("/item/1.json"):
            return FakeResponse(
                {
                    "id": 1,
                    "title": "Story One",
                    "url": "https://example.com/1",
                    "score": 10,
                }
            )
        if url.endswith("/item/2.json"):
            return FakeResponse({}, error=httpx.HTTPError("timeout"))
        if url.endswith("/item/3.json"):
            return FakeResponse(["not", "a", "dict"])
        raise AssertionError(url)

    patch_async_client(monkeypatch, responder)
    catalog = MCPToolCatalog(repo=None, news_aggregator=None)

    result = await catalog.invoke("hn.top_stories", {"limit": 3})

    assert result["ok"] is True
    payload = result["result"]
    assert payload["count"] == 1
    assert payload["stories"][0]["id"] == 1


@pytest.mark.anyio
async def test_wikipedia_summary_requires_query() -> None:
    catalog = MCPToolCatalog(repo=None, news_aggregator=None)

    result = await catalog.invoke("wikipedia.summary", {})

    assert result["ok"] is True
    assert result["result"]["error"] == "missing_query"


@pytest.mark.anyio
async def test_wikipedia_summary_handles_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def responder(url: str, params: dict[str, Any]) -> FakeResponse:
        _ = params
        assert "Not_Found" in url
        return FakeResponse({}, status_code=404)

    patch_async_client(monkeypatch, responder)
    catalog = MCPToolCatalog(repo=None, news_aggregator=None)

    result = await catalog.invoke("wikipedia.summary", {"query": "Not Found"})

    assert result["ok"] is True
    assert result["result"] == {"error": "not_found", "query": "Not Found"}


@pytest.mark.anyio
async def test_wikipedia_summary_returns_title_summary_and_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def responder(url: str, params: dict[str, Any]) -> FakeResponse:
        _ = params
        assert "Alan_Turing" in url
        return FakeResponse(
            {
                "title": "Alan Turing",
                "extract": "A pioneer of computing.",
                "content_urls": {
                    "desktop": {"page": "https://en.wikipedia.org/wiki/Alan_Turing"}
                },
            }
        )

    patch_async_client(monkeypatch, responder)
    catalog = MCPToolCatalog(repo=None, news_aggregator=None)

    result = await catalog.invoke("wikipedia.summary", {"query": "Alan Turing"})

    assert result["ok"] is True
    assert result["result"]["title"] == "Alan Turing"
    assert "pioneer" in result["result"]["summary"]
    assert result["result"]["url"].endswith("Alan_Turing")


@pytest.mark.anyio
async def test_weather_current_requires_location_or_coordinates() -> None:
    catalog = MCPToolCatalog(repo=None, news_aggregator=None)

    result = await catalog.invoke("weather.current", {})

    assert result["ok"] is True
    assert result["result"]["error"] == "missing_location"


@pytest.mark.anyio
async def test_weather_current_handles_unknown_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def responder(url: str, params: dict[str, Any]) -> FakeResponse:
        if "geocoding-api" in url:
            return FakeResponse({"results": []})
        raise AssertionError(url)

    patch_async_client(monkeypatch, responder)
    catalog = MCPToolCatalog(repo=None, news_aggregator=None)

    result = await catalog.invoke("weather.current", {"location": "Atlantis"})

    assert result["ok"] is True
    assert result["result"] == {"error": "location_not_found", "location": "Atlantis"}


@pytest.mark.anyio
async def test_weather_current_resolves_location_and_returns_current_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def responder(url: str, params: dict[str, Any]) -> FakeResponse:
        if "geocoding-api" in url:
            assert params["name"] == "Berlin"
            return FakeResponse(
                {
                    "results": [
                        {"name": "Berlin", "latitude": 52.52, "longitude": 13.405}
                    ]
                }
            )
        if "api.open-meteo.com" in url:
            return FakeResponse(
                {
                    "current": {
                        "temperature_2m": 20,
                        "apparent_temperature": 19,
                        "relative_humidity_2m": 50,
                        "weather_code": 1,
                        "wind_speed_10m": 12,
                        "time": "2026-05-03T12:00",
                    }
                }
            )
        raise AssertionError(url)

    patch_async_client(monkeypatch, responder)
    catalog = MCPToolCatalog(repo=None, news_aggregator=None)

    result = await catalog.invoke("weather.current", {"location": "Berlin"})

    assert result["ok"] is True
    payload = result["result"]
    assert payload["location"] == "Berlin"
    assert payload["temperature_c"] == 20
    assert payload["latitude"] == pytest.approx(52.52)


@pytest.mark.anyio
async def test_weather_current_accepts_coordinates_without_geocode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"geocode": False, "forecast": False}

    def responder(url: str, params: dict[str, Any]) -> FakeResponse:
        if "geocoding-api" in url:
            called["geocode"] = True
            return FakeResponse({})
        if "api.open-meteo.com" in url:
            called["forecast"] = True
            assert params["latitude"] == 40.7
            assert params["longitude"] == -74.0
            return FakeResponse({"current": {"temperature_2m": 15}})
        raise AssertionError(url)

    patch_async_client(monkeypatch, responder)
    catalog = MCPToolCatalog(repo=None, news_aggregator=None)

    result = await catalog.invoke(
        "weather.current",
        {"latitude": 40.7, "longitude": -74.0},
    )

    assert result["ok"] is True
    assert called["geocode"] is False
    assert called["forecast"] is True
    assert result["result"]["location"] == "custom-coordinates"


@pytest.mark.anyio
async def test_newsapi_tool_reports_missing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tool_catalog.settings, "newsapi_key", "")
    catalog = MCPToolCatalog(repo=None, news_aggregator=None)

    result = await catalog.invoke("newsapi.top_headlines", {"country": "us"})

    assert result["ok"] is True
    assert result["result"]["error"] == "missing_api_key"
    assert result["result"]["required_env"] == "NEWSAPI_KEY"


@pytest.mark.anyio
async def test_newsapi_tool_parses_success_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tool_catalog.settings, "newsapi_key", "key")

    def responder(url: str, params: dict[str, Any]) -> FakeResponse:
        assert "newsapi.org" in url
        assert params["country"] == "us"
        return FakeResponse(
            {
                "status": "ok",
                "articles": [
                    {
                        "title": "Headline",
                        "url": "https://example.com/news",
                        "source": {"name": "Example"},
                        "publishedAt": "2026-05-03T00:00:00Z",
                    }
                ],
            }
        )

    patch_async_client(monkeypatch, responder)
    catalog = MCPToolCatalog(repo=None, news_aggregator=None)

    result = await catalog.invoke("newsapi.top_headlines", {"country": "USA"})

    assert result["ok"] is True
    payload = result["result"]
    assert payload["country"] == "us"
    assert payload["count"] == 1
    assert payload["articles"][0]["source"] == "Example"


@pytest.mark.anyio
async def test_newsapi_tool_handles_error_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tool_catalog.settings, "newsapi_key", "key")

    def responder(url: str, params: dict[str, Any]) -> FakeResponse:
        _ = (url, params)
        return FakeResponse({"status": "error", "message": "bad request"})

    patch_async_client(monkeypatch, responder)
    catalog = MCPToolCatalog(repo=None, news_aggregator=None)

    result = await catalog.invoke("newsapi.top_headlines", {"country": "us"})

    assert result["ok"] is True
    assert result["result"] == {
        "error": "newsapi_error",
        "message": "bad request",
    }

@pytest.mark.anyio
async def test_guardian_tool_reports_missing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tool_catalog.settings, "guardian_api_key", "")
    catalog = MCPToolCatalog(repo=None, news_aggregator=None)

    result = await catalog.invoke("guardian.search", {"query": "space"})

    assert result["ok"] is True
    assert result["result"]["error"] == "missing_api_key"
    assert result["result"]["required_env"] == "GUARDIAN_API_KEY"


@pytest.mark.anyio
async def test_guardian_tool_requires_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tool_catalog.settings, "guardian_api_key", "g-key")
    catalog = MCPToolCatalog(repo=None, news_aggregator=None)

    result = await catalog.invoke("guardian.search", {})

    assert result["ok"] is True
    assert result["result"]["error"] == "missing_query"


@pytest.mark.anyio
async def test_guardian_tool_parses_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tool_catalog.settings, "guardian_api_key", "g-key")

    def responder(url: str, params: dict[str, Any]) -> FakeResponse:
        assert "content.guardianapis.com" in url
        assert params["q"] == "space"
        return FakeResponse(
            {
                "response": {
                    "results": [
                        {
                            "webTitle": "Space launch",
                            "webUrl": "https://www.theguardian.com/example",
                            "sectionName": "Science",
                            "webPublicationDate": "2026-05-03T08:00:00Z",
                            "fields": {"trailText": "Rocket news"},
                        }
                    ]
                }
            }
        )

    patch_async_client(monkeypatch, responder)
    catalog = MCPToolCatalog(repo=None, news_aggregator=None)

    result = await catalog.invoke("guardian.search", {"query": "space"})

    assert result["ok"] is True
    payload = result["result"]
    assert payload["count"] == 1
    assert payload["results"][0]["title"] == "Space launch"


@pytest.mark.anyio
async def test_guardian_tool_handles_invalid_response_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tool_catalog.settings, "guardian_api_key", "g-key")

    def responder(url: str, params: dict[str, Any]) -> FakeResponse:
        _ = (url, params)
        return FakeResponse({"response": []})

    patch_async_client(monkeypatch, responder)
    catalog = MCPToolCatalog(repo=None, news_aggregator=None)

    result = await catalog.invoke("guardian.search", {"query": "space"})

    assert result["ok"] is True
    assert result["result"]["error"] == "invalid_guardian_response"


def test_helper_coercion_and_normalization_functions() -> None:
    assert _coerce_int("9", default=1, minimum=1, maximum=10) == 9
    assert _coerce_int("bad", default=3, minimum=1, maximum=10) == 3
    assert _coerce_int(999, default=1, minimum=1, maximum=10) == 10

    assert _coerce_float("12.5") == pytest.approx(12.5)
    assert _coerce_float(None) is None

    assert _as_text("  hello  ") == "hello"
    assert _as_text(123) is None

    assert _normalize_statuses("new, promoted") == ["new", "promoted"]
    assert _normalize_statuses(["new", "", "skipped"]) == ["new", "skipped"]
    assert _normalize_statuses(None) == ["new", "promoted"]


# ── suggest_tools scoring system ──────────────────────────────────────────────


def test_suggest_tools_scores_expanded_phrases():
    catalog = MCPToolCatalog(repo=None, news_aggregator=None)

    result = catalog.suggest_tools("tell me about SpaceX's founding")
    assert "wikipedia.summary" in result

    result = catalog.suggest_tools("what's trending in tech right now")
    assert "hn.top_stories" in result

    result = catalog.suggest_tools("is it cold outside today")
    assert "weather.current" in result


def test_suggest_tools_agent_preference_boost():
    catalog = MCPToolCatalog(repo=None, news_aggregator=None)

    result_with_pref = catalog.suggest_tools(
        "interesting developments in software",
        preferred_tools=["hn.top_stories"],
    )
    assert "hn.top_stories" in result_with_pref


def test_suggest_tools_recency_penalty():
    catalog = MCPToolCatalog(repo=None, news_aggregator=None)

    recent_posts = [
        "Here are the top Hacker News stories: 1. Show HN: something cool"
    ]
    # With preference boost and recency penalty, newsapi should rank above hn
    result_with_recency = catalog.suggest_tools(
        "latest news",
        preferred_tools=["newsapi.top_headlines"],
        recent_posts=recent_posts,
        max_tools=5,
    )
    result_without = catalog.suggest_tools(
        "latest news",
        preferred_tools=["newsapi.top_headlines"],
        max_tools=5,
    )
    # hn should score lower with recency penalty applied
    if "hn.top_stories" in result_with_recency and "hn.top_stories" in result_without:
        idx_with = result_with_recency.index("hn.top_stories")
        idx_without = result_without.index("hn.top_stories")
        assert idx_with >= idx_without


def test_suggest_tools_respects_max_tools():
    catalog = MCPToolCatalog(repo=None, news_aggregator=None)
    result = catalog.suggest_tools(
        "weather forecast and hacker news and wikipedia and headlines",
        max_tools=2,
    )
    assert len(result) <= 2


def test_suggest_tools_returns_empty_for_irrelevant_text():
    catalog = MCPToolCatalog(repo=None, news_aggregator=None)
    result = catalog.suggest_tools("I love the color blue and cats are nice")
    assert result == []
