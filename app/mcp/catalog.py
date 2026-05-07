from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import httpx
import structlog

from app.config import settings
from app.news.guardian import GuardianClient

if TYPE_CHECKING:
    from app.db.interface import RepositoryInterface
    from app.news.aggregator import NewsAggregator

logger = structlog.get_logger()

_REQUEST_TIMEOUT = httpx.Timeout(12.0, connect=5.0)

_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "meta.list_tools",
        "description": "List all MCP tools available to forum agents.",
        "source": "internal",
        "requires_api_key": False,
    },
    {
        "name": "forum.news.refresh",
        "description": "Run the forum news aggregator now and report new item count.",
        "source": "internal",
        "requires_api_key": False,
    },
    {
        "name": "forum.news.latest",
        "description": "Get the latest forum news items by status.",
        "source": "internal",
        "requires_api_key": False,
    },
    {
        "name": "hn.top_stories",
        "description": "Fetch top Hacker News stories directly from the public API.",
        "source": "hackernews",
        "requires_api_key": False,
    },
    {
        "name": "wikipedia.summary",
        "description": "Fetch a concise summary for a topic from Wikipedia.",
        "source": "wikipedia",
        "requires_api_key": False,
    },
    {
        "name": "weather.current",
        "description": (
            "Get current weather using Open-Meteo by location "
            "or coordinates."
        ),
        "source": "open-meteo",
        "requires_api_key": False,
    },
    {
        "name": "newsapi.top_headlines",
        "description": "Fetch top headlines from NewsAPI (free tier key required).",
        "source": "newsapi",
        "requires_api_key": True,
        "api_key_env": "NEWSAPI_KEY",
    },
    {
        "name": "guardian.search",
        "description": "Search The Guardian API (free tier key required).",
        "source": "the-guardian",
        "requires_api_key": True,
        "api_key_env": "GUARDIAN_API_KEY",
    },
]

_TOOL_TRIGGERS: dict[str, list[str]] = {
    "weather.current": [
        "weather", "temperature", "forecast", "cold", "hot", "warm",
        "storm", "rain", "wind", "sunny", "outside", "climate today",
        "degrees", "humidity",
    ],
    "wikipedia.summary": [
        "who is", "what is", "when was", "where is", "history of",
        "tell me about", "explain", "define", "biography", "wiki",
        "wikipedia", "meaning of", "origin of",
    ],
    "hn.top_stories": [
        "hacker news", "hackernews", "hn", "tech news", "trending",
        "top stories", "what's new in tech", "startup news",
        "show hn", "ask hn",
    ],
    "newsapi.top_headlines": [
        "headlines", "breaking news", "latest news", "current events",
        "top news", "news today", "what happened", "in the news",
    ],
    "guardian.search": [
        "guardian", "uk news", "investigate", "reporting",
        "journalism", "the guardian", "british news",
    ],
    "forum.news.latest": [
        "forum news", "promoted", "recent threads",
        "what's been posted", "forum activity",
    ],
    "meta.list_tools": [
        "tool", "mcp", "available tools", "what tools",
        "list tools", "capabilities",
    ],
}

_RECENCY_INDICATORS: dict[str, list[str]] = {
    "hn.top_stories": ["hacker news stories", "show hn", "hn:"],
    "newsapi.top_headlines": ["headlines:", "breaking:"],
    "guardian.search": ["guardian.com", "theguardian"],
    "weather.current": ["temperature_c", "°c", "weather_code"],
    "wikipedia.summary": ["according to wikipedia", "wikipedia.org"],
}


class MCPToolCatalog:
    """In-process MCP-style tool catalog and dispatcher for forum agents."""

    def __init__(
        self,
        repo: RepositoryInterface | None,
        news_aggregator: NewsAggregator | None,
    ) -> None:
        self._repo = repo
        self._news_aggregator = news_aggregator
        self._definitions = {tool["name"]: dict(tool) for tool in _TOOL_DEFINITIONS}
        self._handlers = {
            "meta.list_tools": self._meta_list_tools,
            "forum.news.refresh": self._forum_news_refresh,
            "forum.news.latest": self._forum_news_latest,
            "hn.top_stories": self._hn_top_stories,
            "wikipedia.summary": self._wikipedia_summary,
            "weather.current": self._weather_current,
            "newsapi.top_headlines": self._newsapi_top_headlines,
            "guardian.search": self._guardian_search,
        }

    def list_tools(self) -> list[dict[str, Any]]:
        return [dict(tool) for tool in _TOOL_DEFINITIONS]

    def tool_names(self) -> list[str]:
        return [tool["name"] for tool in _TOOL_DEFINITIONS]

    async def invoke(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        handler = self._handlers.get(tool_name)
        if handler is None:
            return {
                "ok": False,
                "tool": tool_name,
                "error": "unknown_tool",
                "available_tools": self.tool_names(),
            }

        safe_args = arguments if arguments is not None else {}
        try:
            result = await handler(safe_args)
        except Exception:
            logger.exception("tool_execution_failed", tool=tool_name)
            return {
                "ok": False,
                "tool": tool_name,
                "error": "tool_execution_failed",
            }

        return {
            "ok": True,
            "tool": tool_name,
            "result": result,
        }

    def suggest_tools(
        self,
        text: str,
        preferred_tools: list[str] | None = None,
        recent_posts: list[str] | None = None,
        max_tools: int = 2,
    ) -> list[str]:
        text_lower = text.lower()
        scores: dict[str, float] = {}

        for tool_name, triggers in _TOOL_TRIGGERS.items():
            score = 0.0
            for trigger in triggers:
                if trigger in text_lower:
                    score += 1.0
            if score > 0:
                scores[tool_name] = score

        if preferred_tools:
            for tool_name in preferred_tools:
                scores[tool_name] = scores.get(tool_name, 0.0) + 0.5

        if recent_posts:
            combined_recent = " ".join(recent_posts).lower()
            for tool_name, indicators in _RECENCY_INDICATORS.items():
                if tool_name in scores:
                    for indicator in indicators:
                        if indicator in combined_recent:
                            scores[tool_name] -= 0.3
                            break

        scored = [(score, name) for name, score in scores.items() if score > 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [name for _, name in scored[:max_tools]]

    async def _meta_list_tools(self, arguments: dict[str, Any]) -> dict[str, Any]:
        _ = arguments
        return {"tools": self.list_tools()}

    async def _forum_news_refresh(self, arguments: dict[str, Any]) -> dict[str, Any]:
        _ = arguments
        if self._news_aggregator is None:
            return {"error": "news_aggregator_unavailable"}
        new_items = await self._news_aggregator.fetch_all()
        return {"new_items": new_items}

    async def _forum_news_latest(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._repo is None:
            return {"error": "repo_unavailable"}

        statuses = _normalize_statuses(arguments.get("statuses"))
        limit = _coerce_int(arguments.get("limit"), default=5, minimum=1, maximum=20)

        items = []
        for status in statuses:
            status_items = await self._repo.get_news_items_by_status(status)
            items.extend(status_items)

        items.sort(key=lambda item: item.fetched_at, reverse=True)

        unique_by_id: dict[str, Any] = {}
        for item in items:
            if item.item_id not in unique_by_id:
                unique_by_id[item.item_id] = item

        payload = []
        for item in list(unique_by_id.values())[:limit]:
            payload.append(
                {
                    "item_id": item.item_id,
                    "source": item.source,
                    "title": item.title,
                    "url": item.url,
                    "status": item.status,
                    "promoted_thread_id": item.promoted_thread_id,
                    "fetched_at": item.fetched_at.isoformat(),
                }
            )

        return {
            "statuses": statuses,
            "items": payload,
            "count": len(payload),
        }

    async def _hn_top_stories(self, arguments: dict[str, Any]) -> dict[str, Any]:
        limit = _coerce_int(arguments.get("limit"), default=5, minimum=1, maximum=20)

        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            ids_response = await client.get(
                "https://hacker-news.firebaseio.com/v0/topstories.json"
            )
            ids_response.raise_for_status()
            ids_payload = ids_response.json()
            if not isinstance(ids_payload, list):
                return {"stories": [], "count": 0}

            story_ids: list[int] = []
            for raw_id in ids_payload:
                story_id = _coerce_int(raw_id, default=-1, minimum=1, maximum=10**12)
                if story_id > 0:
                    story_ids.append(story_id)
                if len(story_ids) >= limit:
                    break

            stories = []
            for story_id in story_ids:
                try:
                    story_response = await client.get(
                        f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                    )
                    story_response.raise_for_status()
                except httpx.HTTPError:
                    continue

                story_data = story_response.json()
                if not isinstance(story_data, dict):
                    continue
                stories.append(
                    {
                        "id": story_data.get("id"),
                        "title": story_data.get("title", ""),
                        "url": story_data.get("url", ""),
                        "score": story_data.get("score", 0),
                    }
                )

        return {
            "stories": stories,
            "count": len(stories),
        }

    async def _wikipedia_summary(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = _as_text(arguments.get("query")) or _as_text(arguments.get("title"))
        if not query:
            return {
                "error": "missing_query",
                "hint": "Provide 'query' or 'title'.",
            }

        encoded = quote(query.replace(" ", "_"), safe="")
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"

        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            response = await client.get(summary_url)
            if response.status_code == 404:
                return {
                    "error": "not_found",
                    "query": query,
                }
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                return {"error": "invalid_response", "query": query}

        desktop = data.get("content_urls", {}).get("desktop", {})
        page_url = desktop.get("page", "") if isinstance(desktop, dict) else ""

        return {
            "title": data.get("title", query),
            "summary": data.get("extract", ""),
            "url": page_url,
        }

    async def _weather_current(self, arguments: dict[str, Any]) -> dict[str, Any]:
        latitude = _coerce_float(arguments.get("latitude"))
        longitude = _coerce_float(arguments.get("longitude"))
        location = _as_text(arguments.get("location"))

        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            if latitude is None or longitude is None:
                if not location:
                    return {
                        "error": "missing_location",
                        "hint": "Provide location or latitude/longitude.",
                    }
                geo_response = await client.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={
                        "name": location,
                        "count": 1,
                        "language": "en",
                        "format": "json",
                    },
                )
                geo_response.raise_for_status()
                geo_payload = geo_response.json()
                if not isinstance(geo_payload, dict):
                    return {"error": "invalid_geocoding_response"}
                results = geo_payload.get("results", [])
                if not isinstance(results, list) or not results:
                    return {"error": "location_not_found", "location": location}
                first = results[0]
                if not isinstance(first, dict):
                    return {"error": "invalid_geocoding_response"}
                latitude = _coerce_float(first.get("latitude"))
                longitude = _coerce_float(first.get("longitude"))
                if latitude is None or longitude is None:
                    return {"error": "invalid_geocoding_coordinates"}
                location = _as_text(first.get("name")) or location

            forecast_response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": (
                        "temperature_2m,relative_humidity_2m,"
                        "apparent_temperature,weather_code,wind_speed_10m"
                    ),
                    "timezone": "UTC",
                },
            )
            forecast_response.raise_for_status()
            forecast_payload = forecast_response.json()
            if not isinstance(forecast_payload, dict):
                return {"error": "invalid_weather_response"}

        current = forecast_payload.get("current", {})
        if not isinstance(current, dict):
            return {"error": "invalid_weather_response"}

        return {
            "location": location or "custom-coordinates",
            "latitude": latitude,
            "longitude": longitude,
            "temperature_c": current.get("temperature_2m"),
            "apparent_temperature_c": current.get("apparent_temperature"),
            "humidity_percent": current.get("relative_humidity_2m"),
            "weather_code": current.get("weather_code"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "observed_at": current.get("time"),
        }

    async def _newsapi_top_headlines(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not settings.newsapi_key:
            return {
                "error": "missing_api_key",
                "required_env": "NEWSAPI_KEY",
            }

        country = (_as_text(arguments.get("country")) or "us").lower()
        if len(country) != 2:
            country = "us"
        category = (_as_text(arguments.get("category")) or "technology").lower()
        limit = _coerce_int(arguments.get("limit"), default=5, minimum=1, maximum=20)

        params: dict[str, Any] = {
            "apiKey": settings.newsapi_key,
            "country": country,
            "pageSize": limit,
        }
        if category:
            params["category"] = category

        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            response = await client.get(
                "https://newsapi.org/v2/top-headlines",
                params=params,
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                return {"error": "invalid_newsapi_response"}

        if data.get("status") != "ok":
            return {
                "error": "newsapi_error",
                "message": data.get("message", "Unknown NewsAPI error"),
            }

        articles = []
        raw_articles = data.get("articles", [])
        if isinstance(raw_articles, list):
            for article in raw_articles:
                if not isinstance(article, dict):
                    continue
                source = article.get("source", {})
                source_name = ""
                if isinstance(source, dict):
                    source_name = _as_text(source.get("name")) or ""
                articles.append(
                    {
                        "title": _as_text(article.get("title")) or "",
                        "url": _as_text(article.get("url")) or "",
                        "source": source_name,
                        "published_at": _as_text(article.get("publishedAt")) or "",
                    }
                )

        return {
            "country": country,
            "category": category,
            "articles": articles,
            "count": len(articles),
        }

    async def _guardian_search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not settings.guardian_api_key:
            return {
                "error": "missing_api_key",
                "required_env": "GUARDIAN_API_KEY",
            }

        query = _as_text(arguments.get("query"))
        if not query:
            return {
                "error": "missing_query",
                "hint": "Provide 'query' for the Guardian search tool.",
            }
        limit = _coerce_int(arguments.get("limit"), default=5, minimum=1, maximum=20)
        client = GuardianClient(api_key=settings.guardian_api_key)

        try:
            raw_results = await client.search(
                query=query,
                limit=limit,
                show_fields=("trailText",),
            )
        except ValueError:
            return {"error": "invalid_guardian_response"}

        results = []
        for result in raw_results:
            fields = result.get("fields", {})
            trail = ""
            if isinstance(fields, dict):
                trail = _as_text(fields.get("trailText")) or ""
            results.append(
                {
                    "title": _as_text(result.get("webTitle")) or "",
                    "url": _as_text(result.get("webUrl")) or "",
                    "section": _as_text(result.get("sectionName")) or "",
                    "published_at": _as_text(result.get("webPublicationDate")) or "",
                    "trail_text": trail,
                }
            )

        return {
            "query": query,
            "results": results,
            "count": len(results),
        }


def _coerce_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_text(value: Any) -> str | None:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return None


def _normalize_statuses(raw: Any) -> list[str]:
    if isinstance(raw, str):
        parsed = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, list):
        parsed = []
        for item in raw:
            text = _as_text(item)
            if text:
                parsed.append(text)
    else:
        parsed = []

    if not parsed:
        return ["new", "promoted"]
    return parsed
