"""Hacker News fetcher using the public Firebase API.

Docs: https://github.com/HackerNews/API
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import httpx
import structlog

from app.models import NewsItem

_log = structlog.get_logger(__name__)
_BASE = "https://hacker-news.firebaseio.com/v0"


class HackerNewsFetcher:
    """Fetch the current top Hacker News stories."""

    source_name = "hackernews"

    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 10.0,
        limit: int = 20,
    ) -> None:
        self._transport = transport
        self._timeout = timeout
        self._limit = limit

    async def fetch(self) -> list[NewsItem]:
        try:
            async with httpx.AsyncClient(
                transport=self._transport, timeout=self._timeout
            ) as client:
                top_resp = await client.get(f"{_BASE}/topstories.json")
                top_resp.raise_for_status()
                ids_raw = top_resp.json()
                if not isinstance(ids_raw, list):
                    return []
                ids = [int(x) for x in ids_raw[: self._limit] if isinstance(x, int)]
                detail_responses = await asyncio.gather(
                    *[client.get(f"{_BASE}/item/{item_id}.json") for item_id in ids],
                    return_exceptions=True,
                )
        except httpx.HTTPError as exc:
            _log.warning("hackernews.fetch_failed", error=str(exc))
            return []

        items: list[NewsItem] = []
        for resp in detail_responses:
            if isinstance(resp, BaseException):
                continue
            if resp.status_code != 200:
                continue
            try:
                entry = cast(dict[str, Any], resp.json())
            except ValueError:
                continue
            parsed = _parse_item(entry)
            if parsed is not None:
                items.append(parsed)
        return items


def _parse_item(entry: dict[str, Any]) -> NewsItem | None:
    if entry.get("type") != "story":
        return None
    title = entry.get("title")
    url = entry.get("url")
    if not isinstance(title, str) or not title:
        return None
    if not isinstance(url, str) or not url:
        return None
    score = entry.get("score", 0)
    return NewsItem(
        source="hackernews",
        title=title[:500],
        url=url,
        raw_content=f"score={score}",
    )
