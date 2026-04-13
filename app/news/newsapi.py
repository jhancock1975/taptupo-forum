"""NewsAPI.org fetcher.

Docs: https://newsapi.org/docs
"""

from __future__ import annotations

from typing import Any, cast

import httpx
import structlog

from app.models import NewsItem

_log = structlog.get_logger(__name__)
_ENDPOINT = "https://newsapi.org/v2/top-headlines"


class NewsAPIFetcher:
    """Fetch top headlines from NewsAPI.org."""

    source_name = "newsapi"

    def __init__(
        self,
        *,
        api_key: str,
        category: str = "technology",
        language: str = "en",
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 10.0,
        page_size: int = 20,
    ) -> None:
        self._api_key = api_key
        self._category = category
        self._language = language
        self._transport = transport
        self._timeout = timeout
        self._page_size = page_size

    async def fetch(self) -> list[NewsItem]:
        params = {
            "apiKey": self._api_key,
            "category": self._category,
            "language": self._language,
            "pageSize": str(self._page_size),
        }
        try:
            async with httpx.AsyncClient(
                transport=self._transport, timeout=self._timeout
            ) as client:
                resp = await client.get(_ENDPOINT, params=params)
            resp.raise_for_status()
            payload = cast(dict[str, Any], resp.json())
        except (httpx.HTTPError, ValueError) as exc:
            _log.warning("newsapi.fetch_failed", error=str(exc))
            return []
        return _parse(payload)


def _parse(payload: dict[str, Any]) -> list[NewsItem]:
    articles = payload.get("articles", [])
    if not isinstance(articles, list):
        return []
    items: list[NewsItem] = []
    for article in articles:
        if not isinstance(article, dict):
            continue
        title = article.get("title")
        url = article.get("url")
        if not isinstance(title, str) or not title or not isinstance(url, str) or not url:
            continue
        description = article.get("description", "") or ""
        items.append(
            NewsItem(
                source="newsapi",
                title=title[:500],
                url=url,
                raw_content=description if isinstance(description, str) else "",
            )
        )
    return items
