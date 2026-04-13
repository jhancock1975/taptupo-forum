"""Guardian Open Platform news fetcher.

Docs: https://open-platform.theguardian.com/documentation/
"""

from __future__ import annotations

from typing import Any, cast

import httpx
import structlog

from app.models import NewsItem

_log = structlog.get_logger(__name__)
_ENDPOINT = "https://content.guardianapis.com/search"


class GuardianFetcher:
    """Fetch recent articles from The Guardian's Open Platform."""

    source_name = "guardian"

    def __init__(
        self,
        *,
        api_key: str,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 10.0,
        page_size: int = 20,
    ) -> None:
        self._api_key = api_key
        self._transport = transport
        self._timeout = timeout
        self._page_size = page_size

    async def fetch(self) -> list[NewsItem]:
        params: dict[str, str] = {
            "api-key": self._api_key,
            "show-fields": "trailText",
            "page-size": str(self._page_size),
            "order-by": "newest",
        }
        try:
            async with httpx.AsyncClient(
                transport=self._transport, timeout=self._timeout
            ) as client:
                resp = await client.get(_ENDPOINT, params=params)
            resp.raise_for_status()
            payload = cast(dict[str, Any], resp.json())
        except (httpx.HTTPError, ValueError) as exc:
            _log.warning("guardian.fetch_failed", error=str(exc))
            return []
        return _parse(payload)


def _parse(payload: dict[str, Any]) -> list[NewsItem]:
    results = payload.get("response", {}).get("results", [])
    if not isinstance(results, list):
        return []
    items: list[NewsItem] = []
    for entry in results:
        if not isinstance(entry, dict):
            continue
        title = entry.get("webTitle")
        url = entry.get("webUrl")
        if not isinstance(title, str) or not title or not isinstance(url, str) or not url:
            continue
        fields = entry.get("fields") or {}
        trail = fields.get("trailText", "") if isinstance(fields, dict) else ""
        items.append(
            NewsItem(
                source="guardian",
                title=title[:500],
                url=url,
                raw_content=trail if isinstance(trail, str) else "",
            )
        )
    return items
