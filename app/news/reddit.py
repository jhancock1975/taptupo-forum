"""Reddit fetcher using the public JSON listings.

We hit `https://www.reddit.com/r/<sub>/new.json` with a descriptive
User-Agent. No OAuth is required for unauthenticated read-only access.
"""

from __future__ import annotations

from typing import Any, cast

import httpx
import structlog

from app.models import NewsItem

_log = structlog.get_logger(__name__)
_USER_AGENT = "taptupo-forum/0.1 (news aggregator)"


class RedditFetcher:
    """Fetch the newest posts from a list of subreddits."""

    source_name = "reddit"

    def __init__(
        self,
        *,
        subreddits: list[str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 10.0,
        limit: int = 25,
    ) -> None:
        self._subreddits = subreddits or ["MachineLearning", "technology"]
        self._transport = transport
        self._timeout = timeout
        self._limit = limit

    async def fetch(self) -> list[NewsItem]:
        items: list[NewsItem] = []
        async with httpx.AsyncClient(
            transport=self._transport,
            timeout=self._timeout,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            for sub in self._subreddits:
                url = f"https://www.reddit.com/r/{sub}/new.json?limit={self._limit}"
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    payload = cast(dict[str, Any], resp.json())
                except (httpx.HTTPError, ValueError) as exc:
                    _log.warning("reddit.fetch_failed", subreddit=sub, error=str(exc))
                    continue
                items.extend(_parse(payload))
        return items


def _parse(payload: dict[str, Any]) -> list[NewsItem]:
    children = payload.get("data", {}).get("children", [])
    if not isinstance(children, list):
        return []
    items: list[NewsItem] = []
    for child in children:
        if not isinstance(child, dict):
            continue
        data = child.get("data")
        if not isinstance(data, dict):
            continue
        title = data.get("title")
        permalink = data.get("permalink")
        url = data.get("url") if not data.get("is_self") else None
        if not isinstance(title, str) or not title:
            continue
        final_url: str | None = None
        if isinstance(url, str) and url:
            final_url = url
        elif isinstance(permalink, str) and permalink:
            final_url = f"https://www.reddit.com{permalink}"
        if not final_url:
            continue
        selftext = data.get("selftext", "")
        items.append(
            NewsItem(
                source="reddit",
                title=title[:500],
                url=final_url,
                raw_content=selftext if isinstance(selftext, str) else "",
            )
        )
    return items
