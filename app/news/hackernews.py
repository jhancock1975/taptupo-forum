from __future__ import annotations

from datetime import UTC, datetime

import httpx
import structlog

from app.models.schemas import NewsItem

logger = structlog.get_logger()


class HackerNewsFetcher:
    source_name: str = "hackernews"
    _base = "https://hacker-news.firebaseio.com/v0"

    async def fetch(self) -> list[NewsItem]:
        items: list[NewsItem] = []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(f"{self._base}/topstories.json")
                resp.raise_for_status()
                story_ids: list[int] = resp.json()[:30]

                for sid in story_ids:
                    try:
                        sr = await client.get(f"{self._base}/item/{sid}.json")
                        sr.raise_for_status()
                        data = sr.json()
                        if not data or data.get("type") != "story":
                            continue
                        url = data.get(
                            "url", f"https://news.ycombinator.com/item?id={sid}"
                        )
                        items.append(
                            NewsItem(
                                source=self.source_name,
                                title=data.get("title", ""),
                                url=url,
                                raw_content=data.get("text"),
                                fetched_at=datetime.now(UTC),
                            )
                        )
                    except httpx.HTTPError:
                        logger.warning("hn_item_fetch_failed", story_id=sid)
        except httpx.HTTPError:
            logger.error("hn_top_stories_failed")
        return items
