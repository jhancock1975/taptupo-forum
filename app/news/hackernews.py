from __future__ import annotations

import json
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any, Protocol

import httpx
import structlog

from app.config import settings
from app.models.schemas import NewsItem
from app.news.content import extract_article_text

logger = structlog.get_logger()


try:
    from redis import asyncio as redis
except Exception:  # pragma: no cover - exercised only when redis isn't installed
    redis = None


class _AsyncJsonCache(Protocol):
    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str, ex: int | None = None) -> Any: ...


class HackerNewsFetcher:
    source_name: str = "hackernews"
    _base = "https://hacker-news.firebaseio.com/v0"
    _topstories_cache_key = "news:hackernews:topstories:v1"
    _item_cache_key_prefix = "news:hackernews:item:v1:"

    def __init__(self, cache: _AsyncJsonCache | None = None) -> None:
        self._cache = cache
        self._owns_cache = False
        self._cache_disabled = False
        self._topstories_ttl = max(settings.hn_topstories_cache_ttl_seconds, 0)
        self._item_ttl = max(settings.hn_story_cache_ttl_seconds, 0)

        if self._cache is None and settings.redis_url and redis is not None:
            # Redis client creation is lazy; the network connection happens
            # on the first command.
            self._cache = redis.Redis.from_url(
                settings.redis_url,
                decode_responses=True,
            )
            self._owns_cache = True
        elif self._cache is None and settings.redis_url and redis is None:
            logger.warning("hn_cache_unavailable", reason="redis_package_missing")

    async def _disable_cache(self, reason: str) -> None:
        if self._cache_disabled:
            return
        self._cache_disabled = True
        logger.warning("hn_cache_disabled", reason=reason)
        cache = self._cache
        if self._owns_cache and cache is not None and hasattr(cache, "aclose"):
            with suppress(Exception):
                await cache.aclose()  # type: ignore[union-attr]

    async def _cache_get_json(self, key: str) -> Any | None:
        if self._cache is None or self._cache_disabled:
            return None
        try:
            raw = await self._cache.get(key)
        except Exception:
            await self._disable_cache("read_failed")
            return None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def _cache_set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        if self._cache is None or self._cache_disabled or ttl_seconds <= 0:
            return
        try:
            await self._cache.set(key, json.dumps(value), ex=ttl_seconds)
        except Exception:
            await self._disable_cache("write_failed")

    @classmethod
    def _item_cache_key(cls, story_id: int) -> str:
        return f"{cls._item_cache_key_prefix}{story_id}"

    async def _article_text(
        self,
        client: httpx.AsyncClient,
        url: str,
        raw_content: str | None,
    ) -> str | None:
        if raw_content:
            return raw_content
        if not url.startswith(("http://", "https://")):
            return None
        if "news.ycombinator.com/item?id=" in url:
            return None

        try:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
        except httpx.HTTPError:
            logger.warning("hn_article_fetch_failed", url=url)
            return None

        headers = getattr(response, "headers", {})
        content_type = headers.get("content-type", "")
        if content_type and "html" not in content_type.lower():
            return None

        article_text = extract_article_text(getattr(response, "text", ""))
        if not article_text:
            return None
        return article_text[:5000]

    async def fetch(self) -> list[NewsItem]:
        items: list[NewsItem] = []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                cached_story_ids = await self._cache_get_json(
                    self._topstories_cache_key
                )
                story_ids: list[int]
                if isinstance(cached_story_ids, list) and cached_story_ids:
                    story_ids = [int(sid) for sid in cached_story_ids[:30]]
                    logger.info("hn_topstories_cache_hit", count=len(story_ids))
                else:
                    resp = await client.get(f"{self._base}/topstories.json")
                    resp.raise_for_status()
                    story_ids = resp.json()[:30]
                    await self._cache_set_json(
                        self._topstories_cache_key,
                        story_ids,
                        self._topstories_ttl,
                    )

                for sid in story_ids:
                    data: dict[str, Any] | None = None
                    cached_story = await self._cache_get_json(self._item_cache_key(sid))
                    if isinstance(cached_story, dict):
                        data = cached_story
                    else:
                        try:
                            sr = await client.get(f"{self._base}/item/{sid}.json")
                            sr.raise_for_status()
                            raw_data = sr.json()
                            if isinstance(raw_data, dict):
                                data = raw_data
                                await self._cache_set_json(
                                    self._item_cache_key(sid),
                                    data,
                                    self._item_ttl,
                                )
                        except httpx.HTTPError:
                            logger.warning("hn_item_fetch_failed", story_id=sid)
                            continue

                    if not data or data.get("type") != "story":
                        continue
                    url = data.get("url", f"https://news.ycombinator.com/item?id={sid}")
                    raw_content = await self._article_text(
                        client,
                        url,
                        data.get("text"),
                    )
                    items.append(
                        NewsItem(
                            source=self.source_name,
                            title=data.get("title", ""),
                            url=url,
                            raw_content=raw_content,
                            fetched_at=datetime.now(UTC),
                        )
                    )
        except httpx.HTTPError:
            logger.error("hn_top_stories_failed")
        return items
