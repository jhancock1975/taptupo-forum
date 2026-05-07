from __future__ import annotations

import asyncio
import time
import xml.etree.ElementTree as ET
from contextlib import suppress
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx
import structlog

from app.config import settings
from app.models.schemas import NewsItem

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = structlog.get_logger()

_REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_ATOM_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}
_DEFAULT_REQUEST_INTERVAL_SECONDS = 3.0


def _clean_text(text: str | None) -> str:
    return " ".join((text or "").split())


class ArxivRateLimiter:
    def __init__(
        self,
        min_interval_seconds: float,
        *,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._min_interval_seconds = min_interval_seconds
        self._clock = clock or time.monotonic
        self._sleep = sleep or asyncio.sleep
        self._lock = asyncio.Lock()
        self._next_allowed_at = 0.0

    async def wait_turn(self) -> None:
        async with self._lock:
            now = self._clock()
            if now < self._next_allowed_at:
                await self._sleep(self._next_allowed_at - now)
                now = self._clock()
            self._next_allowed_at = now + self._min_interval_seconds


_shared_arxiv_rate_limiter = ArxivRateLimiter(
    min_interval_seconds=max(
        settings.arxiv_request_interval_seconds,
        _DEFAULT_REQUEST_INTERVAL_SECONDS,
    )
)


class ArxivFetcher:
    source_name = "arxiv"
    _base_url = "https://export.arxiv.org/api/query"

    def __init__(self, rate_limiter: ArxivRateLimiter | None = None) -> None:
        self._rate_limiter = rate_limiter or _shared_arxiv_rate_limiter

    async def fetch(self) -> list[NewsItem]:
        params = {
            "search_query": settings.arxiv_search_query,
            "start": 0,
            "max_results": settings.arxiv_max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        await self._rate_limiter.wait_turn()
        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
                response = await client.get(
                    self._base_url,
                    params=params,
                    follow_redirects=True,
                )
                response.raise_for_status()
        except httpx.HTTPError:
            logger.exception("arxiv_fetch_failed")
            return []

        try:
            root = ET.fromstring(response.text)
        except ET.ParseError:
            logger.warning("arxiv_feed_invalid")
            return []

        items: list[NewsItem] = []
        for entry in root.findall("atom:entry", _ATOM_NAMESPACE):
            title = _clean_text(
                entry.findtext(
                    "atom:title",
                    default="",
                    namespaces=_ATOM_NAMESPACE,
                )
            )
            summary = _clean_text(
                entry.findtext("atom:summary", default="", namespaces=_ATOM_NAMESPACE)
            )
            url = self._entry_url(entry)
            if not title or not url:
                continue

            fetched_at = datetime.now(UTC)
            published = entry.findtext(
                "atom:published",
                default="",
                namespaces=_ATOM_NAMESPACE,
            )
            with suppress(ValueError):
                fetched_at = datetime.fromisoformat(
                    published.replace("Z", "+00:00")
                ).astimezone(UTC)

            items.append(
                NewsItem(
                    source=self.source_name,
                    title=title,
                    url=url,
                    raw_content=summary or None,
                    fetched_at=fetched_at,
                )
            )

        return items

    def _entry_url(self, entry: ET.Element) -> str | None:
        for link in entry.findall("atom:link", _ATOM_NAMESPACE):
            if link.get("rel") == "alternate" and link.get("href"):
                return _clean_text(link.get("href"))

        entry_id = entry.findtext("atom:id", default="", namespaces=_ATOM_NAMESPACE)
        normalized = _clean_text(entry_id)
        return normalized or None
