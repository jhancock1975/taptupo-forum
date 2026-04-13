"""Periodic aggregator that runs every registered ``NewsFetcher``.

The aggregator is the single writer into the ``news_items`` table.
Dedup happens in two stages:

1. Within a batch: the first time we see a URL wins (later duplicates
   from other sources are dropped).
2. Across history: we ask the repository whether the URL already exists
   before writing.

A failing fetcher is logged and skipped — it MUST NOT take down the
loop, because a single broken upstream would otherwise starve every
other source.
"""

from __future__ import annotations

import asyncio

import structlog

from app.db.interface import RepositoryInterface
from app.models import NewsItem
from app.news.interface import NewsFetcher

_log = structlog.get_logger(__name__)


class NewsAggregator:
    """Coordinates polling, deduplication, and persistence for news items."""

    def __init__(
        self,
        *,
        repository: RepositoryInterface,
        fetchers: list[NewsFetcher],
        interval_seconds: float = 600.0,
    ) -> None:
        self._repo = repository
        self._fetchers = fetchers
        self._interval = interval_seconds

    async def run_once(self) -> int:
        """Fetch from every source, dedupe, persist. Returns items created."""
        collected: list[NewsItem] = []
        for fetcher in self._fetchers:
            try:
                items = await fetcher.fetch()
            except Exception as exc:  # one rogue source must not kill the loop
                _log.warning(
                    "aggregator.fetcher_failed",
                    source=getattr(fetcher, "source_name", "?"),
                    error=str(exc),
                )
                continue
            collected.extend(items)

        seen_in_batch: set[str] = set()
        created = 0
        for item in collected:
            if item.url in seen_in_batch:
                continue
            seen_in_batch.add(item.url)
            if await self._repo.news_item_exists_by_url(item.url):
                continue
            try:
                await self._repo.create_news_item(item)
            except Exception as exc:  # persistence errors are logged, not fatal
                _log.warning("aggregator.persist_failed", url=item.url, error=str(exc))
                continue
            created += 1
        _log.info("aggregator.run_once", collected=len(collected), created=created)
        return created

    async def run_forever(self) -> None:
        """Loop: run_once, sleep interval, repeat. Cancellable."""
        while True:
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _log.error("aggregator.run_once_crashed", error=str(exc))
            await asyncio.sleep(self._interval)
