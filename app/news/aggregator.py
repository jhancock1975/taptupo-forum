from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from app.config import settings
from app.news.arxiv import ArxivFetcher
from app.news.guardian import GuardianFetcher
from app.news.hackernews import HackerNewsFetcher

if TYPE_CHECKING:
    from app.db.interface import RepositoryInterface
    from app.news.interface import NewsFetcher

logger = structlog.get_logger()


class NewsAggregator:
    def __init__(self, repo: RepositoryInterface) -> None:
        self._repo = repo
        self._fetchers: list[NewsFetcher] = [
            HackerNewsFetcher(),
            ArxivFetcher(),
        ]
        if settings.guardian_api_key:
            self._fetchers.append(GuardianFetcher())

    def register_fetcher(self, fetcher: NewsFetcher) -> None:
        self._fetchers.append(fetcher)

    async def fetch_all(self) -> int:
        total_new = 0
        for fetcher in self._fetchers:
            logger.info("fetching_news", source=fetcher.source_name)
            try:
                items = await fetcher.fetch()
                for item in items:
                    existing = await self._repo.get_news_item_by_url(item.url)
                    if existing is None:
                        await self._repo.create_news_item(item)
                        total_new += 1
                logger.info(
                    "fetch_complete",
                    source=fetcher.source_name,
                    fetched=len(items),
                )
            except Exception:
                logger.exception("fetch_failed", source=fetcher.source_name)
        logger.info("aggregation_complete", new_items=total_new)
        return total_new
