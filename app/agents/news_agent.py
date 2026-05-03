from __future__ import annotations

import structlog

from app.agents.base_agent import BaseAgent
from app.db.interface import RepositoryInterface
from app.models.schemas import Post, Thread, User

logger = structlog.get_logger()


class NewsAgent:
    def __init__(self, user: User, repo: RepositoryInterface) -> None:
        self._agent = BaseAgent(user, repo)
        self._repo = repo
        self._user = user

    async def promote_news(self, max_items: int = 5) -> list[Thread]:
        new_items = await self._repo.get_news_items_by_status("new")
        created_threads: list[Thread] = []

        for item in new_items[:max_items]:
            thread = Thread(
                title=item.title,
                source_url=item.url,
                source_type=item.source,
                summary=item.raw_content[:500] if item.raw_content else None,
                categories=[item.source],
                created_by=self._user.user_id,
            )
            await self._repo.create_thread(thread)

            opening = Post(
                thread_id=thread.thread_id,
                author_id=self._user.user_id,
                content=(
                    f"**{item.title}**\n\n"
                    f"Source: [{item.source}]({item.url})\n\n"
                    f"{item.raw_content[:300] + '...' if item.raw_content and len(item.raw_content) > 300 else item.raw_content or 'No summary available.'}\n\n"
                    f"What are your thoughts on this?"
                ),
            )
            await self._repo.create_post(opening)

            await self._repo.update_news_item_status(
                item.item_id, "promoted", thread.thread_id
            )
            created_threads.append(thread)
            logger.info(
                "news_promoted",
                item_id=item.item_id,
                thread_id=thread.thread_id,
                title=item.title,
            )

        # Mark remaining as skipped
        for item in new_items[max_items:]:
            await self._repo.update_news_item_status(item.item_id, "skipped")

        return created_threads
