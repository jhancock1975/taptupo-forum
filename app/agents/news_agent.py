"""News agent: promotes interesting news items to forum threads.

Pulls pending items from the repository, asks the LLM whether each is
worth posting, and for yes-votes creates a Thread + updates the item's
status. No-votes are marked ``skipped`` so we don't re-evaluate them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

import structlog

from app.db.interface import RepositoryInterface
from app.models import NewsItem, Thread

_log = structlog.get_logger(__name__)

_RELEVANCE_PROMPT = (
    "You curate a discussion forum. Decide whether this news item is "
    "worth posting.\n\nTitle: {title}\nSummary: {summary}\n\n"
    "Answer with a single word: YES or NO."
)


class _LLMResponse(Protocol):
    content: str


class _LLM(Protocol):
    async def ainvoke(self, prompt: str, /) -> _LLMResponse: ...


class NewsAgent:
    """Promotes fresh NewsItems to Threads when the LLM judges them interesting."""

    def __init__(
        self,
        *,
        repository: RepositoryInterface,
        llm: _LLM,
        creator_user_id: str,
        batch_limit: int = 100,
    ) -> None:
        self._repo = repository
        self._llm = llm
        self._creator = creator_user_id
        self._limit = batch_limit

    async def run_once(self) -> int:
        """Walk the pending-items queue, promoting interesting ones. Returns count promoted."""
        items = await self._repo.list_new_news_items(limit=self._limit)
        promoted = 0
        for item in items:
            try:
                verdict = await self._is_interesting(item)
            except Exception as exc:
                _log.warning("news_agent.llm_failed", item_id=item.item_id, error=str(exc))
                continue
            if verdict:
                thread = Thread(
                    title=item.title[:300],
                    source_url=item.url,
                    source_type=item.source,
                    summary=item.raw_content[:2000],
                    created_by=self._creator,
                    last_activity_at=datetime.now(UTC),
                )
                await self._repo.create_thread(thread)
                await self._repo.update_news_item_status(
                    item.item_id, "promoted", promoted_thread_id=thread.thread_id
                )
                promoted += 1
            else:
                await self._repo.update_news_item_status(item.item_id, "skipped")
        _log.info("news_agent.run_once", considered=len(items), promoted=promoted)
        return promoted

    async def _is_interesting(self, item: NewsItem) -> bool:
        prompt = _RELEVANCE_PROMPT.format(title=item.title, summary=item.raw_content[:500])
        resp = await self._llm.ainvoke(prompt)
        content = (resp.content or "").strip().upper()
        return content.startswith("YES")
