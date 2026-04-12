"""NewsItem Pydantic model."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

NewsSource = Literal[
    "guardian", "arxiv", "hackernews", "reddit", "newsapi", "rss"
]
NewsStatus = Literal["new", "promoted", "skipped"]


class NewsItem(BaseModel):
    """An item fetched from an external news or research source."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    item_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: NewsSource
    title: str = Field(min_length=1, max_length=500)
    url: str
    raw_content: str
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: NewsStatus = "new"
    promoted_thread_id: str | None = None
