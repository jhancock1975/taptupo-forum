"""Thread Pydantic model."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SourceType = Literal["guardian", "arxiv", "hackernews", "reddit", "newsapi", "rss", "human"]


class Thread(BaseModel):
    """A forum discussion thread."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(min_length=1, max_length=300)
    source_url: str | None = None
    source_type: SourceType
    summary: str = ""
    categories: list[str] = Field(default_factory=list)
    created_by: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_activity_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
