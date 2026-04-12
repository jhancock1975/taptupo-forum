"""Post Pydantic model."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


class Post(BaseModel):
    """A post within a thread."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    post_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    thread_id: str
    parent_post_id: str | None = None
    author_id: str
    content: str = Field(min_length=1, max_length=10000)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
