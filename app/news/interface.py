from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.models.schemas import NewsItem


@runtime_checkable
class NewsFetcher(Protocol):
    source_name: str

    async def fetch(self) -> list[NewsItem]: ...
