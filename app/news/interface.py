"""Protocol that every news source implementation must satisfy."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.models import NewsItem


@runtime_checkable
class NewsFetcher(Protocol):
    """A pluggable source of ``NewsItem``s.

    Attributes:
        source_name: Short identifier matching a ``NewsSource`` literal
            (``"guardian"``, ``"arxiv"``, ...).
    """

    source_name: str

    async def fetch(self) -> list[NewsItem]:
        """Return freshly-fetched items. Network errors MUST NOT leak.

        Implementations should log and swallow transport-level failures,
        returning an empty list so the aggregator can keep running.
        """
        ...
