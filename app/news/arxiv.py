"""arXiv fetcher using the public Atom query API.

Docs: https://info.arxiv.org/help/api/user-manual.html

We parse the Atom XML with ``defusedxml`` to avoid XXE and similar pitfalls.
"""

from __future__ import annotations

import httpx
import structlog
from defusedxml import ElementTree as DefusedET  # type: ignore[import-untyped,unused-ignore]

from app.models import NewsItem

_log = structlog.get_logger(__name__)
_ENDPOINT = "https://export.arxiv.org/api/query"
_ATOM_NS = "{http://www.w3.org/2005/Atom}"


class ArxivFetcher:
    """Fetch recent arXiv submissions for a configurable category query."""

    source_name = "arxiv"

    def __init__(
        self,
        *,
        query: str = "cat:cs.AI OR cat:cs.LG",
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 15.0,
        max_results: int = 20,
    ) -> None:
        self._query = query
        self._transport = transport
        self._timeout = timeout
        self._max_results = max_results

    async def fetch(self) -> list[NewsItem]:
        params = {
            "search_query": self._query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": str(self._max_results),
        }
        try:
            async with httpx.AsyncClient(
                transport=self._transport, timeout=self._timeout
            ) as client:
                resp = await client.get(_ENDPOINT, params=params)
            resp.raise_for_status()
            body = resp.text
        except httpx.HTTPError as exc:
            _log.warning("arxiv.fetch_failed", error=str(exc))
            return []
        return _parse(body)


def _parse(body: str) -> list[NewsItem]:
    try:
        root = DefusedET.fromstring(body)
    except DefusedET.ParseError:
        return []
    items: list[NewsItem] = []
    for entry in root.findall(f"{_ATOM_NS}entry"):
        title_el = entry.find(f"{_ATOM_NS}title")
        summary_el = entry.find(f"{_ATOM_NS}summary")
        link = None
        for link_el in entry.findall(f"{_ATOM_NS}link"):
            if link_el.get("rel") in (None, "alternate") and link_el.get("type") == "text/html":
                link = link_el.get("href")
                break
        if link is None:
            id_el = entry.find(f"{_ATOM_NS}id")
            link = (id_el.text or "").strip() if id_el is not None else None
        title = (title_el.text or "").strip() if title_el is not None else ""
        summary = (summary_el.text or "").strip() if summary_el is not None else ""
        if not title or not link:
            continue
        items.append(
            NewsItem(
                source="arxiv",
                title=title[:500],
                url=link,
                raw_content=summary,
            )
        )
    return items
