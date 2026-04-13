"""Generic RSS/Atom fetcher for arbitrary feed URLs.

Handles the RSS 2.0 ``<item>`` shape and the Atom ``<entry>`` shape with
a single parser. Uses ``defusedxml`` for safe parsing.
"""

from __future__ import annotations

import httpx
import structlog
from defusedxml import ElementTree as DefusedET  # type: ignore[import-untyped,unused-ignore]

from app.models import NewsItem

_log = structlog.get_logger(__name__)
_ATOM_NS = "{http://www.w3.org/2005/Atom}"


class RSSFetcher:
    """Fetch and parse one or more RSS/Atom feeds."""

    source_name = "rss"

    def __init__(
        self,
        *,
        feed_urls: list[str],
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._feed_urls = feed_urls
        self._transport = transport
        self._timeout = timeout

    async def fetch(self) -> list[NewsItem]:
        items: list[NewsItem] = []
        async with httpx.AsyncClient(transport=self._transport, timeout=self._timeout) as client:
            for url in self._feed_urls:
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    body = resp.text
                except httpx.HTTPError as exc:
                    _log.warning("rss.fetch_failed", feed=url, error=str(exc))
                    continue
                items.extend(_parse(body))
        return items


def _parse(body: str) -> list[NewsItem]:
    try:
        root = DefusedET.fromstring(body)
    except DefusedET.ParseError:
        return []
    items: list[NewsItem] = []
    # RSS 2.0: <rss><channel><item>
    for rss_item in root.iter("item"):
        title = (rss_item.findtext("title") or "").strip()
        link = (rss_item.findtext("link") or "").strip()
        desc = (rss_item.findtext("description") or "").strip()
        if title and link:
            items.append(
                NewsItem(
                    source="rss",
                    title=title[:500],
                    url=link,
                    raw_content=desc,
                )
            )
    # Atom: <feed><entry>
    for entry in root.iter(f"{_ATOM_NS}entry"):
        title_el = entry.find(f"{_ATOM_NS}title")
        title = (title_el.text or "").strip() if title_el is not None else ""
        link = ""
        for link_el in entry.findall(f"{_ATOM_NS}link"):
            href = link_el.get("href")
            if href:
                link = href
                break
        summary_el = entry.find(f"{_ATOM_NS}summary")
        summary = (summary_el.text or "").strip() if summary_el is not None else ""
        if title and link:
            items.append(
                NewsItem(
                    source="rss",
                    title=title[:500],
                    url=link,
                    raw_content=summary,
                )
            )
    return items
