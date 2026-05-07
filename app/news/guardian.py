from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from app.config import settings
from app.models.schemas import NewsItem

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

logger = structlog.get_logger()

_REQUEST_TIMEOUT = httpx.Timeout(12.0, connect=5.0)
_DEFAULT_REQUEST_INTERVAL_SECONDS = 86400 / 500
_DEFAULT_SHOW_FIELDS = ("trailText", "bodyText")


def _as_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _coerce_results(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("invalid_guardian_response")
    response_block = payload.get("response", {})
    if not isinstance(response_block, dict):
        raise ValueError("invalid_guardian_response")
    raw_results = response_block.get("results", [])
    if not isinstance(raw_results, list):
        raise ValueError("invalid_guardian_response")
    return [result for result in raw_results if isinstance(result, dict)]


def _show_fields(fields: Iterable[str]) -> str:
    return ",".join(dict.fromkeys(field for field in fields if field))


class GuardianRateLimiter:
    def __init__(
        self,
        min_interval_seconds: float,
        *,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._min_interval_seconds = min_interval_seconds
        self._clock = clock or time.monotonic
        self._sleep = sleep or asyncio.sleep
        self._lock = asyncio.Lock()
        self._next_allowed_at = 0.0

    async def wait_turn(self) -> None:
        async with self._lock:
            now = self._clock()
            if now < self._next_allowed_at:
                await self._sleep(self._next_allowed_at - now)
                now = self._clock()
            self._next_allowed_at = now + self._min_interval_seconds


_shared_guardian_rate_limiter = GuardianRateLimiter(
    min_interval_seconds=max(
        settings.guardian_request_interval_seconds,
        _DEFAULT_REQUEST_INTERVAL_SECONDS,
    )
)


class GuardianClient:
    _base_url = "https://content.guardianapis.com/search"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        rate_limiter: GuardianRateLimiter | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else settings.guardian_api_key
        self._rate_limiter = rate_limiter or _shared_guardian_rate_limiter

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    async def search(
        self,
        *,
        query: str | None = None,
        limit: int = 5,
        section: str | None = None,
        show_fields: Iterable[str] = _DEFAULT_SHOW_FIELDS,
        order_by: str = "newest",
    ) -> list[dict[str, Any]]:
        if not self._api_key:
            raise ValueError("missing_api_key")

        params: dict[str, Any] = {
            "api-key": self._api_key,
            "page-size": limit,
            "show-fields": _show_fields(show_fields),
            "order-by": order_by,
        }
        if query:
            params["q"] = query
        if section:
            params["section"] = section

        await self._rate_limiter.wait_turn()
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            response = await client.get(self._base_url, params=params)
            response.raise_for_status()
            return _coerce_results(response.json())


class GuardianFetcher:
    source_name = "guardian"

    def __init__(self, client: GuardianClient | None = None) -> None:
        self._client = client or GuardianClient()

    async def fetch(self) -> list[NewsItem]:
        if not self._client.configured:
            return []

        try:
            results = await self._client.search(limit=10)
        except ValueError as error:
            logger.warning("guardian_fetch_skipped", reason=str(error))
            return []
        except httpx.HTTPError:
            logger.exception("guardian_fetch_failed")
            return []

        items: list[NewsItem] = []
        for result in results:
            title = _as_text(result.get("webTitle"))
            url = _as_text(result.get("webUrl"))
            if not title or not url:
                continue

            fields = result.get("fields", {})
            trail_text = ""
            body_text = ""
            if isinstance(fields, dict):
                trail_text = _as_text(fields.get("trailText")) or ""
                body_text = _as_text(fields.get("bodyText")) or ""

            raw_content = body_text or trail_text or None
            published = _as_text(result.get("webPublicationDate"))
            fetched_at = datetime.now(UTC)
            if published:
                with suppress(ValueError):
                    fetched_at = datetime.fromisoformat(
                        published.replace("Z", "+00:00")
                    ).astimezone(UTC)

            items.append(
                NewsItem(
                    source=self.source_name,
                    title=title,
                    url=url,
                    raw_content=raw_content,
                    fetched_at=fetched_at,
                )
            )

        return items
