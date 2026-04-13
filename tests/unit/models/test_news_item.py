"""Unit tests for the NewsItem model."""

from __future__ import annotations

import re
import uuid
from datetime import datetime

import pytest
from pydantic import ValidationError

from app.models.news_item import NewsItem

pytestmark = pytest.mark.unit

UUID4_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def test_news_item_constructs_with_defaults() -> None:
    item = NewsItem(
        source="guardian",
        title="Breaking",
        url="https://example.com",
        raw_content="body",
    )
    assert UUID4_RE.match(item.item_id)
    uuid.UUID(item.item_id, version=4)
    assert item.source == "guardian"
    assert item.status == "new"
    assert item.promoted_thread_id is None
    assert isinstance(item.fetched_at, datetime)
    assert item.fetched_at.tzinfo is not None


def test_news_item_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        NewsItem(
            source="guardian",
            title="T",
            url="u",
            raw_content="r",
            bogus=1,  # type: ignore[call-arg]
        )


def test_news_item_rejects_invalid_source() -> None:
    with pytest.raises(ValidationError):
        NewsItem(
            source="human",  # type: ignore[arg-type]
            title="T",
            url="u",
            raw_content="r",
        )


def test_news_item_rejects_empty_title() -> None:
    with pytest.raises(ValidationError):
        NewsItem(source="guardian", title="", url="u", raw_content="r")


def test_news_item_rejects_overlong_title() -> None:
    with pytest.raises(ValidationError):
        NewsItem(
            source="guardian",
            title="x" * 501,
            url="u",
            raw_content="r",
        )


def test_news_item_rejects_invalid_status() -> None:
    with pytest.raises(ValidationError):
        NewsItem(
            source="guardian",
            title="T",
            url="u",
            raw_content="r",
            status="archived",  # type: ignore[arg-type]
        )
