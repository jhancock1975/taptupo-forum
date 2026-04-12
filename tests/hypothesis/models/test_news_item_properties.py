"""Hypothesis property tests for the NewsItem model."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from app.models.news_item import NewsItem

pytestmark = pytest.mark.hypothesis


sources = st.sampled_from(
    ["guardian", "arxiv", "hackernews", "reddit", "newsapi", "rss"]
)
statuses = st.sampled_from(["new", "promoted", "skipped"])


@st.composite
def news_items(draw: st.DrawFn) -> NewsItem:
    return NewsItem(
        source=draw(sources),
        title=draw(st.text(min_size=1, max_size=500)),
        url=draw(st.text(min_size=1, max_size=200)),
        raw_content=draw(st.text(max_size=1000)),
        status=draw(statuses),
        promoted_thread_id=draw(
            st.one_of(st.none(), st.text(min_size=1, max_size=50))
        ),
    )


@given(news_items())
def test_news_item_round_trip(item: NewsItem) -> None:
    assert NewsItem(**item.model_dump()) == item
