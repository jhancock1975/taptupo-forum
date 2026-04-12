"""Hypothesis property tests for the Thread model."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from app.models.thread import Thread

pytestmark = pytest.mark.hypothesis


source_types = st.sampled_from(
    ["guardian", "arxiv", "hackernews", "reddit", "newsapi", "rss", "human"]
)


@st.composite
def threads(draw: st.DrawFn) -> Thread:
    return Thread(
        title=draw(st.text(min_size=1, max_size=300)),
        source_url=draw(st.one_of(st.none(), st.text(max_size=100))),
        source_type=draw(source_types),
        summary=draw(st.text(max_size=200)),
        categories=draw(st.lists(st.text(min_size=1, max_size=20), max_size=5)),
        created_by=draw(st.text(min_size=1, max_size=50)),
    )


@given(threads())
def test_thread_round_trip(thread: Thread) -> None:
    assert Thread(**thread.model_dump()) == thread


@given(
    title=st.text(min_size=1, max_size=300),
    source_type=source_types,
    created_by=st.text(min_size=1, max_size=50),
)
def test_thread_default_timestamps_are_monotonic(
    title: str, source_type: str, created_by: str
) -> None:
    # When no timestamps are supplied, last_activity_at is sampled after
    # created_at, so it is never earlier.
    thread = Thread(title=title, source_type=source_type, created_by=created_by)  # type: ignore[arg-type]
    assert thread.created_at <= thread.last_activity_at
