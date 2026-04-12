"""Hypothesis property tests for the Post model."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from app.models.post import Post

pytestmark = pytest.mark.hypothesis


@st.composite
def posts(draw: st.DrawFn) -> Post:
    return Post(
        thread_id=draw(st.text(min_size=1, max_size=50)),
        parent_post_id=draw(st.one_of(st.none(), st.text(min_size=1, max_size=50))),
        author_id=draw(st.text(min_size=1, max_size=50)),
        content=draw(st.text(min_size=1, max_size=10000)),
    )


@given(posts())
def test_post_round_trip(post: Post) -> None:
    assert Post(**post.model_dump()) == post
