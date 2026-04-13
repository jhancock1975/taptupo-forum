"""Property tests: no parser crashes on random garbage.

Each source's module-level ``_parse`` must tolerate any JSON-shaped or
XML-shaped input without raising. These properties bound the blast radius
of a malicious or buggy upstream API: the worst case should be an empty
list, never an unhandled exception propagating into the aggregator loop.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.news.arxiv import _parse as arxiv_parse
from app.news.guardian import _parse as guardian_parse
from app.news.newsapi import _parse as newsapi_parse
from app.news.reddit import _parse as reddit_parse
from app.news.rss import _parse as rss_parse

_json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=20),
)
_json_values = st.recursive(
    _json_scalars,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.text(max_size=10), children, max_size=4),
    ),
    max_leaves=10,
)
_json_dicts = st.dictionaries(st.text(max_size=10), _json_values, max_size=6)


@pytest.mark.hypothesis
@given(payload=_json_dicts)
@settings(max_examples=75, deadline=None)
def test_guardian_parse_is_total(payload: dict[str, object]) -> None:
    result = guardian_parse(payload)
    assert isinstance(result, list)


@pytest.mark.hypothesis
@given(payload=_json_dicts)
@settings(max_examples=75, deadline=None)
def test_reddit_parse_is_total(payload: dict[str, object]) -> None:
    result = reddit_parse(payload)
    assert isinstance(result, list)


@pytest.mark.hypothesis
@given(payload=_json_dicts)
@settings(max_examples=75, deadline=None)
def test_newsapi_parse_is_total(payload: dict[str, object]) -> None:
    result = newsapi_parse(payload)
    assert isinstance(result, list)


@pytest.mark.hypothesis
@given(body=st.text(max_size=200))
@settings(max_examples=75, deadline=None)
def test_arxiv_parse_is_total(body: str) -> None:
    result = arxiv_parse(body)
    assert isinstance(result, list)


@pytest.mark.hypothesis
@given(body=st.text(max_size=200))
@settings(max_examples=75, deadline=None)
def test_rss_parse_is_total(body: str) -> None:
    result = rss_parse(body)
    assert isinstance(result, list)
