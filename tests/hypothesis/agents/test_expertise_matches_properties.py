"""Property tests for expertise_matches."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.agents.base_agent import expertise_matches


@pytest.mark.hypothesis
@given(text=st.text(min_size=0, max_size=200))
@settings(max_examples=100, deadline=None)
def test_empty_areas_never_match(text: str) -> None:
    assert expertise_matches(text, []) is False


_ASCII_LETTERS = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"),
        whitelist_characters=" ",
        max_codepoint=127,
    ),
    min_size=1,
    max_size=20,
).filter(lambda s: s.strip() != "")


@pytest.mark.hypothesis
@given(
    prefix=st.text(max_size=50),
    area=_ASCII_LETTERS,
    suffix=st.text(max_size=50),
)
@settings(max_examples=100, deadline=None)
def test_case_insensitive_round_trip(prefix: str, area: str, suffix: str) -> None:
    # Skip unicode pairs where upper/lower are not round-trip-stable
    # (e.g. ß -> SS -> ss). Bound to ASCII-ish letters via the strategy.
    text = f"{prefix} {area.upper()} {suffix}"
    assert expertise_matches(text, [area.lower()]) is True
    assert expertise_matches(text.lower(), [area.upper()]) is True


@pytest.mark.hypothesis
@given(text=st.text(max_size=200))
@settings(max_examples=50, deadline=None)
def test_whitespace_around_area_tolerated(text: str) -> None:
    # If "foo" matches, so must "  foo  " as an expertise area.
    if expertise_matches(text, ["foo"]):
        assert expertise_matches(text, ["  foo  "])
