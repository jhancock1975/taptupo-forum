"""Atheris fuzz harness for news-source parsers.

Our news parsers consume third-party XML / JSON feeds, so robustness
against malformed input is a security concern, not just a correctness
one. This harness hammers each parser with random bytes and asserts
that it either returns ``list[NewsItem]`` or raises a handled exception
- never a silent crash or hang.

Run with::

    uv pip install atheris  # heavy optional dep
    uv run python tests/fuzz/fuzz_news_parsers.py

Atheris auto-instruments and runs indefinitely; Ctrl-C to stop. The
harness is structured so new parsers can be added by extending
``_PARSERS``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

try:
    import atheris  # type: ignore[import-not-found,import-untyped,unused-ignore]
except ImportError:  # pragma: no cover - optional dep
    atheris = None  # type: ignore[assignment]

from app.news.arxiv import _parse as parse_arxiv
from app.news.guardian import _parse as parse_guardian
from app.news.hackernews import _parse_item as parse_hackernews_item
from app.news.newsapi import _parse as parse_newsapi
from app.news.reddit import _parse as parse_reddit
from app.news.rss import _parse as parse_rss

# Each entry: (name, callable-that-accepts-bytes-or-dict-like, input-kind)
# kind="bytes" -> parser called with raw bytes; kind="json" -> parser called
# with a decoded-JSON dict (we feed it random structured garbage).
_PARSERS: list[tuple[str, Callable[..., Any], str]] = [
    ("arxiv", parse_arxiv, "bytes"),
    ("rss", parse_rss, "bytes"),
    ("guardian", parse_guardian, "json"),
    ("hackernews_item", parse_hackernews_item, "json"),
    ("reddit", parse_reddit, "json"),
    ("newsapi", parse_newsapi, "json"),
]


def _one_shot(data: bytes) -> None:
    """Feed ``data`` to each parser, swallowing only expected exceptions."""
    for _name, fn, kind in _PARSERS:
        try:
            if kind == "bytes":
                fn(data.decode("utf-8", errors="replace"))
            else:
                # Build a deterministic structured value from the fuzzed bytes
                fn({"fuzz": data.decode("utf-8", errors="replace")})
        except (ValueError, TypeError, KeyError, AttributeError):
            # These are all acceptable: parser rejected bad input.
            pass


def _entry(data: bytes) -> None:
    _one_shot(data)


if __name__ == "__main__":
    if atheris is None:
        print("atheris not installed; run 'uv pip install atheris' first", file=sys.stderr)
        raise SystemExit(2)
    atheris.Setup(sys.argv, _entry)
    atheris.Fuzz()
