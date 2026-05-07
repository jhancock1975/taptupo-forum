from __future__ import annotations

import re

from markupsafe import Markup, escape

_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
_URL_RE = re.compile(r"https?://[^\s<]+")
_TRAILING_PUNCTUATION = ".,!?;:"


def _external_link(url: str, label: str) -> Markup:
    escaped_url = escape(url)
    escaped_label = escape(label)
    return Markup(
        f'<a href="{escaped_url}" target="_blank" '
        f'rel="noopener noreferrer">{escaped_label}</a>'
    )


def _trim_url(url: str) -> tuple[str, str]:
    trimmed = url
    trailing = ""

    while trimmed and trimmed[-1] in _TRAILING_PUNCTUATION:
        trailing = trimmed[-1] + trailing
        trimmed = trimmed[:-1]

    while trimmed.endswith(")") and trimmed.count("(") < trimmed.count(")"):
        trailing = ")" + trailing
        trimmed = trimmed[:-1]

    return trimmed, trailing


def _render_plain_text(content: str) -> list[Markup]:
    parts: list[Markup] = []
    last_index = 0

    for match in _URL_RE.finditer(content):
        start, end = match.span()
        url, trailing = _trim_url(match.group(0))
        if not url:
            continue

        parts.append(escape(content[last_index:start]))
        parts.append(_external_link(url, url))
        parts.append(escape(trailing))
        last_index = end

    parts.append(escape(content[last_index:]))
    return parts


def render_post_content(content: str) -> Markup:
    parts: list[Markup] = []
    last_index = 0

    for match in _MARKDOWN_LINK_RE.finditer(content):
        start, end = match.span()
        label, url = match.groups()
        parts.extend(_render_plain_text(content[last_index:start]))
        parts.append(_external_link(url, label))
        last_index = end

    parts.extend(_render_plain_text(content[last_index:]))
    return Markup("").join(parts)
