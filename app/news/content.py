from __future__ import annotations

import re
from html.parser import HTMLParser

_BLOCK_TAGS = {
    "article",
    "aside",
    "blockquote",
    "br",
    "div",
    "figcaption",
    "figure",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "li",
    "main",
    "ol",
    "p",
    "section",
    "ul",
}
_SKIP_TAGS = {"script", "style", "noscript", "svg"}
_META_DESCRIPTION_RE = re.compile(
    r"""<meta[^>]+(?:name|property)=["'](?:description|og:description|twitter:description)["'][^>]+content=["'](.*?)["'][^>]*>""",
    re.IGNORECASE | re.DOTALL,
)


class _ArticleTextParser(HTMLParser):
    def __init__(self, *, prefer_main_content: bool) -> None:
        super().__init__(convert_charrefs=True)
        self._prefer_main_content = prefer_main_content
        self._skip_depth = 0
        self._preferred_depth = 0
        self._parts: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del attrs
        normalized = tag.lower()
        if normalized in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if normalized in {"article", "main"}:
            self._preferred_depth += 1
        if self._should_capture() and normalized in _BLOCK_TAGS:
            self._push_break()

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized in _SKIP_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if self._should_capture() and normalized in _BLOCK_TAGS:
            self._push_break()
        if normalized in {"article", "main"} and self._preferred_depth > 0:
            self._preferred_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._should_capture():
            return
        text = " ".join(data.split())
        if not text:
            return
        if self._parts and not self._parts[-1].endswith((" ", "\n")):
            self._parts.append(" ")
        self._parts.append(text)

    def text(self) -> str:
        joined = "".join(self._parts)
        lines = [" ".join(line.split()) for line in joined.splitlines()]
        cleaned = "\n".join(line for line in lines if line)
        return cleaned.strip()

    def _should_capture(self) -> bool:
        if self._skip_depth > 0:
            return False
        if not self._prefer_main_content:
            return True
        return self._preferred_depth > 0

    def _push_break(self) -> None:
        if self._parts and not self._parts[-1].endswith("\n"):
            self._parts.append("\n")


def _parse_text(html: str, *, prefer_main_content: bool) -> str:
    parser = _ArticleTextParser(prefer_main_content=prefer_main_content)
    parser.feed(html)
    parser.close()
    return parser.text()


def extract_article_text(html: str) -> str | None:
    preferred = _parse_text(html, prefer_main_content=True)
    if len(preferred) >= 140:
        return preferred

    meta_match = _META_DESCRIPTION_RE.search(html)
    if meta_match:
        description = " ".join(meta_match.group(1).split()).strip()
        if description:
            return description

    fallback = _parse_text(html, prefer_main_content=False)
    return fallback or None
