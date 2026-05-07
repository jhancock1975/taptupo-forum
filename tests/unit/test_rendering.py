from __future__ import annotations

from app.rendering import render_post_content


def test_render_post_content_linkifies_markdown_links_and_urls() -> None:
    rendered = render_post_content(
        "Source: [Example](https://example.com)\nRead https://example.org/test."
    )

    html = str(rendered)
    assert 'href="https://example.com"' in html
    assert ">Example</a>" in html
    assert 'href="https://example.org/test"' in html
    assert 'target="_blank"' in html
    assert html.endswith("</a>.")


def test_render_post_content_escapes_html_around_links() -> None:
    rendered = render_post_content('<script>alert("x")</script> https://example.com')

    html = str(rendered)
    assert "&lt;script&gt;" in html
    assert "<script>" not in html
    assert 'href="https://example.com"' in html


def test_render_post_content_trims_unmatched_closing_parenthesis() -> None:
    rendered = render_post_content("Read this (https://example.com/test(path)).")

    html = str(rendered)
    assert 'href="https://example.com/test(path)"' in html
    assert html.endswith("</a>).")
