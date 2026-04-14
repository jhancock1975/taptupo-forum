"""Smoke end-to-end tests for the forum UI via Playwright.

Gated behind the ``e2e`` pytest marker so they are skipped by default.
Run with: ``uv run pytest -m e2e``. Requires ``playwright install chromium``.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from app.models import Thread
from tests.e2e.conftest import InMemoryRepo


@pytest.mark.e2e
def test_index_page_shows_empty_state(page: Page, live_app: tuple[str, InMemoryRepo]) -> None:
    base_url, _ = live_app
    page.goto(f"{base_url}/")
    expect(page.get_by_text("No threads yet")).to_be_visible()


@pytest.mark.e2e
def test_index_page_lists_seeded_thread(
    page: Page,
    live_app: tuple[str, InMemoryRepo],
) -> None:
    base_url, repo = live_app
    repo.threads["t-1"] = Thread(
        thread_id="t-1",
        title="E2E test topic",
        source_type="human",
        created_by="u-1",
    )
    page.goto(f"{base_url}/")
    expect(page.get_by_text("E2E test topic")).to_be_visible()


@pytest.mark.e2e
def test_register_then_post_reply_shows_up_on_thread(
    page: Page,
    live_app: tuple[str, InMemoryRepo],
) -> None:
    base_url, repo = live_app
    repo.threads["t-1"] = Thread(
        thread_id="t-1",
        title="reply target",
        source_type="human",
        created_by="u-1",
    )

    page.goto(f"{base_url}/register")
    page.fill("input[name=username]", "e2e_user")
    page.fill("input[name=password]", "correcthorsebattery")  # pragma: allowlist secret
    page.click("button[type=submit]")

    page.goto(f"{base_url}/threads/t-1")
    page.fill("textarea[name=content]", "hello from playwright")
    page.click("button[type=submit]")
    expect(page.get_by_text("hello from playwright")).to_be_visible()
