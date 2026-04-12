"""Unit tests for the abstract repository interface."""

from __future__ import annotations

import pytest

from app.db.interface import RepositoryError, RepositoryInterface, UserExistsError

pytestmark = pytest.mark.unit


EXPECTED_METHODS = {
    "create_user",
    "get_user",
    "get_user_by_username",
    "list_agents",
    "create_thread",
    "get_thread",
    "list_threads",
    "update_thread_activity",
    "create_post",
    "get_post",
    "get_posts_by_thread",
    "create_news_item",
    "get_news_item",
    "list_new_news_items",
    "update_news_item_status",
    "news_item_exists_by_url",
}


def test_interface_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        RepositoryInterface()  # type: ignore[abstract]


def test_interface_declares_all_expected_abstract_methods() -> None:
    assert RepositoryInterface.__abstractmethods__ >= EXPECTED_METHODS


def test_user_exists_error_is_repository_error() -> None:
    assert issubclass(UserExistsError, RepositoryError)
