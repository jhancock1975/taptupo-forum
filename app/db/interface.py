"""Abstract repository interface for taptupo-forum data access.

All methods are ``async`` so any implementation can perform non-blocking
I/O. The interface is expressed as an :class:`abc.ABC` (not a
:class:`typing.Protocol`) so that forgetting to implement a method is a
run-time error, not a silent one.
"""

from __future__ import annotations

import abc
from datetime import datetime
from typing import Literal

from app.models import NewsItem, Post, Thread, User


class RepositoryError(Exception):
    """Base class for all repository-layer errors."""


class UserExistsError(RepositoryError):
    """Raised when attempting to create a user whose username is taken."""


class RepositoryInterface(abc.ABC):
    """Abstract data-access contract.

    Concrete implementations (e.g. :class:`app.db.dynamo.DynamoRepository`)
    satisfy this interface. Callers MUST depend on this interface rather
    than any concrete class so the backend can be swapped at runtime.
    """

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------
    @abc.abstractmethod
    async def create_user(self, user: User) -> None:
        """Persist a new ``User``.

        Raises :class:`UserExistsError` if the username is already taken.
        """

    @abc.abstractmethod
    async def get_user(self, user_id: str) -> User | None:
        """Return the ``User`` with this ``user_id``, or ``None`` if missing."""

    @abc.abstractmethod
    async def get_user_by_username(self, username: str) -> User | None:
        """Return the ``User`` with this ``username``, or ``None`` if missing."""

    @abc.abstractmethod
    async def list_agents(self) -> list[User]:
        """Return all users for which ``is_agent`` is true."""

    # ------------------------------------------------------------------
    # Threads
    # ------------------------------------------------------------------
    @abc.abstractmethod
    async def create_thread(self, thread: Thread) -> None:
        """Persist a new ``Thread``, seeding ``last_activity_at``."""

    @abc.abstractmethod
    async def get_thread(self, thread_id: str) -> Thread | None:
        """Return the ``Thread`` with this ``thread_id``, or ``None`` if missing."""

    @abc.abstractmethod
    async def list_threads(self, limit: int = 50) -> list[Thread]:
        """Return up to ``limit`` threads, newest ``last_activity_at`` first.

        Implementations may perform a full table scan and sort in Python;
        this is acceptable at v1 scale.
        """

    @abc.abstractmethod
    async def update_thread_activity(self, thread_id: str, when: datetime) -> None:
        """Update a thread's ``last_activity_at`` timestamp to ``when``."""

    # ------------------------------------------------------------------
    # Posts
    # ------------------------------------------------------------------
    @abc.abstractmethod
    async def create_post(self, post: Post) -> None:
        """Persist a new ``Post`` and bump its thread's ``last_activity_at``."""

    @abc.abstractmethod
    async def get_post(self, post_id: str) -> Post | None:
        """Return the ``Post`` with this ``post_id``, or ``None`` if missing."""

    @abc.abstractmethod
    async def get_posts_by_thread(self, thread_id: str) -> list[Post]:
        """Return all posts in a thread, oldest ``created_at`` first."""

    # ------------------------------------------------------------------
    # NewsItems
    # ------------------------------------------------------------------
    @abc.abstractmethod
    async def create_news_item(self, item: NewsItem) -> None:
        """Persist a new ``NewsItem``."""

    @abc.abstractmethod
    async def get_news_item(self, item_id: str) -> NewsItem | None:
        """Return the ``NewsItem`` with this ``item_id``, or ``None`` if missing."""

    @abc.abstractmethod
    async def list_new_news_items(self, limit: int = 100) -> list[NewsItem]:
        """Return up to ``limit`` news items whose ``status`` is ``"new"``."""

    @abc.abstractmethod
    async def update_news_item_status(
        self,
        item_id: str,
        status: Literal["new", "promoted", "skipped"],
        promoted_thread_id: str | None = None,
    ) -> None:
        """Set a news item's ``status`` and optional ``promoted_thread_id``."""

    @abc.abstractmethod
    async def news_item_exists_by_url(self, url: str) -> bool:
        """Return ``True`` if a news item with this URL already exists.

        Used for dedup before inserting a newly-fetched item. Implementations
        backed by DynamoDB will scan ``news_items`` with a ``FilterExpression``
        on ``url``; for v1 this is acceptable. A later optimisation is to add
        a GSI on ``url`` to convert the scan into a point query.
        """
