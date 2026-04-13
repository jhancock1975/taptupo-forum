"""DynamoDB-backed implementation of :class:`RepositoryInterface`.

A single :class:`DynamoRepository` serves both ``dynamodb-local`` (dev)
and real AWS (prod). The only difference is the ``endpoint_url`` passed
in at construction time: ``None`` for AWS, a URL such as
``http://localhost:8000`` for local.

Implementation notes
--------------------

* We use ``aioboto3`` for truly async I/O. A single
  :class:`aioboto3.Session` is created in ``__init__``; a fresh
  ``resource("dynamodb", ...)`` is obtained per call using ``async with``.
  This avoids the lifecycle headaches of long-lived aioboto3 resources
  while keeping each call non-blocking.

* Pydantic models round-trip through ``model_dump(mode="json")`` which
  serialises datetimes to ISO strings. Floats are converted to
  :class:`decimal.Decimal` because DynamoDB's Python type serialiser
  rejects raw floats.
"""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal, cast

import aioboto3  # type: ignore[import-untyped,import-not-found,unused-ignore]
from botocore.exceptions import (  # type: ignore[import-untyped,import-not-found,unused-ignore]
    ClientError,
)

from app.db.interface import RepositoryInterface, UserExistsError
from app.models import NewsItem, Post, Thread, User

if TYPE_CHECKING:
    from collections.abc import Mapping

_USERS = "users"
_THREADS = "threads"
_POSTS = "posts"
_NEWS_ITEMS = "news_items"


class DynamoRepository(RepositoryInterface):
    """DynamoDB repository backed by ``aioboto3``.

    Parameters
    ----------
    endpoint_url:
        Override for the DynamoDB endpoint. ``None`` uses the real AWS
        endpoint resolved from credentials and region; a URL such as
        ``http://localhost:8000`` targets ``dynamodb-local``.
    region_name:
        AWS region name. Defaults to ``us-east-1``.
    """

    endpoint_url: str | None
    region_name: str
    _session: aioboto3.Session

    def __init__(self, endpoint_url: str | None, region_name: str = "us-east-1") -> None:
        self.endpoint_url = endpoint_url
        self.region_name = region_name
        self._session = aioboto3.Session()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _resource(self) -> Any:
        kwargs: dict[str, Any] = {"region_name": self.region_name}
        if self.endpoint_url is not None:
            kwargs["endpoint_url"] = self.endpoint_url
        return self._session.resource("dynamodb", **kwargs)

    @staticmethod
    def _to_item(model: User | Thread | Post | NewsItem) -> dict[str, Any]:
        """Serialise a Pydantic model to a DynamoDB-friendly dict.

        Datetimes become ISO strings (via ``mode="json"``). Floats are
        converted to :class:`decimal.Decimal` because DynamoDB's Python
        type serialiser rejects raw floats.
        """
        dumped = model.model_dump(mode="json")
        return cast(
            "dict[str, Any]",
            json.loads(json.dumps(dumped), parse_float=Decimal),
        )

    @staticmethod
    def _from_user(item: Mapping[str, Any]) -> User:
        return User.model_validate(dict(item))

    @staticmethod
    def _from_thread(item: Mapping[str, Any]) -> Thread:
        return Thread.model_validate(dict(item))

    @staticmethod
    def _from_post(item: Mapping[str, Any]) -> Post:
        return Post.model_validate(dict(item))

    @staticmethod
    def _from_news_item(item: Mapping[str, Any]) -> NewsItem:
        return NewsItem.model_validate(dict(item))

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------
    async def create_user(self, user: User) -> None:
        async with self._resource() as ddb:
            users = await ddb.Table(_USERS)
            # Primary guard: username uniqueness via GSI.
            resp = await users.query(
                IndexName="username-index",
                KeyConditionExpression="username = :u",
                ExpressionAttributeValues={":u": user.username},
                Limit=1,
            )
            if resp.get("Items"):
                raise UserExistsError(f"username already exists: {user.username}")
            try:
                await users.put_item(
                    Item=self._to_item(user),
                    ConditionExpression="attribute_not_exists(user_id)",
                )
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code")
                if code == "ConditionalCheckFailedException":
                    raise UserExistsError(f"user_id already exists: {user.user_id}") from exc
                raise

    async def get_user(self, user_id: str) -> User | None:
        async with self._resource() as ddb:
            users = await ddb.Table(_USERS)
            resp = await users.get_item(Key={"user_id": user_id})
        item = resp.get("Item")
        if item is None:
            return None
        return self._from_user(item)

    async def get_user_by_username(self, username: str) -> User | None:
        async with self._resource() as ddb:
            users = await ddb.Table(_USERS)
            resp = await users.query(
                IndexName="username-index",
                KeyConditionExpression="username = :u",
                ExpressionAttributeValues={":u": username},
                Limit=1,
            )
            items = resp.get("Items") or []
            if not items:
                return None
            # GSI projection is KEYS_ONLY, so fetch the full record by PK.
            user_id = items[0]["user_id"]
            full = await users.get_item(Key={"user_id": user_id})
        raw = full.get("Item")
        if raw is None:
            return None
        return self._from_user(raw)

    async def list_agents(self) -> list[User]:
        async with self._resource() as ddb:
            users = await ddb.Table(_USERS)
            resp = await users.scan(
                FilterExpression="is_agent = :t",
                ExpressionAttributeValues={":t": True},
            )
        items = resp.get("Items") or []
        return [self._from_user(i) for i in items]

    # ------------------------------------------------------------------
    # Threads
    # ------------------------------------------------------------------
    async def create_thread(self, thread: Thread) -> None:
        async with self._resource() as ddb:
            threads = await ddb.Table(_THREADS)
            await threads.put_item(Item=self._to_item(thread))

    async def get_thread(self, thread_id: str) -> Thread | None:
        async with self._resource() as ddb:
            threads = await ddb.Table(_THREADS)
            resp = await threads.get_item(Key={"thread_id": thread_id})
        item = resp.get("Item")
        return None if item is None else self._from_thread(item)

    async def list_threads(self, limit: int = 50) -> list[Thread]:
        async with self._resource() as ddb:
            threads = await ddb.Table(_THREADS)
            resp = await threads.scan()
        items = resp.get("Items") or []
        parsed = [self._from_thread(i) for i in items]
        parsed.sort(key=lambda t: t.last_activity_at, reverse=True)
        return parsed[:limit]

    async def update_thread_activity(self, thread_id: str, when: datetime) -> None:
        async with self._resource() as ddb:
            threads = await ddb.Table(_THREADS)
            await threads.update_item(
                Key={"thread_id": thread_id},
                UpdateExpression="SET last_activity_at = :w",
                ExpressionAttributeValues={":w": when.isoformat()},
            )

    # ------------------------------------------------------------------
    # Posts
    # ------------------------------------------------------------------
    async def create_post(self, post: Post) -> None:
        async with self._resource() as ddb:
            posts = await ddb.Table(_POSTS)
            await posts.put_item(Item=self._to_item(post))
        await self.update_thread_activity(post.thread_id, post.created_at)

    async def get_post(self, post_id: str) -> Post | None:
        async with self._resource() as ddb:
            posts = await ddb.Table(_POSTS)
            resp = await posts.get_item(Key={"post_id": post_id})
        item = resp.get("Item")
        return None if item is None else self._from_post(item)

    async def get_posts_by_thread(self, thread_id: str) -> list[Post]:
        async with self._resource() as ddb:
            posts = await ddb.Table(_POSTS)
            resp = await posts.query(
                IndexName="thread-posts-index",
                KeyConditionExpression="thread_id = :t",
                ExpressionAttributeValues={":t": thread_id},
                ScanIndexForward=True,  # oldest first
            )
        items = resp.get("Items") or []
        return [self._from_post(i) for i in items]

    # ------------------------------------------------------------------
    # NewsItems
    # ------------------------------------------------------------------
    async def create_news_item(self, item: NewsItem) -> None:
        async with self._resource() as ddb:
            news = await ddb.Table(_NEWS_ITEMS)
            await news.put_item(Item=self._to_item(item))

    async def get_news_item(self, item_id: str) -> NewsItem | None:
        async with self._resource() as ddb:
            news = await ddb.Table(_NEWS_ITEMS)
            resp = await news.get_item(Key={"item_id": item_id})
        item = resp.get("Item")
        return None if item is None else self._from_news_item(item)

    async def list_new_news_items(self, limit: int = 100) -> list[NewsItem]:
        async with self._resource() as ddb:
            news = await ddb.Table(_NEWS_ITEMS)
            resp = await news.query(
                IndexName="status-index",
                KeyConditionExpression="#s = :new",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={":new": "new"},
                Limit=limit,
                ScanIndexForward=True,  # oldest fetched first
            )
        items = resp.get("Items") or []
        return [self._from_news_item(i) for i in items]

    async def update_news_item_status(
        self,
        item_id: str,
        status: Literal["new", "promoted", "skipped"],
        promoted_thread_id: str | None = None,
    ) -> None:
        async with self._resource() as ddb:
            news = await ddb.Table(_NEWS_ITEMS)
            if promoted_thread_id is None:
                await news.update_item(
                    Key={"item_id": item_id},
                    UpdateExpression="SET #s = :s",
                    ExpressionAttributeNames={"#s": "status"},
                    ExpressionAttributeValues={":s": status},
                )
            else:
                await news.update_item(
                    Key={"item_id": item_id},
                    UpdateExpression="SET #s = :s, promoted_thread_id = :p",
                    ExpressionAttributeNames={"#s": "status"},
                    ExpressionAttributeValues={":s": status, ":p": promoted_thread_id},
                )

    async def news_item_exists_by_url(self, url: str) -> bool:
        async with self._resource() as ddb:
            news = await ddb.Table(_NEWS_ITEMS)
            resp = await news.scan(
                FilterExpression="#u = :u",
                ExpressionAttributeNames={"#u": "url"},
                ExpressionAttributeValues={":u": url},
                ProjectionExpression="item_id",
                Limit=1,
            )
        return bool(resp.get("Items"))
