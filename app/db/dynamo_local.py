from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal
from functools import partial
from typing import Any, Optional

import boto3
import structlog
from botocore.exceptions import ClientError

from app.config import settings
from app.db.interface import RepositoryInterface
from app.models.schemas import AgentConfig, NewsItem, Post, Thread, User

logger = structlog.get_logger()

TABLE_DEFS = [
    {
        "TableName": "Users",
        "KeySchema": [{"AttributeName": "user_id", "KeyType": "HASH"}],
        "AttributeDefinitions": [
            {"AttributeName": "user_id", "AttributeType": "S"},
            {"AttributeName": "username", "AttributeType": "S"},
        ],
        "GlobalSecondaryIndexes": [
            {
                "IndexName": "username-index",
                "KeySchema": [{"AttributeName": "username", "KeyType": "HASH"}],
                "Projection": {"ProjectionType": "ALL"},
                "ProvisionedThroughput": {
                    "ReadCapacityUnits": 5,
                    "WriteCapacityUnits": 5,
                },
            }
        ],
        "ProvisionedThroughput": {
            "ReadCapacityUnits": 5,
            "WriteCapacityUnits": 5,
        },
    },
    {
        "TableName": "Threads",
        "KeySchema": [{"AttributeName": "thread_id", "KeyType": "HASH"}],
        "AttributeDefinitions": [
            {"AttributeName": "thread_id", "AttributeType": "S"},
        ],
        "ProvisionedThroughput": {
            "ReadCapacityUnits": 5,
            "WriteCapacityUnits": 5,
        },
    },
    {
        "TableName": "Posts",
        "KeySchema": [{"AttributeName": "post_id", "KeyType": "HASH"}],
        "AttributeDefinitions": [
            {"AttributeName": "post_id", "AttributeType": "S"},
            {"AttributeName": "thread_id", "AttributeType": "S"},
        ],
        "GlobalSecondaryIndexes": [
            {
                "IndexName": "thread-index",
                "KeySchema": [{"AttributeName": "thread_id", "KeyType": "HASH"}],
                "Projection": {"ProjectionType": "ALL"},
                "ProvisionedThroughput": {
                    "ReadCapacityUnits": 5,
                    "WriteCapacityUnits": 5,
                },
            }
        ],
        "ProvisionedThroughput": {
            "ReadCapacityUnits": 5,
            "WriteCapacityUnits": 5,
        },
    },
    {
        "TableName": "NewsItems",
        "KeySchema": [{"AttributeName": "item_id", "KeyType": "HASH"}],
        "AttributeDefinitions": [
            {"AttributeName": "item_id", "AttributeType": "S"},
            {"AttributeName": "status", "AttributeType": "S"},
        ],
        "GlobalSecondaryIndexes": [
            {
                "IndexName": "status-index",
                "KeySchema": [{"AttributeName": "status", "KeyType": "HASH"}],
                "Projection": {"ProjectionType": "ALL"},
                "ProvisionedThroughput": {
                    "ReadCapacityUnits": 5,
                    "WriteCapacityUnits": 5,
                },
            }
        ],
        "ProvisionedThroughput": {
            "ReadCapacityUnits": 5,
            "WriteCapacityUnits": 5,
        },
    },
]


def _replace_floats(obj: Any) -> Any:
    """DynamoDB doesn't accept float; convert to Decimal."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _replace_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_floats(i) for i in obj]
    return obj


def _restore_floats(obj: Any) -> Any:
    """Convert Decimal back to float for Pydantic."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _restore_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore_floats(i) for i in obj]
    return obj


class DynamoLocalRepository(RepositoryInterface):
    def __init__(self) -> None:
        self._resource = boto3.resource(
            "dynamodb",
            endpoint_url=settings.dynamodb_endpoint,
            region_name=settings.dynamodb_region,
            aws_access_key_id="local",
            aws_secret_access_key="local",
        )

    def _table(self, name: str) -> Any:
        return self._resource.Table(name)

    async def _run(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.get_event_loop().run_in_executor(
            None, partial(fn, *args, **kwargs)
        )

    # ── Table Init ──

    async def init_tables(self) -> None:
        existing = await self._run(self._resource.meta.client.list_tables)
        existing_names: list[str] = existing.get("TableNames", [])
        for tdef in TABLE_DEFS:
            if tdef["TableName"] not in existing_names:
                try:
                    await self._run(self._resource.create_table, **tdef)
                    logger.info("table_created", table=tdef["TableName"])
                except ClientError as exc:
                    if exc.response["Error"]["Code"] != "ResourceInUseException":
                        raise

    # ── Users ──

    async def create_user(self, user: User) -> User:
        item = _replace_floats(user.model_dump(mode="json"))
        await self._run(self._table("Users").put_item, Item=item)
        return user

    async def get_user(self, user_id: str) -> Optional[User]:
        resp = await self._run(
            self._table("Users").get_item, Key={"user_id": user_id}
        )
        raw = resp.get("Item")
        if not raw:
            return None
        return self._user_from_item(raw)

    async def get_user_by_username(self, username: str) -> Optional[User]:
        resp = await self._run(
            self._table("Users").query,
            IndexName="username-index",
            KeyConditionExpression="username = :u",
            ExpressionAttributeValues={":u": username},
        )
        items = resp.get("Items", [])
        if not items:
            return None
        return self._user_from_item(items[0])

    async def list_agents(self) -> list[User]:
        resp = await self._run(
            self._table("Users").scan,
            FilterExpression="is_agent = :t",
            ExpressionAttributeValues={":t": True},
        )
        return [self._user_from_item(i) for i in resp.get("Items", [])]

    @staticmethod
    def _user_from_item(item: dict[str, Any]) -> User:
        item = _restore_floats(item)
        if item.get("agent_config") and isinstance(item["agent_config"], str):
            item["agent_config"] = json.loads(item["agent_config"])
        return User.model_validate(item)

    # ── Threads ──

    async def create_thread(self, thread: Thread) -> Thread:
        item = _replace_floats(thread.model_dump(mode="json"))
        await self._run(self._table("Threads").put_item, Item=item)
        return thread

    async def get_thread(self, thread_id: str) -> Optional[Thread]:
        resp = await self._run(
            self._table("Threads").get_item, Key={"thread_id": thread_id}
        )
        raw = resp.get("Item")
        if not raw:
            return None
        return Thread.model_validate(_restore_floats(raw))

    async def list_threads(self, limit: int = 50) -> list[Thread]:
        resp = await self._run(self._table("Threads").scan, Limit=limit)
        threads = [
            Thread.model_validate(_restore_floats(i))
            for i in resp.get("Items", [])
        ]
        threads.sort(key=lambda t: t.last_activity_at, reverse=True)
        return threads

    async def update_thread_activity(self, thread_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        await self._run(
            self._table("Threads").update_item,
            Key={"thread_id": thread_id},
            UpdateExpression="SET last_activity_at = :t, reply_count = reply_count + :inc",
            ExpressionAttributeValues={":t": now, ":inc": 1},
        )

    # ── Posts ──

    async def create_post(self, post: Post) -> Post:
        item = _replace_floats(post.model_dump(mode="json"))
        await self._run(self._table("Posts").put_item, Item=item)
        return post

    async def get_posts_by_thread(self, thread_id: str) -> list[Post]:
        resp = await self._run(
            self._table("Posts").query,
            IndexName="thread-index",
            KeyConditionExpression="thread_id = :t",
            ExpressionAttributeValues={":t": thread_id},
        )
        posts = [
            Post.model_validate(_restore_floats(i))
            for i in resp.get("Items", [])
        ]
        posts.sort(key=lambda p: p.created_at)
        return posts

    # ── News Items ──

    async def create_news_item(self, item: NewsItem) -> NewsItem:
        data = _replace_floats(item.model_dump(mode="json"))
        await self._run(self._table("NewsItems").put_item, Item=data)
        return item

    async def get_news_items_by_status(self, status: str) -> list[NewsItem]:
        resp = await self._run(
            self._table("NewsItems").query,
            IndexName="status-index",
            KeyConditionExpression="#s = :s",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":s": status},
        )
        return [
            NewsItem.model_validate(_restore_floats(i))
            for i in resp.get("Items", [])
        ]

    async def update_news_item_status(
        self,
        item_id: str,
        status: str,
        promoted_thread_id: Optional[str] = None,
    ) -> None:
        expr = "SET #s = :s"
        vals: dict[str, Any] = {":s": status}
        if promoted_thread_id:
            expr += ", promoted_thread_id = :p"
            vals[":p"] = promoted_thread_id
        await self._run(
            self._table("NewsItems").update_item,
            Key={"item_id": item_id},
            UpdateExpression=expr,
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues=vals,
        )

    async def get_news_item_by_url(self, url: str) -> Optional[NewsItem]:
        resp = await self._run(
            self._table("NewsItems").scan,
            FilterExpression="contains(#u, :u)",
            ExpressionAttributeNames={"#u": "url"},
            ExpressionAttributeValues={":u": url},
        )
        items = resp.get("Items", [])
        if not items:
            return None
        return NewsItem.model_validate(_restore_floats(items[0]))

    # ── Storage ──

    async def get_storage_bytes(self) -> int:
        """Sum TableSizeBytes across all tables via describe_table."""
        client = self._resource.meta.client
        total = 0
        for tdef in TABLE_DEFS:
            resp = await self._run(client.describe_table, TableName=tdef["TableName"])
            total += resp.get("Table", {}).get("TableSizeBytes", 0)
        return total

    # ── Agents ──

    async def update_agent_config(self, user_id: str, config: AgentConfig) -> None:
        """Overwrite the agent_config field for an existing agent user."""
        data = _replace_floats(config.model_dump(mode="json"))
        await self._run(
            self._table("Users").update_item,
            Key={"user_id": user_id},
            UpdateExpression="SET agent_config = :c",
            ExpressionAttributeValues={":c": data},
        )
