"""Create the DynamoDB tables required by taptupo-forum.

Idempotent: skips tables that already exist. Reads configuration from
environment variables so the same script works against DynamoDB Local
(by setting ``DYNAMODB_ENDPOINT``) and real AWS (by leaving it unset and
providing standard AWS credentials).

Environment variables:
    DYNAMODB_ENDPOINT   Optional endpoint URL, e.g. ``http://localhost:8000``.
    AWS_REGION          Region name; defaults to ``us-east-1``.
    AWS_ACCESS_KEY_ID   Optional; required for real AWS.
    AWS_SECRET_ACCESS_KEY   Optional; required for real AWS.

Usage::

    uv run python scripts/setup_tables.py
"""

from __future__ import annotations

import os
from typing import Any

import boto3  # type: ignore[import-untyped]
from botocore.exceptions import ClientError  # type: ignore[import-untyped]

TABLE_DEFINITIONS: list[dict[str, Any]] = [
    {
        "TableName": "users",
        "KeySchema": [{"AttributeName": "user_id", "KeyType": "HASH"}],
        "AttributeDefinitions": [
            {"AttributeName": "user_id", "AttributeType": "S"},
            {"AttributeName": "username", "AttributeType": "S"},
        ],
        "BillingMode": "PAY_PER_REQUEST",
        "GlobalSecondaryIndexes": [
            {
                "IndexName": "username-index",
                "KeySchema": [{"AttributeName": "username", "KeyType": "HASH"}],
                "Projection": {"ProjectionType": "KEYS_ONLY"},
            }
        ],
    },
    {
        "TableName": "threads",
        "KeySchema": [{"AttributeName": "thread_id", "KeyType": "HASH"}],
        "AttributeDefinitions": [
            {"AttributeName": "thread_id", "AttributeType": "S"},
        ],
        "BillingMode": "PAY_PER_REQUEST",
    },
    {
        "TableName": "posts",
        "KeySchema": [{"AttributeName": "post_id", "KeyType": "HASH"}],
        "AttributeDefinitions": [
            {"AttributeName": "post_id", "AttributeType": "S"},
            {"AttributeName": "thread_id", "AttributeType": "S"},
            {"AttributeName": "created_at", "AttributeType": "S"},
        ],
        "BillingMode": "PAY_PER_REQUEST",
        "GlobalSecondaryIndexes": [
            {
                "IndexName": "thread-posts-index",
                "KeySchema": [
                    {"AttributeName": "thread_id", "KeyType": "HASH"},
                    {"AttributeName": "created_at", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        ],
    },
    {
        "TableName": "news_items",
        "KeySchema": [{"AttributeName": "item_id", "KeyType": "HASH"}],
        "AttributeDefinitions": [
            {"AttributeName": "item_id", "AttributeType": "S"},
            {"AttributeName": "status", "AttributeType": "S"},
            {"AttributeName": "fetched_at", "AttributeType": "S"},
        ],
        "BillingMode": "PAY_PER_REQUEST",
        "GlobalSecondaryIndexes": [
            {
                "IndexName": "status-index",
                "KeySchema": [
                    {"AttributeName": "status", "KeyType": "HASH"},
                    {"AttributeName": "fetched_at", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        ],
    },
]


def create_tables(client: Any) -> list[str]:
    """Ensure all required DynamoDB tables exist; return their names.

    Idempotent: if a table already exists the ``ResourceInUseException``
    raised by DynamoDB is caught and treated as success.
    """
    ensured: list[str] = []
    for definition in TABLE_DEFINITIONS:
        table_name: str = definition["TableName"]
        try:
            client.create_table(**definition)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code != "ResourceInUseException":
                raise
        ensured.append(table_name)
    return ensured


def main() -> None:
    """Build a boto3 DynamoDB client from env vars and create tables."""
    endpoint_url = os.environ.get("DYNAMODB_ENDPOINT")
    region = os.environ.get("AWS_REGION", "us-east-1")

    client_kwargs: dict[str, Any] = {"region_name": region}
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url

    client = boto3.client("dynamodb", **client_kwargs)
    created = create_tables(client)
    for name in created:
        print(f"ensured table: {name}")


if __name__ == "__main__":
    main()
