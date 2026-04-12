"""Unit tests for scripts.setup_tables using moto."""

from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from scripts.setup_tables import create_tables

EXPECTED_TABLES = {"users", "threads", "posts", "news_items"}


@pytest.mark.unit
@mock_aws
def test_create_tables_creates_all_four() -> None:
    client = boto3.client("dynamodb", region_name="us-east-1")
    create_tables(client)
    listed = set(client.list_tables()["TableNames"])
    assert listed == EXPECTED_TABLES


@pytest.mark.unit
@mock_aws
def test_create_tables_sets_expected_key_schemas() -> None:
    client = boto3.client("dynamodb", region_name="us-east-1")
    create_tables(client)

    expected_pks = {
        "users": "user_id",
        "threads": "thread_id",
        "posts": "post_id",
        "news_items": "item_id",
    }
    for table_name, pk_name in expected_pks.items():
        desc = client.describe_table(TableName=table_name)["Table"]
        key_schema = desc["KeySchema"]
        assert len(key_schema) == 1
        assert key_schema[0]["AttributeName"] == pk_name
        assert key_schema[0]["KeyType"] == "HASH"
        attr_defs = {a["AttributeName"]: a["AttributeType"] for a in desc["AttributeDefinitions"]}
        assert attr_defs[pk_name] == "S"
        assert desc["BillingModeSummary"]["BillingMode"] == "PAY_PER_REQUEST"


@pytest.mark.unit
@mock_aws
def test_create_tables_sets_expected_gsis() -> None:
    client = boto3.client("dynamodb", region_name="us-east-1")
    create_tables(client)

    # users: username-index
    users = client.describe_table(TableName="users")["Table"]
    users_gsis = {g["IndexName"]: g for g in users.get("GlobalSecondaryIndexes", [])}
    assert "username-index" in users_gsis
    users_keys = users_gsis["username-index"]["KeySchema"]
    assert users_keys[0]["AttributeName"] == "username"
    assert users_keys[0]["KeyType"] == "HASH"

    # posts: thread-posts-index
    posts = client.describe_table(TableName="posts")["Table"]
    posts_gsis = {g["IndexName"]: g for g in posts.get("GlobalSecondaryIndexes", [])}
    assert "thread-posts-index" in posts_gsis
    posts_keys = {k["KeyType"]: k["AttributeName"] for k in posts_gsis["thread-posts-index"]["KeySchema"]}
    assert posts_keys["HASH"] == "thread_id"
    assert posts_keys["RANGE"] == "created_at"

    # news_items: status-index
    news = client.describe_table(TableName="news_items")["Table"]
    news_gsis = {g["IndexName"]: g for g in news.get("GlobalSecondaryIndexes", [])}
    assert "status-index" in news_gsis
    news_keys = {k["KeyType"]: k["AttributeName"] for k in news_gsis["status-index"]["KeySchema"]}
    assert news_keys["HASH"] == "status"
    assert news_keys["RANGE"] == "fetched_at"


@pytest.mark.unit
@mock_aws
def test_create_tables_is_idempotent() -> None:
    client = boto3.client("dynamodb", region_name="us-east-1")
    create_tables(client)
    # Second call must not raise.
    create_tables(client)
    listed = set(client.list_tables()["TableNames"])
    assert listed == EXPECTED_TABLES
