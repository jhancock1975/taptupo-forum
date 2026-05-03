#!/usr/bin/env python3
"""Create DynamoDB tables for local development.

Usage:
    python -m scripts.setup_tables
"""
from __future__ import annotations

import sys

import boto3
from botocore.exceptions import ClientError


def main() -> None:
    endpoint = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    print(f"Connecting to DynamoDB at {endpoint}")

    client = boto3.client(
        "dynamodb",
        endpoint_url=endpoint,
        region_name="us-east-1",
        aws_access_key_id="local",
        aws_secret_access_key="local",
    )

    existing = client.list_tables().get("TableNames", [])

    tables = [
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

    for tdef in tables:
        name = tdef["TableName"]
        if name in existing:
            print(f"  Table '{name}' already exists, skipping.")
            continue
        try:
            client.create_table(**tdef)
            print(f"  Created table '{name}'.")
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceInUseException":
                print(f"  Table '{name}' already exists (race).")
            else:
                raise

    print("Done.")


if __name__ == "__main__":
    main()
