"""S3 media storage service for agent-generated non-text content."""
from __future__ import annotations

import asyncio
from functools import partial
from typing import Any

import boto3
import structlog

from app.config import settings

logger = structlog.get_logger()


def _make_client() -> Any:
    kwargs: dict[str, Any] = {
        "region_name": settings.s3_region,
        "aws_access_key_id": settings.s3_access_key_id,
        "aws_secret_access_key": settings.s3_secret_access_key,
    }
    if settings.s3_endpoint:
        kwargs["endpoint_url"] = settings.s3_endpoint
    return boto3.client("s3", **kwargs)


async def _run(fn: Any, *args: Any, **kwargs: Any) -> Any:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(fn, *args, **kwargs))


async def ensure_bucket_exists() -> None:
    """Create the media bucket if it does not already exist."""
    client = _make_client()
    try:
        await _run(client.head_bucket, Bucket=settings.s3_bucket)
    except Exception:
        try:
            if settings.s3_region == "us-east-1":
                await _run(client.create_bucket, Bucket=settings.s3_bucket)
            else:
                await _run(
                    client.create_bucket,
                    Bucket=settings.s3_bucket,
                    CreateBucketConfiguration={"LocationConstraint": settings.s3_region},
                )
            logger.info("s3_bucket_created", bucket=settings.s3_bucket)
        except Exception:
            logger.exception("s3_bucket_create_failed", bucket=settings.s3_bucket)


async def upload_media(data: bytes, key: str, content_type: str) -> str:
    """Upload bytes to S3 and return the full object URL."""
    client = _make_client()
    await _run(
        client.put_object,
        Bucket=settings.s3_bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
    )
    if settings.s3_endpoint:
        return f"{settings.s3_endpoint.rstrip('/')}/{settings.s3_bucket}/{key}"
    return f"https://{settings.s3_bucket}.s3.{settings.s3_region}.amazonaws.com/{key}"


async def get_storage_bytes() -> int:
    """Return total bytes stored in the media bucket by paginating list_objects_v2."""
    client = _make_client()
    total = 0
    paginator = client.get_paginator("list_objects_v2")
    try:
        pages = await _run(
            lambda: list(
                paginator.paginate(Bucket=settings.s3_bucket)
            )
        )
        for page in pages:
            for obj in page.get("Contents", []):
                total += obj.get("Size", 0)
    except Exception:
        logger.exception("s3_list_objects_failed", bucket=settings.s3_bucket)
    return total
