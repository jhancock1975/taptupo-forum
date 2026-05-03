"""Tests for app/storage/s3.py."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.storage import s3 as s3_store


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[str, dict[str, Any]] = {}
        self.put_calls: list[dict[str, Any]] = []
        self.bucket_exists: bool = True
        self.create_bucket_calls: list[dict[str, Any]] = []

    def head_bucket(self, Bucket: str) -> dict[str, Any]:
        if not self.bucket_exists:
            raise Exception("NoSuchBucket")
        return {}

    def create_bucket(self, **kwargs: Any) -> dict[str, Any]:
        self.create_bucket_calls.append(kwargs)
        self.bucket_exists = True
        return {}

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        self.put_calls.append(kwargs)
        key = kwargs["Key"]
        self.objects[key] = {"Size": len(kwargs.get("Body", b""))}
        return {}

    def get_paginator(self, operation: str) -> "FakePaginator":
        return FakePaginator(self.objects)


class FakePaginator:
    def __init__(self, objects: dict[str, Any]) -> None:
        self._objects = objects

    def paginate(self, Bucket: str) -> list[dict[str, Any]]:
        contents = [{"Size": v["Size"]} for v in self._objects.values()]
        return [{"Contents": contents}] if contents else [{}]


@pytest.mark.anyio
async def test_ensure_bucket_exists_skips_when_already_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeS3Client()
    client.bucket_exists = True
    monkeypatch.setattr(s3_store, "_make_client", lambda: client)

    await s3_store.ensure_bucket_exists()

    assert client.create_bucket_calls == []


@pytest.mark.anyio
async def test_ensure_bucket_exists_creates_bucket_in_us_east_1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeS3Client()
    client.bucket_exists = False
    monkeypatch.setattr(s3_store, "_make_client", lambda: client)
    monkeypatch.setattr(s3_store.settings, "s3_region", "us-east-1")

    await s3_store.ensure_bucket_exists()

    assert len(client.create_bucket_calls) == 1
    assert "CreateBucketConfiguration" not in client.create_bucket_calls[0]


@pytest.mark.anyio
async def test_ensure_bucket_exists_creates_bucket_with_location_for_other_regions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeS3Client()
    client.bucket_exists = False
    monkeypatch.setattr(s3_store, "_make_client", lambda: client)
    monkeypatch.setattr(s3_store.settings, "s3_region", "eu-west-1")

    await s3_store.ensure_bucket_exists()

    assert client.create_bucket_calls[0]["CreateBucketConfiguration"] == {
        "LocationConstraint": "eu-west-1"
    }


@pytest.mark.anyio
async def test_ensure_bucket_exists_logs_on_create_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_client() -> FakeS3Client:
        c = FakeS3Client()
        c.bucket_exists = False

        def failing_create(**kwargs: Any) -> None:
            raise RuntimeError("permission denied")

        c.create_bucket = failing_create  # type: ignore[method-assign]
        return c

    monkeypatch.setattr(s3_store, "_make_client", broken_client)
    # Should not raise
    await s3_store.ensure_bucket_exists()


@pytest.mark.anyio
async def test_upload_media_returns_minio_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeS3Client()
    monkeypatch.setattr(s3_store, "_make_client", lambda: client)
    monkeypatch.setattr(s3_store.settings, "s3_endpoint", "http://localhost:9000")
    monkeypatch.setattr(s3_store.settings, "s3_bucket", "taptupo-media")

    url = await s3_store.upload_media(b"imagedata", "media/test.png", "image/png")

    assert url == "http://localhost:9000/taptupo-media/media/test.png"
    assert client.put_calls[0]["Body"] == b"imagedata"
    assert client.put_calls[0]["ContentType"] == "image/png"


@pytest.mark.anyio
async def test_upload_media_returns_aws_url_when_no_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeS3Client()
    monkeypatch.setattr(s3_store, "_make_client", lambda: client)
    monkeypatch.setattr(s3_store.settings, "s3_endpoint", "")
    monkeypatch.setattr(s3_store.settings, "s3_bucket", "mybucket")
    monkeypatch.setattr(s3_store.settings, "s3_region", "us-east-1")

    url = await s3_store.upload_media(b"data", "key/obj.png", "image/png")

    assert url == "https://mybucket.s3.us-east-1.amazonaws.com/key/obj.png"


@pytest.mark.anyio
async def test_get_storage_bytes_sums_object_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeS3Client()
    client.objects = {"a": {"Size": 100}, "b": {"Size": 250}}
    monkeypatch.setattr(s3_store, "_make_client", lambda: client)

    total = await s3_store.get_storage_bytes()

    assert total == 350


@pytest.mark.anyio
async def test_get_storage_bytes_returns_zero_for_empty_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeS3Client()
    monkeypatch.setattr(s3_store, "_make_client", lambda: client)

    total = await s3_store.get_storage_bytes()

    assert total == 0


@pytest.mark.anyio
async def test_get_storage_bytes_returns_zero_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ErrorPaginator:
        def paginate(self, Bucket: str) -> None:
            raise RuntimeError("access denied")

    def broken_client() -> FakeS3Client:
        c = FakeS3Client()
        c.get_paginator = lambda op: ErrorPaginator()  # type: ignore[method-assign]
        return c

    monkeypatch.setattr(s3_store, "_make_client", broken_client)
    total = await s3_store.get_storage_bytes()
    assert total == 0
