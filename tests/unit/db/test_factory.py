"""Unit tests for the repository factory."""

from __future__ import annotations

import pytest

from app.db.dynamo import DynamoRepository
from app.db.factory import get_repository

pytestmark = pytest.mark.unit


def test_factory_default_returns_local_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DB_BACKEND", raising=False)
    monkeypatch.delenv("DYNAMODB_ENDPOINT", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    repo = get_repository()
    assert isinstance(repo, DynamoRepository)
    assert repo.endpoint_url == "http://localhost:8000"
    assert repo.region_name == "us-east-1"


def test_factory_local_respects_custom_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_BACKEND", "local")
    monkeypatch.setenv("DYNAMODB_ENDPOINT", "http://ddb:8765")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    repo = get_repository()
    assert isinstance(repo, DynamoRepository)
    assert repo.endpoint_url == "http://ddb:8765"
    assert repo.region_name == "eu-west-1"


def test_factory_aws_backend_has_no_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_BACKEND", "aws")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    repo = get_repository()
    assert isinstance(repo, DynamoRepository)
    assert repo.endpoint_url is None
    assert repo.region_name == "us-west-2"


def test_factory_rejects_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_BACKEND", "postgres")
    with pytest.raises(ValueError, match="unknown DB_BACKEND"):
        get_repository()
