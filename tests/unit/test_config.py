"""Unit tests for app.config.Settings."""

from __future__ import annotations

import pytest

from app.config import Settings


@pytest.mark.unit
def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "DB_BACKEND",
        "DYNAMODB_ENDPOINT",
        "AWS_REGION",
        "OPENROUTER_API_KEY",
        "GUARDIAN_API_KEY",
        "NEWSAPI_KEY",
        "SESSION_SECRET",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.db_backend == "local"
    assert settings.dynamodb_endpoint == "http://localhost:8000"
    assert settings.aws_region == "us-east-1"
    assert settings.session_secret == "changeme"  # pragma: allowlist secret
    assert settings.openrouter_api_key == ""
    assert settings.guardian_api_key == ""
    assert settings.newsapi_key == ""


@pytest.mark.unit
def test_settings_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_BACKEND", "aws")
    monkeypatch.setenv("AWS_REGION", "eu-west-2")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")  # pragma: allowlist secret
    monkeypatch.setenv("SESSION_SECRET", "super-secret")  # pragma: allowlist secret

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.db_backend == "aws"
    assert settings.aws_region == "eu-west-2"
    assert settings.openrouter_api_key == "or-test"  # pragma: allowlist secret
    assert settings.session_secret == "super-secret"  # pragma: allowlist secret


@pytest.mark.unit
def test_settings_rejects_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_BACKEND", "bogus")
    with pytest.raises(ValueError):
        Settings(_env_file=None)  # type: ignore[call-arg]
