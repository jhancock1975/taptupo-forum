"""Application settings, loaded from environment variables (or ``.env``).

Settings are strictly validated at startup by Pydantic. Unknown
``DB_BACKEND`` values raise ``ValueError`` so misconfiguration fails
loudly instead of silently falling back.
"""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the forum application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    db_backend: Literal["local", "aws"] = "local"
    dynamodb_endpoint: str = "http://localhost:8000"
    aws_region: str = "us-east-1"

    openrouter_api_key: str = ""
    guardian_api_key: str = ""
    newsapi_key: str = ""

    session_secret: str = "changeme"  # noqa: S105  # dev-only default; production sets via env
