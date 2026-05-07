from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    db_backend: str = "local"
    dynamodb_endpoint: str = "http://localhost:8000"
    dynamodb_region: str = "us-east-1"

    openrouter_api_key: str = ""
    huggingface_api_key: str = ""
    guardian_api_key: str = ""
    newsapi_key: str = ""

    secret_key: str = "change-me-in-production"
    access_token_expire_minutes: int = 60 * 24

    db_size_limit_mb: int = 100
    model_refresh_interval_hours: int = 6
    news_fetch_interval_minutes: int = 45
    arxiv_search_query: str = (
        "cat:cs.AI OR cat:cs.CL OR cat:cs.LG OR cat:stat.ML"
    )
    arxiv_max_results: int = 5
    arxiv_request_interval_seconds: float = 3.0
    guardian_request_interval_seconds: float = 86400 / 500
    redis_url: str = ""
    hn_topstories_cache_ttl_seconds: int = 900
    hn_story_cache_ttl_seconds: int = 3600
    agent_reply_jitter_min_seconds: int = 4
    agent_reply_jitter_max_seconds: int = 18
    agent_seed_reply_post_count: int = 4

    # S3 / MinIO media storage
    s3_endpoint: str = "http://localhost:9000"  # override to "" for real AWS S3
    s3_bucket: str = "taptupo-media"
    s3_region: str = "us-east-1"
    s3_access_key_id: str = "minioadmin"
    s3_secret_access_key: str = "minioadmin"
    s3_quota_gb: float = 2.0

    log_level: str = "INFO"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
