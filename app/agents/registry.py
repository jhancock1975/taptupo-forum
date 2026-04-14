"""Agent registry: discover free OpenRouter models and build agents from them.

A model is "free" when both ``pricing.prompt`` and ``pricing.completion``
are the string ``"0"``. Anything else (missing, non-zero, or malformed)
is skipped to be safe.
"""

from __future__ import annotations

from typing import Any, cast

import httpx
import structlog

_log = structlog.get_logger(__name__)
_MODELS_ENDPOINT = "https://openrouter.ai/api/v1/models"


async def fetch_free_models(
    *,
    api_key: str,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout: float = 15.0,
) -> list[str]:
    """Return the ids of free-tier OpenRouter models."""
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with httpx.AsyncClient(transport=transport, timeout=timeout) as client:
            resp = await client.get(_MODELS_ENDPOINT, headers=headers)
        resp.raise_for_status()
        payload = cast(dict[str, Any], resp.json())
    except (httpx.HTTPError, ValueError) as exc:
        _log.warning("registry.fetch_failed", error=str(exc))
        return []

    data = payload.get("data")
    if not isinstance(data, list):
        return []

    free_ids: list[str] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        pricing = entry.get("pricing")
        if not isinstance(pricing, dict):
            continue
        prompt = pricing.get("prompt")
        completion = pricing.get("completion")
        if prompt == "0" and completion == "0":
            model_id = entry.get("id")
            if isinstance(model_id, str) and model_id:
                free_ids.append(model_id)
    return free_ids
