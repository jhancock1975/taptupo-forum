"""Unit tests for app.agents.registry."""

from __future__ import annotations

import httpx
import pytest

from app.agents.registry import fetch_free_models

_SAMPLE = {
    "data": [
        {"id": "meta/llama-free", "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "anthropic/claude-paid", "pricing": {"prompt": "0.001", "completion": "0.005"}},
        {"id": "google/gemma-free", "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "broken-model"},  # missing pricing — skipped
    ]
}


def _transport(payload: dict[str, object], *, status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fetch_free_models_filters_to_zero_pricing() -> None:
    models = await fetch_free_models(api_key="test", transport=_transport(_SAMPLE))
    assert sorted(models) == ["google/gemma-free", "meta/llama-free"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fetch_free_models_returns_empty_on_http_error() -> None:
    models = await fetch_free_models(api_key="test", transport=_transport({}, status=500))
    assert models == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fetch_free_models_handles_garbage_shape() -> None:
    models = await fetch_free_models(api_key="test", transport=_transport({"data": "not-a-list"}))
    assert models == []
