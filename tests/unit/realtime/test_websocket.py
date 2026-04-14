"""Unit tests for app.realtime.websocket."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.realtime.broker import Broker
from app.realtime.websocket import create_websocket_router


@pytest.mark.unit
def test_websocket_forwards_broker_events() -> None:
    broker = Broker()
    app = FastAPI()
    app.include_router(create_websocket_router(broker=broker))

    with TestClient(app) as client, client.websocket_connect("/ws/threads/t-1") as ws:
        # Give the server a tick to register the subscription.
        async def _publish() -> None:
            # Wait briefly for subscriber registration in the endpoint loop.
            for _ in range(50):
                if broker._subs.get("t-1"):
                    break
                await asyncio.sleep(0.01)
            await broker.publish("t-1", {"type": "post.created", "n": 1})

        asyncio.run(_publish())
        event = ws.receive_json()
        assert event == {"type": "post.created", "n": 1}


@pytest.mark.unit
def test_websocket_other_topic_does_not_leak() -> None:
    broker = Broker()
    app = FastAPI()
    app.include_router(create_websocket_router(broker=broker))

    with TestClient(app) as client, client.websocket_connect("/ws/threads/t-1") as ws:

        async def _publish() -> None:
            for _ in range(50):
                if broker._subs.get("t-1"):
                    break
                await asyncio.sleep(0.01)
            await broker.publish("t-other", {"n": 99})
            await broker.publish("t-1", {"n": 1})

        asyncio.run(_publish())
        event = ws.receive_json()
        assert event == {"n": 1}
