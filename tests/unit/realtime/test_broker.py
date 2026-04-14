"""Unit tests for app.realtime.broker."""

from __future__ import annotations

import asyncio

import pytest

from app.realtime.broker import Broker


@pytest.mark.unit
@pytest.mark.asyncio
async def test_subscriber_receives_published_event() -> None:
    broker = Broker()
    q = await broker.subscribe("t-1")
    await broker.publish("t-1", {"hello": "world"})
    received = await asyncio.wait_for(q.get(), timeout=0.1)
    assert received == {"hello": "world"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_multiple_subscribers_on_same_topic_both_receive() -> None:
    broker = Broker()
    a = await broker.subscribe("t-1")
    b = await broker.subscribe("t-1")
    await broker.publish("t-1", {"n": 1})
    assert (await asyncio.wait_for(a.get(), 0.1))["n"] == 1
    assert (await asyncio.wait_for(b.get(), 0.1))["n"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_subscriber_on_other_topic_does_not_receive() -> None:
    broker = Broker()
    q = await broker.subscribe("t-other")
    await broker.publish("t-1", {"n": 1})
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(q.get(), timeout=0.05)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unsubscribe_stops_delivery() -> None:
    broker = Broker()
    q = await broker.subscribe("t-1")
    await broker.unsubscribe("t-1", q)
    await broker.publish("t-1", {"n": 1})
    assert q.empty()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_publish_with_no_subscribers_is_noop() -> None:
    broker = Broker()
    await broker.publish("t-nobody", {"n": 1})  # should not raise


@pytest.mark.unit
@pytest.mark.asyncio
async def test_full_queue_drops_message_without_blocking_publisher() -> None:
    broker = Broker(queue_maxsize=1)
    q = await broker.subscribe("t-1")
    await broker.publish("t-1", {"n": 1})
    # Second publish should not block even though queue is full.
    await asyncio.wait_for(broker.publish("t-1", {"n": 2}), timeout=0.1)
    first = await asyncio.wait_for(q.get(), 0.1)
    assert first == {"n": 1}
    assert q.empty()
