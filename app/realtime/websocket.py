"""WebSocket endpoint streaming live events for a thread.

One connection per thread subscribes to :class:`app.realtime.broker.Broker`
and forwards every event to the client as JSON. The loop exits cleanly
when the client disconnects or when any unexpected exception bubbles
from the socket; subscription is removed in ``finally`` so dead
connections don't leak queue slots.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.realtime.broker import Broker

_log = structlog.get_logger(__name__)


def create_websocket_router(*, broker: Broker) -> APIRouter:
    """Build an ``APIRouter`` exposing ``/ws/threads/{thread_id}``."""
    router = APIRouter()

    @router.websocket("/ws/threads/{thread_id}")
    async def thread_stream(websocket: WebSocket, thread_id: str) -> None:
        await websocket.accept()
        queue = await broker.subscribe(thread_id)
        try:
            while True:
                event = await queue.get()
                await websocket.send_json(event)
        except WebSocketDisconnect:
            _log.info("realtime.ws_disconnect", thread_id=thread_id)
        finally:
            await broker.unsubscribe(thread_id, queue)

    return router
