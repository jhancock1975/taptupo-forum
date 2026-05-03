from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, thread_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.setdefault(thread_id, []).append(ws)

    def disconnect(self, thread_id: str, ws: WebSocket) -> None:
        conns = self._connections.get(thread_id, [])
        if ws in conns:
            conns.remove(ws)

    async def broadcast(self, thread_id: str, html: str) -> None:
        dead: list[WebSocket] = []
        for ws in self._connections.get(thread_id, []):
            try:
                await ws.send_text(html)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(thread_id, ws)


ws_manager = ConnectionManager()


@router.websocket("/ws/thread/{thread_id}")
async def thread_ws(websocket: WebSocket, thread_id: str) -> None:
    await ws_manager.connect(thread_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(thread_id, websocket)
