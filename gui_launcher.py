import asyncio
import threading
from typing import Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import webview

from engine.instance_worker import InstanceWorker

app = FastAPI()
active_workers: Dict[str, InstanceWorker] = {}
_workers_lock = threading.Lock()


def register_worker(serial: str, worker: InstanceWorker) -> None:
    """Registers an active worker instance for telemetry broadcasting."""
    with _workers_lock:
        active_workers[serial] = worker


def unregister_worker(serial: str) -> None:
    """Removes a worker instance from telemetry broadcasting."""
    with _workers_lock:
        active_workers.pop(serial, None)


def _get_workers_telemetry_snapshot() -> list:
    """Safely extracts a snapshot of worker statuses under thread lock without blocking asyncio."""
    data = []
    with _workers_lock:
        for serial, worker in list(active_workers.items()):
            data.append(
                {
                    "serial": serial,
                    "state": getattr(worker.state, "name", "UNKNOWN"),
                    "is_running": getattr(worker, "is_running", False),
                }
            )
    return data


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Broadcasts telemetry updates without blocking the asyncio event loop."""
    await websocket.accept()
    try:
        while True:
            # Delegate thread-bound state collection to a thread worker to keep asyncio non-blocking
            data = await asyncio.to_thread(_get_workers_telemetry_snapshot)
            await websocket.send_json({"type": "STATE_UPDATE", "instances": data})
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    except Exception as err:
        print(f"[WebSocket] Telemetry connection closed: {err}")


def start_gui():
    """Launches native desktop window containing the Liquid Glass HTML GUI."""
    webview.create_window(
        "Ragnarok: The New World - Multi-Bot Launcher",
        "http://127.0.0.1:8000",
        width=1400,
        height=900,
        frameless=False,
    )
    webview.start()


if __name__ == "__main__":
    server_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error"),
        daemon=True,
    )
    server_thread.start()

    start_gui()
