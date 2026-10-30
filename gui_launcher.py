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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
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
            await websocket.send_json({"type": "STATE_UPDATE", "instances": data})
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, Exception):
        pass


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
