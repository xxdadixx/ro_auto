import asyncio
import base64
import json
import threading
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import uvicorn
import webview

from core.adb_manager import ADBManager
from engine.instance_worker import InstanceWorker
from vision.vision import VisionEngine

app = FastAPI()
adb_mgr = ADBManager()
vision = VisionEngine()
active_workers = {}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Broadcast instance statuses, frames, and logs to the GUI
            data = []
            for serial, worker in active_workers.items():
                data.append(
                    {
                        "serial": serial,
                        "state": worker.state.name,
                        "is_running": worker.is_running,
                    }
                )
            await websocket.send_json({"type": "STATE_UPDATE", "instances": data})
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass


def start_gui():
    # Launches native desktop window containing the Liquid Glass HTML GUI
    webview.create_window(
        "Ragnarok: The New World - Multi-Bot Launcher",
        "http://127.0.0.1:8000",
        width=1400,
        height=900,
        frameless=False,
    )
    webview.start()


if __name__ == "__main__":
    # Start FastAPI server in a background thread
    server_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error"),
        daemon=True,
    )
    server_thread.start()

    # Launch Desktop Webview GUI Window
    start_gui()
