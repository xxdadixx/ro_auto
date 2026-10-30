import os
import shutil
import socket
import subprocess
import numpy as np
import cv2
from typing import List, Tuple, Optional

# Relative import within the same package directory
from .path_finder import MuMuPathFinder


class ADBManager:
    """Handles ADB device auto-discovery, screen frame capture, and input commands."""

    def __init__(self, host: str = "127.0.0.1", adb_path: Optional[str] = None) -> None:
        self.host = host
        self.adb_path = adb_path or MuMuPathFinder.auto_detect_adb()
        print(f"[ADBManager] Using ADB binary at: {self.adb_path}")

    def scan_mumu_instances(
        self, start_port: int = 7555, max_ports: int = 20
    ) -> List[str]:
        """Scans local ports to discover active MuMuPlayer ADB instances."""
        active_devices: List[str] = []
        candidate_ports = [7555] + list(range(16384, 16384 + max_ports))

        for port in candidate_ports:
            serial = f"{self.host}:{port}"
            if self._is_port_open(self.host, port):
                subprocess.run(
                    [self.adb_path, "connect", serial],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                active_devices.append(serial)

        result = subprocess.run(
            [self.adb_path, "devices"], capture_output=True, text=True
        )
        connected: List[str] = []
        for line in result.stdout.splitlines():
            if "\tdevice" in line:
                dev_serial = line.split("\t")[0]
                if dev_serial in active_devices or dev_serial.startswith(self.host):
                    connected.append(dev_serial)

        return connected

    def _is_port_open(self, host: str, port: int, timeout: float = 0.5) -> bool:
        """Utility to check if a TCP port is open."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            return sock.connect_ex((host, port)) == 0

    def capture_frame(self, device_serial: str) -> Optional[np.ndarray]:
        """Captures frame directly from ADB stdout stream without focusing the window."""
        cmd = [self.adb_path, "-s", device_serial, "exec-out", "screencap", "-p"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw_png, stderr = process.communicate()

        if process.returncode != 0 or not raw_png:
            return None

        frame_array = np.frombuffer(raw_png, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        return frame

    def tap(self, device_serial: str, x: int, y: int) -> None:
        """Executes a single tap at specified coordinates."""
        subprocess.run(
            [
                self.adb_path,
                "-s",
                device_serial,
                "shell",
                "input",
                "tap",
                str(x),
                str(y),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def swipe(
        self,
        device_serial: str,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration_ms: int = 300,
    ) -> None:
        """Executes a swipe gesture between coordinates."""
        cmd = [
            self.adb_path,
            "-s",
            device_serial,
            "shell",
            "input",
            "swipe",
            str(start_x),
            str(start_y),
            str(end_x),
            str(end_y),
            str(duration_ms),
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
