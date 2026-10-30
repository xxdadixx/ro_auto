import socket
import subprocess
from typing import List, Optional
import cv2
import numpy as np

from config import ADBConfig
from .path_finder import MuMuPathFinder


class ADBManager:
    """Handles ADB device auto-discovery, screen frame capture, and input commands."""

    def __init__(
        self, host: Optional[str] = None, adb_path: Optional[str] = None
    ) -> None:
        self.config = ADBConfig()
        self.host = host or self.config.host
        self.adb_path = (
            adb_path or self.config.custom_adb_path or MuMuPathFinder.auto_detect_adb()
        )
        print(f"[ADBManager] Using ADB binary at: {self.adb_path}")

    def scan_mumu_instances(self) -> List[str]:
        """Scans local TCP ports to discover and connect active MuMuPlayer ADB instances."""
        candidate_ports = [self.config.default_port] + list(
            range(
                self.config.start_port, self.config.start_port + self.config.max_ports
            )
        )

        for port in candidate_ports:
            if self._is_port_open(self.host, port):
                serial = f"{self.host}:{port}"
                try:
                    subprocess.run(
                        [self.adb_path, "connect", serial],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=3.0,
                    )
                except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                    print(
                        f"[ADBManager] Connection attempt to {serial} timed out or failed; skipping."
                    )
                    continue

        try:
            result = subprocess.run(
                [self.adb_path, "devices"], capture_output=True, text=True, timeout=5.0
            )
            connected: List[str] = []
            for line in result.stdout.splitlines():
                if "\tdevice" in line:
                    dev_serial = line.split("\t")[0]
                    if dev_serial.startswith(self.host):
                        connected.append(dev_serial)

            return connected
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            print("[ADBManager] Error executing 'adb devices' query.")
            return []

    def _is_port_open(self, host: str, port: int) -> bool:
        """Utility to check if a local TCP port is accepting connections."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.config.connection_timeout)
                return sock.connect_ex((host, port)) == 0
        except OSError:
            return False

    def capture_frame(self, device_serial: str) -> Optional[np.ndarray]:
        """Captures raw image frame from ADB stdout stream without focusing emulator window."""
        cmd = [self.adb_path, "-s", device_serial, "exec-out", "screencap", "-p"]
        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            raw_png, _ = process.communicate(timeout=3.0)
            if process.returncode != 0 or not raw_png:
                return None

            frame_array = np.frombuffer(raw_png, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            return frame
        except (subprocess.TimeoutExpired, Exception):
            return None

    def tap(self, device_serial: str, x: int, y: int) -> None:
        """Executes a single tap action at specified coordinates."""
        cmd = [
            self.adb_path,
            "-s",
            device_serial,
            "shell",
            "input",
            "tap",
            str(x),
            str(y),
        ]
        try:
            subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2.0
            )
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass

    def swipe(
        self,
        device_serial: str,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration_ms: int = 300,
    ) -> None:
        """Executes a linear swipe gesture between specified coordinates."""
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
        try:
            subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2.0
            )
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass
