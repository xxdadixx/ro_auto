import concurrent.futures
import re
import socket
import subprocess
from typing import Dict, List, Optional, Set
import cv2
import numpy as np

from config import ADBConfig
from .path_finder import MuMuPathFinder


class ADBManager:
    """Handles ADB device auto-discovery, kernel boot UUID deduplication, fast frame capture, and input commands."""

    def __init__(
        self, host: Optional[str] = None, adb_path: Optional[str] = None
    ) -> None:
        self.config = ADBConfig()
        self.host = host or self.config.host
        self.adb_path = (
            adb_path or self.config.custom_adb_path or MuMuPathFinder.auto_detect_adb()
        )
        print(f"[ADBManager] Using ADB binary at: {self.adb_path}")

    def _generate_candidate_ports(self) -> List[int]:
        """Generates candidate TCP ports considering MuMuPlayer 12 32-port stride allocation."""
        ports: Set[int] = set()

        # Standard Emulator Ports
        ports.add(self.config.default_port)  # 7555
        ports.add(5555)
        ports.add(22471)

        # MuMu 12 Stride Ports
        for i in range(self.config.max_instances):
            ports.add(self.config.start_port + (i * self.config.port_stride))

        # Continuous Fallback Range
        for p in range(self.config.start_port, self.config.start_port + 32):
            ports.add(p)

        return sorted(list(ports))

    def _get_vm_boot_uuid(self, serial: str) -> Optional[str]:
        """Queries kernel boot_id UUID to uniquely identify distinct Android VM instances."""
        # Tier 1: Fast Kernel sysfs boot_id read (100% unique per running VM instance)
        try:
            res = subprocess.run(
                [
                    self.adb_path,
                    "-s",
                    serial,
                    "shell",
                    "cat",
                    "/proc/sys/kernel/random/boot_id",
                ],
                capture_output=True,
                text=True,
                timeout=1.5,
            )
            val = res.stdout.strip()
            if val and re.match(r"^[0-9a-fA-F\-]{32,36}$", val):
                return val.lower()
        except Exception:
            pass

        # Tier 2: MAC Address fallback
        try:
            res = subprocess.run(
                [
                    self.adb_path,
                    "-s",
                    serial,
                    "shell",
                    "cat",
                    "/sys/class/net/eth0/address",
                ],
                capture_output=True,
                text=True,
                timeout=1.5,
            )
            val = res.stdout.strip()
            if val and re.match(r"^([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}$", val):
                return f"mac_{val.lower()}"
        except Exception:
            pass

        # Tier 3: Network Hostname fallback
        try:
            res = subprocess.run(
                [self.adb_path, "-s", serial, "shell", "getprop", "net.hostname"],
                capture_output=True,
                text=True,
                timeout=1.5,
            )
            val = res.stdout.strip()
            if val and "error" not in val.lower() and "unknown" not in val.lower():
                return f"host_{val.lower()}"
        except Exception:
            pass

        return None

    def scan_mumu_instances(self) -> List[str]:
        """Scans local TCP ports in parallel and deduplicates duplicate alias ports mapping to the same VM."""
        candidate_ports = self._generate_candidate_ports()

        # Step 1: Rapid parallel TCP socket probing
        open_ports: List[int] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(32, len(candidate_ports))
        ) as executor:
            future_to_port = {
                executor.submit(self._is_port_open, self.host, port): port
                for port in candidate_ports
            }
            for future in concurrent.futures.as_completed(future_to_port):
                port = future_to_port[future]
                try:
                    if future.result():
                        open_ports.append(port)
                except Exception:
                    pass

        # Step 2: Parallel ADB connect
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(open_ports))
        ) as executor:
            futures = [
                executor.submit(self._connect_device, f"{self.host}:{port}")
                for port in open_ports
            ]
            concurrent.futures.wait(futures, timeout=3.0)

        # Step 3: Read connected devices list from ADB CLI
        connected_serials: List[str] = []
        try:
            result = subprocess.run(
                [self.adb_path, "devices"], capture_output=True, text=True, timeout=3.0
            )
            for line in result.stdout.splitlines():
                if "\tdevice" in line:
                    dev_serial = line.split("\t")[0]
                    if dev_serial.startswith(self.host) or dev_serial.startswith(
                        "127.0.0.1"
                    ):
                        if dev_serial not in connected_serials:
                            connected_serials.append(dev_serial)
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            print("[ADBManager] Error executing 'adb devices' query.")
            return []

        if not connected_serials:
            return []

        # Step 4: Extract kernel boot UUID for each connected port concurrently
        uuid_map: Dict[str, str] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(connected_serials))
        ) as executor:
            future_to_serial = {
                executor.submit(self._get_vm_boot_uuid, serial): serial
                for serial in connected_serials
            }
            for future in concurrent.futures.as_completed(future_to_serial):
                serial = future_to_serial[future]
                vm_uuid = future.result()
                if vm_uuid:
                    uuid_map[serial] = vm_uuid

        # Step 5: Group serials by VM boot UUID and pick exactly ONE canonical serial per VM
        grouped_by_uuid: Dict[str, List[str]] = {}
        unmapped_serials: List[str] = []

        for serial in connected_serials:
            if serial in uuid_map:
                vm_uuid = uuid_map[serial]
                grouped_by_uuid.setdefault(vm_uuid, []).append(serial)
            else:
                unmapped_serials.append(serial)

        unique_canonical_serials: List[str] = []

        def sort_port_key(s: str) -> int:
            try:
                return int(s.split(":")[-1])
            except ValueError:
                return 0

        # For each physical VM UUID, pick the primary port (lowest port number)
        for vm_uuid, serials in grouped_by_uuid.items():
            serials.sort(key=sort_port_key)
            canonical_port = serials[0]
            unique_canonical_serials.append(canonical_port)

        # Include unmapped serials only if no UUID was obtained
        for serial in unmapped_serials:
            if serial not in unique_canonical_serials:
                unique_canonical_serials.append(serial)

        unique_canonical_serials.sort(key=sort_port_key)
        return unique_canonical_serials

    def _connect_device(self, serial: str) -> None:
        """Issues an adb connect command for a target serial."""
        try:
            subprocess.run(
                [self.adb_path, "connect", serial],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1.5,
            )
        except Exception:
            pass

    def _is_port_open(self, host: str, port: int) -> bool:
        """Utility to check if a local TCP port is accepting connections."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.config.connection_timeout)
                return sock.connect_ex((host, port)) == 0
        except OSError:
            return False

    def capture_frame(self, device_serial: str) -> Optional[np.ndarray]:
        """Captures frame from ADB stdout stream using fast raw RGBA parsing with PNG fallback."""
        cmd_raw = [self.adb_path, "-s", device_serial, "exec-out", "screencap"]
        try:
            process = subprocess.Popen(
                cmd_raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            raw_data, _ = process.communicate(timeout=2.0)
            if process.returncode == 0 and len(raw_data) > 16:
                w = int.from_bytes(raw_data[0:4], byteorder="little")
                h = int.from_bytes(raw_data[4:8], byteorder="little")

                expected_size = w * h * 4
                header_size = 12 if len(raw_data) == 12 + expected_size else 16

                if len(raw_data) >= header_size + expected_size and w > 0 and h > 0:
                    pixel_bytes = raw_data[header_size : header_size + expected_size]
                    img = np.frombuffer(pixel_bytes, dtype=np.uint8).reshape((h, w, 4))
                    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        except Exception:
            pass

        cmd_png = [self.adb_path, "-s", device_serial, "exec-out", "screencap", "-p"]
        try:
            process = subprocess.Popen(
                cmd_png, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            raw_png, _ = process.communicate(timeout=2.0)
            if process.returncode != 0 or not raw_png:
                return None

            frame_array = np.frombuffer(raw_png, dtype=np.uint8)
            return cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        except Exception:
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
