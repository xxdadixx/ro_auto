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
        """Discover connected local emulators and deduplicate aliases safely.

        Existing ADB connections are checked before port probing, so a device
        already connected by MuMu is not missed merely because its port is not
        in the candidate list. ADB command failures are raised instead of being
        converted into a misleading empty device list.
        """

        def read_connected_serials() -> List[str]:
            result = subprocess.run(
                [self.adb_path, "devices", "-l"],
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )

            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(
                    "ADB devices query failed" + (f": {detail}" if detail else "")
                )

            discovered: List[str] = []
            accepted_prefixes = (
                f"{self.host}:",
                "127.",
                "localhost:",
                "emulator-",
            )

            for line in result.stdout.splitlines():
                fields = line.split()
                if len(fields) < 2 or fields[1] != "device":
                    continue

                serial = fields[0].strip()
                if serial.startswith(accepted_prefixes) and serial not in discovered:
                    discovered.append(serial)

            return discovered

        # Ensure the ADB server is available before querying its device table.
        start_result = subprocess.run(
            [self.adb_path, "start-server"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if start_result.returncode != 0:
            detail = start_result.stderr.strip() or start_result.stdout.strip()
            raise RuntimeError(
                "Could not start the ADB server" + (f": {detail}" if detail else "")
            )

        # Capture devices already connected by MuMu or by a previous scan.
        connected_serials = read_connected_serials()

        # Probe likely MuMu ports, then ask ADB to connect to ports that answer.
        candidate_ports = self._generate_candidate_ports()
        open_ports: List[int] = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(32, max(1, len(candidate_ports)))
        ) as executor:
            future_to_port = {
                executor.submit(self._is_port_open, self.host, port): port
                for port in candidate_ports
            }

            for future in concurrent.futures.as_completed(future_to_port):
                try:
                    if future.result():
                        open_ports.append(future_to_port[future])
                except Exception:
                    # A single failed socket probe must not abort discovery.
                    continue

        if open_ports:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(16, len(open_ports))
            ) as executor:
                futures = [
                    executor.submit(
                        self._connect_device,
                        f"{self.host}:{port}",
                    )
                    for port in sorted(open_ports)
                ]
                concurrent.futures.wait(futures)

        # Re-query after connection attempts and merge with the initial result.
        for serial in read_connected_serials():
            if serial not in connected_serials:
                connected_serials.append(serial)

        if not connected_serials:
            return []

        # Resolve duplicate ADB aliases by Android boot UUID / MAC only.
        # Hostname is not a safe identity because cloned VMs may share it.
        uuid_map: Dict[str, str] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(16, len(connected_serials))
        ) as executor:
            future_to_serial = {
                executor.submit(self._get_vm_boot_uuid, serial): serial
                for serial in connected_serials
            }

            for future in concurrent.futures.as_completed(future_to_serial):
                serial = future_to_serial[future]
                try:
                    vm_uuid = future.result()
                except Exception:
                    vm_uuid = None

                if vm_uuid and not vm_uuid.startswith("host_"):
                    uuid_map[serial] = vm_uuid

        grouped_by_uuid: Dict[str, List[str]] = {}
        unmapped_serials: List[str] = []

        for serial in connected_serials:
            vm_uuid = uuid_map.get(serial)
            if vm_uuid:
                grouped_by_uuid.setdefault(vm_uuid, []).append(serial)
            else:
                unmapped_serials.append(serial)

        def sort_port_key(serial: str) -> int:
            try:
                return int(serial.rsplit(":", 1)[-1])
            except ValueError:
                return 0

        canonical_serials: List[str] = []

        for serials in grouped_by_uuid.values():
            canonical_serials.append(min(serials, key=sort_port_key))

        canonical_serials.extend(unmapped_serials)
        return sorted(set(canonical_serials), key=sort_port_key)

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
        """Captures frame from ADB stdout stream using fast raw RGBA parsing with process lifetime safety and PNG fallback."""
        cmd_raw = [self.adb_path, "-s", device_serial, "exec-out", "screencap"]
        try:
            process = subprocess.Popen(
                cmd_raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            try:
                raw_data, _ = process.communicate(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raw_data = b""

            if process.returncode == 0 and len(raw_data) > 16:
                w = int.from_bytes(raw_data[0:4], byteorder="little")
                h = int.from_bytes(raw_data[4:8], byteorder="little")

                if w > 0 and h > 0:
                    expected_pixel_bytes = w * h * 4
                    header_size = len(raw_data) - expected_pixel_bytes
                    if (
                        header_size in (12, 16)
                        and len(raw_data) >= header_size + expected_pixel_bytes
                    ):
                        pixel_bytes = raw_data[
                            header_size : header_size + expected_pixel_bytes
                        ]
                        img = np.frombuffer(pixel_bytes, dtype=np.uint8).reshape(
                            (h, w, 4)
                        )
                        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        except Exception:
            pass

        cmd_png = [self.adb_path, "-s", device_serial, "exec-out", "screencap", "-p"]
        try:
            process = subprocess.Popen(
                cmd_png, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            try:
                raw_png, _ = process.communicate(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                return None

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
