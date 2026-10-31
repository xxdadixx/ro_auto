import glob
import os
import shutil
import string
import subprocess
import sys
from typing import Optional

# Platform-safe Windows Registry import guard
if sys.platform == "win32":
    import winreg
else:
    winreg = None


class MuMuPathFinder:
    """Automated locator engine for MuMuPlayer installations and ADB binaries."""

    @classmethod
    def auto_detect_adb(cls) -> str:
        """Locate ADB using environment variables, PATH, and MuMu installation paths."""
        env_adb = os.environ.get("ADB_PATH")

        if env_adb:
            candidate = os.path.abspath(
                os.path.expandvars(
                    os.path.expanduser(env_adb.strip())
                )
            )

            if os.path.isfile(candidate):
                return candidate

        candidate = cls._find_from_running_processes()

        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)

        candidate = shutil.which("adb")

        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)

        candidate = cls._find_from_registry()

        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)

        candidate = cls._find_from_drive_scan()

        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)

        raise FileNotFoundError(
            "Could not locate ADB. Set ADB_PATH, add adb to PATH, "
            "or configure ADBConfig.custom_adb_path."
        )

    @classmethod
    def _find_from_drive_scan(cls) -> Optional[str]:
        """Search conventional MuMu installation folders on Windows drives."""
        if sys.platform != "win32":
            return None

        program_dirs = (
            "Program Files",
            "Program Files (x86)",
        )

        product_dirs = (
            "MuMuPlayer*",
            "MuMuPlayerGlobal*",
            "MuMuPlayer-12.0*",
        )

        relative_patterns = []

        for program_dir in program_dirs:
            for product_dir in product_dirs:
                relative_patterns.extend(
                    [
                        os.path.join(
                            program_dir,
                            "Netease",
                            product_dir,
                            "nx_device",
                            "*",
                            "shell",
                            "adb.exe",
                        ),
                        os.path.join(
                            program_dir,
                            "Netease",
                            product_dir,
                            "shell",
                            "adb.exe",
                        ),
                    ]
                )

        for drive_letter in string.ascii_uppercase:
            root = f"{drive_letter}:\\"

            if not os.path.isdir(root):
                continue

            for pattern in relative_patterns:
                search_pattern = os.path.join(root, pattern)

                for candidate in sorted(glob.glob(search_pattern)):
                    if os.path.isfile(candidate):
                        return os.path.normpath(candidate)

        return None

    @classmethod
    def _find_from_running_processes(cls) -> Optional[str]:
        """Detects ADB path from active emulator process locations via platform commands."""
        if sys.platform != "win32":
            return None

        try:
            cmd = [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-Process -Name 'MuMuPlayer', 'NemuPlayer', 'MuMuVMMHeadless' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Path",
            ]
            output = subprocess.check_output(
                cmd, text=True, stderr=subprocess.DEVNULL, timeout=3.0
            )
            for line in output.splitlines():
                line = line.strip()
                if line and os.path.exists(line):
                    mumu_dir = os.path.dirname(line)
                    for candidate in [
                        os.path.join(mumu_dir, "adb.exe"),
                        os.path.join(mumu_dir, "shell", "adb.exe"),
                        os.path.join(mumu_dir, "..", "shell", "adb.exe"),
                    ]:
                        resolved = os.path.normpath(candidate)
                        if os.path.exists(resolved):
                            return resolved
        except Exception:
            pass
        return None

    @classmethod
    def _find_from_registry(cls) -> Optional[str]:
        if winreg is None or sys.platform != "win32":
            return None

        registry_keys = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\NetEase\MuMuPlayer"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\NetEase\MuMuPlayerGlobal"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\NetEase\MuMuPlayer-12.0"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\NetEase\MuMuPlayer"),
            (
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\WOW6432Node\NetEase\MuMuPlayerGlobal",
            ),
            (
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\WOW6432Node\NetEase\MuMuPlayer-12.0",
            ),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\NetEase\MuMuPlayer"),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\NetEase\MuMuPlayer-12.0"),
        ]

        for hive, key_path in registry_keys:
            try:
                with winreg.OpenKey(hive, key_path) as key:
                    install_dir, _ = winreg.QueryValueEx(key, "InstallDir")
                    candidates = [
                        os.path.join(install_dir, "shell", "adb.exe"),
                        os.path.join(install_dir, "adb.exe"),
                        os.path.join(install_dir, "nx_player", "adb.exe"),
                    ]
                    for candidate in candidates:
                        if os.path.exists(candidate):
                            return candidate
            except (OSError, AttributeError):
                continue
        return None
