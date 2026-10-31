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
        """Executes a 5-tier resolution pipeline to locate adb.exe automatically."""
        env_adb = os.environ.get("ADB_PATH")
        if env_adb and os.path.exists(env_adb):
            return env_adb

        adb_path = cls._find_from_running_processes()
        if adb_path:
            return adb_path

        system_adb = shutil.which("adb")
        if system_adb:
            return system_adb

        adb_path = cls._find_from_registry()
        if adb_path:
            return adb_path

        adb_path = cls._find_from_drive_scan()
        if adb_path:
            return adb_path

        raise FileNotFoundError(
            "Automatic detection could not locate 'adb.exe'. "
            "Please ensure MuMuPlayer is installed, set the 'ADB_PATH' environment variable, "
            "or specify 'adb_path' directly in config.py / ADBManager initialization."
        )

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
