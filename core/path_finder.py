import glob
import os
import shutil
import string
import subprocess
import winreg
from typing import Optional


class MuMuPathFinder:
    """Automated locator engine for MuMuPlayer installations and ADB binaries."""

    @classmethod
    def auto_detect_adb(cls) -> str:
        """Executes a 5-tier resolution pipeline to locate adb.exe automatically."""
        # Tier 0: Explicit environment variable check
        env_adb = os.environ.get("ADB_PATH")
        if env_adb and os.path.exists(env_adb):
            return env_adb

        # Tier 1: Running process detection
        adb_path = cls._find_from_running_processes()
        if adb_path:
            return adb_path

        # Tier 2: System PATH check
        system_adb = shutil.which("adb")
        if system_adb:
            return system_adb

        # Tier 3: Registry lookup
        adb_path = cls._find_from_registry()
        if adb_path:
            return adb_path

        # Tier 4: Comprehensive drive scan
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
        """Detects ADB path from active emulator process locations via PowerShell."""
        try:
            cmd = [
                "powershell",
                "-Command",
                "Get-Process -Name 'MuMuPlayer', 'NemuPlayer', 'MuMuVMMHeadless' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Path",
            ]
            output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
            for line in output.splitlines():
                line = line.strip()
                if line and os.path.exists(line):
                    mumu_dir = os.path.dirname(line)
                    # Check direct directory and common MuMu 12 subdirectories
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
            except OSError:
                continue
        return None

    @classmethod
    def _find_from_drive_scan(cls) -> Optional[str]:
        """Scans all active fixed Windows drives for standard MuMuPlayer ADB paths."""
        drives = [
            f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")
        ]
        scan_patterns = [
            r"Program Files*\NetEase\MuMuPlayer*\shell\adb.exe",
            r"Program Files*\NetEase\MuMuPlayer*\adb.exe",
            r"NetEase\MuMuPlayer*\shell\adb.exe",
            r"NetEase\MuMuPlayer*\adb.exe",
            r"MuMuPlayer*\shell\adb.exe",
            r"MuMu*\shell\adb.exe",
            r"Nemu*\vms\bin\adb.exe",
        ]

        for drive in drives:
            for pattern in scan_patterns:
                matches = glob.glob(os.path.join(drive, pattern))
                if matches and os.path.exists(matches[0]):
                    return matches[0]
        return None
