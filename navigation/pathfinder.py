import glob
import os
import shutil
import subprocess
import winreg
from typing import Optional


class MuMuPathFinder:
    """Automated locator engine for MuMuPlayer installations and ADB binaries."""

    @classmethod
    def auto_detect_adb(cls) -> str:
        """Executes a 4-tier resolution pipeline to find adb.exe automatically."""
        adb_path = cls._find_from_running_processes()
        if adb_path:
            return adb_path

        if shutil.which("adb"):
            return "adb"

        adb_path = cls._find_from_registry()
        if adb_path:
            return adb_path

        adb_path = cls._find_from_drive_scan()
        if adb_path:
            return adb_path

        raise FileNotFoundError(
            "Automatic detection could not locate 'adb.exe'. Ensure MuMuPlayer is installed."
        )

    @classmethod
    def _find_from_running_processes(cls) -> Optional[str]:
        try:
            cmd = [
                "wmic",
                "process",
                "where",
                "name='MuMuPlayer.exe' or name='NemuPlayer.exe'",
                "get",
                "ExecutablePath",
            ]
            output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
            for line in output.splitlines():
                line = line.strip()
                if line and os.path.exists(line):
                    mumu_dir = os.path.dirname(line)
                    adb_candidate = os.path.join(mumu_dir, "adb.exe")
                    if os.path.exists(adb_candidate):
                        return adb_candidate
        except Exception:
            pass
        return None

    @classmethod
    def _find_from_registry(cls) -> Optional[str]:
        registry_keys = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\NetEase\MuMuPlayer"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\NetEase\MuMuPlayerGlobal"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\NetEase\MuMuPlayer"),
            (
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\WOW6432Node\NetEase\MuMuPlayerGlobal",
            ),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\NetEase\MuMuPlayer"),
        ]

        for hive, key_path in registry_keys:
            try:
                with winreg.OpenKey(hive, key_path) as key:
                    install_dir, _ = winreg.QueryValueEx(key, "InstallDir")
                    candidate = os.path.join(install_dir, "shell", "adb.exe")
                    if os.path.exists(candidate):
                        return candidate
            except OSError:
                continue
        return None

    @classmethod
    def _find_from_drive_scan(cls) -> Optional[str]:
        drives = [f"{d}:\\" for d in ["C", "D", "E", "F"] if os.path.exists(f"{d}:\\")]
        scan_patterns = [
            r"Program Files*\NetEase\MuMuPlayer*\shell\adb.exe",
            r"NetEase\MuMuPlayer*\shell\adb.exe",
            r"MuMuPlayer*\shell\adb.exe",
            r"Nemu*\vms\bin\adb.exe",
        ]

        for drive in drives:
            for pattern in scan_patterns:
                matches = glob.glob(os.path.join(drive, pattern))
                if matches and os.path.exists(matches[0]):
                    return matches[0]
        return None
