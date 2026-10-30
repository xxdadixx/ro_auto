# main.py
import concurrent.futures
import signal
import sys
from typing import List

from core.adb_manager import ADBManager
from vision.vision import VisionEngine
from engine.instance_worker import InstanceWorker


def main() -> None:
    print("=== Ragnarok: The New World Multi-Bot Launcher ===")

    # Do not pass a hardcoded adb_path; let ADBManager scan all drives automatically
    adb_mgr = ADBManager()
    vision = VisionEngine(use_gpu=False)

    active_devices: List[str] = adb_mgr.scan_mumu_instances()
    if not active_devices:
        print(
            "No active MuMuPlayer ADB instances detected. Ensure MuMuPlayer is running with ADB Debugging enabled."
        )
        return

    print(f"Discovered {len(active_devices)} instance(s): {active_devices}")

    workers: List[InstanceWorker] = [
        InstanceWorker(serial, adb_mgr, vision) for serial in active_devices
    ]

    def shutdown_handler(sig, frame):
        print("\nShutting down workers...")
        for w in workers:
            w.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(workers), thread_name_prefix="InstanceWorker"
    ) as executor:
        futures = [executor.submit(worker.run) for worker in workers]
        concurrent.futures.wait(futures)


if __name__ == "__main__":
    main()
