import time
from enum import Enum, auto
from typing import Optional
import numpy as np

from core.adb_manager import ADBManager
from vision.vision import VisionEngine


class BotState(Enum):
    INIT = auto()
    CHECK_QUEST = auto()
    AUTO_BATTLE = auto()
    HANDLE_DIALOG = auto()
    RECOVER_HEALTH = auto()


class InstanceWorker:
    """Isolated state machine worker handling a single MuMuPlayer instance."""

    def __init__(self, device_serial: str, adb_mgr: ADBManager, vision: VisionEngine) -> None:
        self.device_serial = device_serial
        self.adb = adb_mgr
        self.vision = vision
        self.state = BotState.INIT
        self.is_running = False

    def run(self) -> None:
        """Main execution loop for the emulator instance."""
        self.is_running = True
        print(f"[{self.device_serial}] Worker loop started.")

        while self.is_running:
            frame = self.adb.capture_frame(self.device_serial)
            if frame is None:
                time.sleep(1.0)
                continue

            self._process_state(frame)
            time.sleep(0.5)  # Loop control delay

    def stop(self) -> None:
        self.is_running = False

    def _process_state(self, frame: np.ndarray) -> None:
        """Executes state-specific logic and transition rules."""
        match self.state:
            case BotState.INIT:
                print(f"[{self.device_serial}] Initializing instance state...")
                self.state = BotState.CHECK_QUEST

            case BotState.CHECK_QUEST:
                # Quest UI ROI: top-left area
                quest_coord = self.vision.match_template(
                    frame, "templates/quest_button.png", threshold=0.75
                )
                if quest_coord:
                    print(f"[{self.device_serial}] Quest found. Tapping quest log.")
                    self.adb.tap(self.device_serial, quest_coord[0], quest_coord[1])
                    self.state = BotState.HANDLE_DIALOG
                else:
                    # OCR Fallback check for Thai quest text in quest box ROI (x=10, y=200, w=200, h=150)
                    texts = self.vision.read_thai_text(frame, roi=(10, 200, 200, 150))
                    if any("เควส" in text or "ส่งเควส" for text in texts):
                        print(f"[{self.device_serial}] Thai Quest text detected via OCR.")
                        self.adb.tap(self.device_serial, 100, 250)
                    else:
                        self.state = BotState.AUTO_BATTLE

            case BotState.HANDLE_DIALOG:
                confirm_coord = self.vision.match_template(
                    frame, "templates/confirm_dialog.png", threshold=0.8
                )
                if confirm_coord:
                    print(f"[{self.device_serial}] Confirming dialog window.")
                    self.adb.tap(self.device_serial, confirm_coord[0], confirm_coord[1])
                self.state = BotState.CHECK_QUEST

            case BotState.AUTO_BATTLE:
                # Keep active auto-battle state
                self.state = BotState.CHECK_QUEST