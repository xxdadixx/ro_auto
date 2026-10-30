import time
from enum import Enum, auto
from typing import Optional
import numpy as np

from config import BotConfig
from core.adb_manager import ADBManager
from gui.signals import signals
from vision.vision import VisionEngine


class BotState(Enum):
    INIT = auto()
    CHECK_QUEST = auto()
    AUTO_BATTLE = auto()
    HANDLE_DIALOG = auto()
    RECOVER_HEALTH = auto()


class InstanceWorker:
    """Isolated state machine worker handling a single MuMuPlayer instance."""

    def __init__(
        self,
        device_serial: str,
        adb_mgr: ADBManager,
        vision: VisionEngine,
        config: BotConfig = BotConfig(),
    ) -> None:
        self.device_serial = device_serial
        self.adb = adb_mgr
        self.vision = vision
        self.config = config
        self.state = BotState.INIT
        self.is_running = False

    def run(self) -> None:
        """Main execution loop sharing captured frames directly with the UI."""
        self.is_running = True
        signals.log_emitted.emit(
            "INFO", "STATE", "Worker loop started.", self.device_serial
        )
        signals.state_changed.emit(self.device_serial, self.state.name)

        while self.is_running:
            try:
                frame = self.adb.capture_frame(self.device_serial)
                if frame is None:
                    time.sleep(1.0)
                    continue

                # Pass captured frame directly to UI without spawning extra adb.exe processes
                signals.frame_ready.emit(self.device_serial, frame)

                self._process_state(frame)
            except Exception as err:
                signals.log_emitted.emit(
                    "ERROR", "WORKER", f"Error in loop: {err}", self.device_serial
                )

            time.sleep(self.config.loop_delay_sec)

    def stop(self) -> None:
        """Signals the worker loop to terminate gracefully."""
        self.is_running = False
        signals.state_changed.emit(self.device_serial, "STOPPED")

    def _process_state(self, frame: np.ndarray) -> None:
        """Executes state-specific logic and transition rules."""
        match self.state:
            case BotState.INIT:
                self.state = BotState.CHECK_QUEST
                signals.state_changed.emit(self.device_serial, self.state.name)

            case BotState.CHECK_QUEST:
                quest_coord = self.vision.match_template(
                    frame, "templates/quest_button.png", threshold=0.75
                )
                if quest_coord:
                    signals.log_emitted.emit(
                        "INFO",
                        "VISION",
                        f"Quest found at {quest_coord}.",
                        self.device_serial,
                    )
                    self.adb.tap(self.device_serial, quest_coord[0], quest_coord[1])
                    self.state = BotState.HANDLE_DIALOG
                    signals.state_changed.emit(self.device_serial, self.state.name)
                else:
                    texts = self.vision.read_thai_text(frame, roi=self.config.quest_roi)
                    if any("เควส" in text or "ส่งเควส" for text in texts):
                        signals.log_emitted.emit(
                            "INFO",
                            "OCR",
                            "Thai Quest text detected.",
                            self.device_serial,
                        )
                        self.adb.tap(self.device_serial, 100, 250)
                    else:
                        self.state = BotState.AUTO_BATTLE
                        signals.state_changed.emit(self.device_serial, self.state.name)

            case BotState.HANDLE_DIALOG:
                confirm_coord = self.vision.match_template(
                    frame, "templates/confirm_dialog.png", threshold=0.8
                )
                if confirm_coord:
                    signals.log_emitted.emit(
                        "INFO",
                        "VISION",
                        f"Dialog confirmed at {confirm_coord}.",
                        self.device_serial,
                    )
                    self.adb.tap(self.device_serial, confirm_coord[0], confirm_coord[1])
                self.state = BotState.CHECK_QUEST
                signals.state_changed.emit(self.device_serial, self.state.name)

            case BotState.AUTO_BATTLE:
                self.state = BotState.CHECK_QUEST
                signals.state_changed.emit(self.device_serial, self.state.name)
