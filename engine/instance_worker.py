import os
import threading
import time
from enum import Enum, auto
from typing import List, Optional, Tuple
import cv2
import numpy as np

from config import BotConfig
from core.adb_manager import ADBManager
from gui.signals import signals
from vision.vision import VisionEngine


class BotState(Enum):
    INIT = auto()
    CHECK_QUEST = auto()
    OPEN_QUEST_GUI = auto()
    SELECT_MAIN_QUEST = auto()
    CLICK_GO_BUTTON = auto()
    WAIT_QUEST_PROGRESS = auto()
    AUTO_BATTLE = auto()
    HANDLE_DIALOG = auto()
    SKIP_CUTSCENE = auto()
    RECOVER_HEALTH = auto()


def get_abs_roi(
    rel_roi: Tuple[float, float, float, float], frame_w: int, frame_h: int
) -> Tuple[int, int, int, int]:
    """Converts relative percentage ROI (x_ratio, y_ratio, w_ratio, h_ratio) to absolute frame pixel coordinates."""
    xr, yr, wr, hr = rel_roi
    rx = max(0, min(int(xr * frame_w), frame_w - 1))
    ry = max(0, min(int(yr * frame_h), frame_h - 1))
    rw = max(1, min(int(wr * frame_w), frame_w - rx))
    rh = max(1, min(int(hr * frame_h), frame_h - ry))
    return (rx, ry, rw, rh)


def get_abs_point(
    rel_pt: Tuple[float, float], frame_w: int, frame_h: int
) -> Tuple[int, int]:
    """Converts relative percentage point (x_ratio, y_ratio) to absolute frame pixel coordinates."""
    px, py = rel_pt
    return (int(px * frame_w), int(py * frame_h))


def draw_tap_indicator(
    frame: np.ndarray,
    x: int,
    y: int,
    label: str = "TAP",
    color: Tuple[int, int, int] = (0, 242, 254),
) -> np.ndarray:
    """Draws a high-visibility target crosshair and coordinate badge overlay on the frame matrix."""
    if frame is None or frame.size == 0:
        return frame

    fh, fw = frame.shape[:2]
    if not (0 <= x < fw and 0 <= y < fh):
        return frame

    frame_copy = frame.copy()

    # Concentric circles
    cv2.circle(frame_copy, (x, y), 18, color, 2, cv2.LINE_AA)
    cv2.circle(frame_copy, (x, y), 6, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(frame_copy, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)

    # Reticle crosshairs
    cv2.line(frame_copy, (x - 25, y), (x - 8, y), color, 2, cv2.LINE_AA)
    cv2.line(frame_copy, (x + 8, y), (x + 25, y), color, 2, cv2.LINE_AA)
    cv2.line(frame_copy, (x, y - 25), (x, y - 8), color, 2, cv2.LINE_AA)
    cv2.line(frame_copy, (x, y + 8), (x, y + 25), color, 2, cv2.LINE_AA)

    # Label box
    badge_text = f" 🎯 {label} [{x}, {y}] "
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(
        badge_text, font, font_scale, thickness
    )

    label_x = max(0, min(x + 20, fw - text_w - 5))
    label_y = max(text_h + 5, min(y - 10, fh - 5))

    cv2.rectangle(
        frame_copy,
        (label_x, label_y - text_h - 4),
        (label_x + text_w, label_y + baseline),
        (15, 21, 37),
        -1,
    )
    cv2.rectangle(
        frame_copy,
        (label_x, label_y - text_h - 4),
        (label_x + text_w, label_y + baseline),
        color,
        1,
    )
    cv2.putText(
        frame_copy,
        badge_text,
        (label_x, label_y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    return frame_copy


def draw_scan_roi(
    frame: np.ndarray,
    roi: Tuple[int, int, int, int],
    label: str,
    color: Tuple[int, int, int] = (255, 191, 0),
) -> np.ndarray:
    """Draws scanning zone bounding boxes onto live instance frames."""
    if frame is None or frame.size == 0 or not roi:
        return frame

    x, y, w, h = roi
    fh, fw = frame.shape[:2]
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > fw or y + h > fh:
        return frame

    frame_copy = frame.copy()
    cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f" DYNAMIC SCAN: {label} "
    cv2.putText(
        frame_copy,
        text,
        (x, max(y - 4, 12)),
        font,
        0.35,
        color,
        1,
        cv2.LINE_AA,
    )
    return frame_copy


class InstanceWorker:
    """Thread-safe state machine worker using dynamic OCR text anchors and resolution-adaptive multi-scale vision."""

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

        self._lock = threading.RLock()
        self._state = BotState.INIT
        self._is_running = False

        self._stop_event = threading.Event()
        self._stop_event.set()
        self._start_reserved = False

        self._active_target_template: Optional[np.ndarray] = None
        self._active_target_roi: Optional[Tuple[int, int, int, int]] = None

        self._one_shot_target: bool = True
        self._last_target_click_time: float = 0.0
        self._last_heartbeat_time: float = 0.0
        self._last_cutscene_ocr_time: float = 0.0

        # Thread-safe active tap markers: list of (x, y, label, timestamp)
        self._active_tap_markers: List[Tuple[int, int, str, float]] = []

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._is_running

    @is_running.setter
    def is_running(self, value: bool) -> None:
        with self._lock:
            self._is_running = value

    @property
    def state(self) -> BotState:
        with self._lock:
            return self._state

    @state.setter
    def state(self, value: BotState) -> None:
        with self._lock:
            self._state = value

    @property
    def active_target_template(self) -> Optional[np.ndarray]:
        with self._lock:
            return (
                self._active_target_template.copy()
                if self._active_target_template is not None
                else None
            )

    @active_target_template.setter
    def active_target_template(self, template: Optional[np.ndarray]) -> None:
        with self._lock:
            self._active_target_template = (
                template.copy() if template is not None else None
            )

    @property
    def active_target_roi(self) -> Optional[Tuple[int, int, int, int]]:
        with self._lock:
            return self._active_target_roi

    @active_target_roi.setter
    def active_target_roi(self, roi: Optional[Tuple[int, int, int, int]]) -> None:
        with self._lock:
            self._active_target_roi = roi

    @property
    def one_shot_target(self) -> bool:
        with self._lock:
            return self._one_shot_target

    @one_shot_target.setter
    def one_shot_target(self, value: bool) -> None:
        with self._lock:
            self._one_shot_target = value

    def register_tap(self, x: int, y: int, label: str = "TAP") -> None:
        """Executes ADB tap action and registers a 1.5-second visual overlay marker."""
        self.adb.tap(self.device_serial, x, y)
        with self._lock:
            self._active_tap_markers.append((x, y, label, time.time()))

    def _apply_visual_overlays(self, frame: np.ndarray) -> np.ndarray:
        """Applies active scanning zone ROIs and tap indicators to live frames."""
        annotated_frame = frame.copy()
        fh, fw = frame.shape[:2]

        hud_roi = get_abs_roi(self.config.quest_hud_rel_roi, fw, fh)
        annotated_frame = draw_scan_roi(
            annotated_frame, hud_roi, "QUEST HUD ANCHOR", (255, 191, 0)
        )

        cutscene_roi = get_abs_roi(self.config.cutscene_rel_roi, fw, fh)
        annotated_frame = draw_scan_roi(
            annotated_frame, cutscene_roi, "CUTSCENE ANCHOR", (0, 255, 255)
        )

        with self._lock:
            target_roi = self._active_target_roi
        if target_roi:
            annotated_frame = draw_scan_roi(
                annotated_frame, target_roi, "DYNAMIC TARGET", (0, 242, 254)
            )

        now = time.time()
        with self._lock:
            self._active_tap_markers = [
                m for m in self._active_tap_markers if (now - m[3]) < 1.5
            ]
            current_markers = list(self._active_tap_markers)

        for x, y, label, _ in current_markers:
            annotated_frame = draw_tap_indicator(annotated_frame, x, y, label=label)

        return annotated_frame

    def prepare_start(self) -> bool:
        """Reserve a single worker execution before executor submission."""
        with self._lock:
            if self._is_running or self._start_reserved:
                return False

            self._stop_event.clear()
            self._start_reserved = True

            return True

    def cancel_start(self) -> None:
        """Cancel a pending start when executor submission fails."""
        with self._lock:
            if not self._is_running:
                self._start_reserved = False
                self._stop_event.set()

    def run(self) -> None:
        """Execute one guarded worker loop with cooperative cancellation."""
        with self._lock:
            if (
                self._is_running
                or not self._start_reserved
                or self._stop_event.is_set()
            ):
                self._start_reserved = False
                return

            self._start_reserved = False
            self._is_running = True
            self._state = BotState.CHECK_QUEST

            current_state = self._state

        try:
            signals.log_emitted.emit(
                "INFO",
                "STATE",
                "Dynamic Worker execution loop started.",
                self.device_serial,
            )

            signals.state_changed.emit(
                self.device_serial,
                current_state.name,
            )

            while not self._stop_event.is_set():
                try:
                    frame = self.adb.capture_frame(self.device_serial)

                    if frame is None:
                        self._stop_event.wait(1.0)
                        continue

                    self._process_state(frame)

                    if self._stop_event.is_set():
                        break

                    annotated_frame = self._apply_visual_overlays(frame)

                    signals.frame_ready.emit(
                        self.device_serial,
                        annotated_frame,
                    )

                except Exception as err:
                    signals.log_emitted.emit(
                        "ERROR",
                        "WORKER",
                        f"Error in loop: {err}",
                        self.device_serial,
                    )

                self._stop_event.wait(max(0.0, float(self.config.loop_delay_sec)))

        finally:
            with self._lock:
                self._is_running = False
                self._start_reserved = False
                self._state = BotState.INIT

            signals.state_changed.emit(
                self.device_serial,
                "STOPPED",
            )

    def stop(self) -> None:
        """Request graceful shutdown or cancel a worker that has not started."""
        with self._lock:
            self._stop_event.set()
            self._start_reserved = False

            is_running = self._is_running

        if not is_running:
            signals.state_changed.emit(
                self.device_serial,
                "STOPPED",
            )

    def _process_state(self, frame: np.ndarray) -> None:
        """Executes dynamic text anchor discovery and state progression without hardcoded pixel reliance."""
        current_time = time.time()
        fh, fw = frame.shape[:2]

        with self._lock:
            target_tmpl = (
                self._active_target_template.copy()
                if self._active_target_template is not None
                else None
            )
            target_roi = self._active_target_roi
            is_one_shot = self._one_shot_target
            current_state = self._state

        # Diagnostic Heartbeat Telemetry (Every 3 Seconds)
        if (current_time - self._last_heartbeat_time) > 3.0:
            self._last_heartbeat_time = current_time
            signals.log_emitted.emit(
                "INFO",
                "LOOP",
                f"Active State: {current_state.name} | Viewport: {fw}x{fh}px",
                self.device_serial,
            )

        cutscene_abs_roi = get_abs_roi(self.config.cutscene_rel_roi, fw, fh)
        hud_abs_roi = get_abs_roi(self.config.quest_hud_rel_roi, fw, fh)
        go_gui_abs_roi = get_abs_roi(self.config.quest_gui_go_rel_roi, fw, fh)

        # Step 1: Preemptive Structural Cutscene Detection
        if self.vision.detect_cutscene_skip_button(frame, roi=cutscene_abs_roi):
            skip_x, skip_y = get_abs_point(self.config.skip_button_rel_point, fw, fh)
            signals.log_emitted.emit(
                "INFO",
                "ACTION",
                f"🎬 Cutscene Skip Button verified via structural recognition! Tapping at [{skip_x}, {skip_y}].",
                self.device_serial,
            )
            self.register_tap(skip_x, skip_y, label="SKIP CUTSCENE")

            with self._lock:
                self._state = BotState.SKIP_CUTSCENE
                current_state = self._state

            signals.state_changed.emit(self.device_serial, current_state.name)
            time.sleep(0.8)
            return

        # Step 2: Dynamic User Crop Target Match (Scale-Invariant)
        if (
            target_tmpl is not None
            and target_tmpl.size > 0
            and (current_time - self._last_target_click_time) > 1.0
        ):
            click_coord = self.vision.match_template_multiscale(
                frame,
                target_tmpl,
                threshold=0.70,
                roi=target_roi,
            )
            if click_coord:
                signals.log_emitted.emit(
                    "INFO",
                    "ACTION",
                    f"🎯 Dynamic crop target matched at {click_coord}. Executing tap!",
                    self.device_serial,
                )
                self.register_tap(
                    click_coord[0], click_coord[1], label="DYNAMIC TARGET"
                )
                self._last_target_click_time = current_time

                with self._lock:
                    if is_one_shot:
                        self._active_target_template = None
                    self._state = BotState.CHECK_QUEST
                    current_state = self._state

                signals.state_changed.emit(self.device_serial, current_state.name)
                time.sleep(0.3)

        # Step 3: Fully Dynamic Quest GUI Automation
        match current_state:
            case BotState.INIT | BotState.SKIP_CUTSCENE:
                self.state = BotState.CHECK_QUEST
                signals.state_changed.emit(self.device_serial, self.state.name)

            case BotState.CHECK_QUEST:
                # Dynamic OCR Text Anchor Search across Left HUD
                quest_keywords = [
                    "การเดินทาง",
                    "งานฉลอง",
                    "คุยกับ",
                    "เควส",
                    "Senior",
                    "Knight",
                    "Job",
                    "เป้าหมาย",
                ]
                text_anchor = self.vision.find_text_anchor(
                    frame, keywords=quest_keywords, roi=hud_abs_roi
                )

                if text_anchor:
                    (tap_x, tap_y), matched_text = text_anchor
                    signals.log_emitted.emit(
                        "INFO",
                        "VISION",
                        f"✨ Dynamic OCR Text Anchor matched '{matched_text}' at [{tap_x}, {tap_y}]. Opening Quest GUI.",
                        self.device_serial,
                    )
                    self.register_tap(tap_x, tap_y, label=f"QUEST: {matched_text[:10]}")
                    self.state = BotState.OPEN_QUEST_GUI
                    signals.state_changed.emit(self.device_serial, self.state.name)
                    time.sleep(0.8)
                else:
                    signals.log_emitted.emit(
                        "WARN",
                        "VISION",
                        "⏳ Searching for dynamic quest text anchors on HUD...",
                        self.device_serial,
                    )

            case BotState.OPEN_QUEST_GUI | BotState.SELECT_MAIN_QUEST:
                # Select top main quest item in GUI menu
                top_x, top_y = get_abs_point(
                    self.config.top_main_quest_rel_point, fw, fh
                )
                signals.log_emitted.emit(
                    "INFO",
                    "ACTION",
                    f"📌 Selecting Main Quest in GUI menu at [{top_x}, {top_y}].",
                    self.device_serial,
                )
                self.register_tap(top_x, top_y, label="SELECT MAIN QUEST")
                self.state = BotState.CLICK_GO_BUTTON
                signals.state_changed.emit(self.device_serial, self.state.name)
                time.sleep(0.4)

            case BotState.CLICK_GO_BUTTON:
                # Dynamic OCR search for yellow "ไป" button in bottom-right GUI region
                go_anchor = self.vision.find_text_anchor(
                    frame, keywords=["ไป", "Go"], roi=go_gui_abs_roi
                )

                if go_anchor:
                    (go_x, go_y), matched_text = go_anchor
                    signals.log_emitted.emit(
                        "INFO",
                        "VISION",
                        f"🟡 Dynamic 'ไป' Go button text anchor verified at [{go_x}, {go_y}]. Executing tap.",
                        self.device_serial,
                    )
                else:
                    go_x, go_y = get_abs_point(self.config.go_button_rel_point, fw, fh)
                    signals.log_emitted.emit(
                        "INFO",
                        "ACTION",
                        f"🟡 Tapping 'ไป' Go button location at [{go_x}, {go_y}].",
                        self.device_serial,
                    )

                self.register_tap(go_x, go_y, label="GO MAIN QUEST")
                self.state = BotState.WAIT_QUEST_PROGRESS
                signals.state_changed.emit(self.device_serial, self.state.name)
                time.sleep(1.0)

            case BotState.WAIT_QUEST_PROGRESS:
                self.state = BotState.CHECK_QUEST
                signals.state_changed.emit(self.device_serial, self.state.name)

            case BotState.HANDLE_DIALOG:
                self.state = BotState.CHECK_QUEST
                signals.state_changed.emit(self.device_serial, self.state.name)

            case BotState.AUTO_BATTLE:
                self.state = BotState.CHECK_QUEST
                signals.state_changed.emit(self.device_serial, self.state.name)
