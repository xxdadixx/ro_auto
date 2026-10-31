import concurrent.futures
import sys
import time
import html
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, QSize, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QRubberBand,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import ADBConfig, BotConfig, VisionConfig
from core.adb_manager import ADBManager
from engine.instance_worker import BotState, InstanceWorker
from gui.signals import signals
from gui.styles import LiquidGlassTheme
from vision.vision import VisionEngine


def cv_frame_to_qpixmap(
    frame: np.ndarray, width: int = 320, height: int = 180
) -> QPixmap:
    """Converts OpenCV BGR image matrix to a scaled Qt QPixmap with memory safety guarantee."""
    if frame is None or frame.size == 0:
        return QPixmap()

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w

    q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

    pixmap = QPixmap.fromImage(q_img)
    if width > 0 and height > 0:
        return pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.FastTransformation)
    return pixmap


class ADBScanWorker(QThread):
    """Background worker for discovering ADB instances."""

    finished_scan = Signal(list)
    failed_scan = Signal(str)

    def __init__(self, adb_mgr: ADBManager) -> None:
        super().__init__()
        self.adb_mgr = adb_mgr

    def run(self) -> None:
        """Distinguish an empty successful scan from an actual scan failure."""
        try:
            devices = self.adb_mgr.scan_mumu_instances()

        except Exception as exc:
            message = str(exc)

            signals.log_emitted.emit(
                "ERROR",
                "ADB",
                f"Device scan failed: {message}",
                "SYSTEM",
            )

            self.failed_scan.emit(message)
            return

        self.finished_scan.emit(devices)


def draw_roi_overlay(
    frame: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (254, 242, 0),  # Neon Cyan/Yellow BGR
    label: str = "SAVED ROI",
) -> np.ndarray:
    """Draws a high-visibility glowing bounding box overlay on the image frame matrix."""
    if frame is None or frame.size == 0 or roi is None:
        return frame

    x, y, w, h = roi
    fh, fw = frame.shape[:2]

    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > fw or y + h > fh:
        return frame

    frame_copy = frame.copy()

    cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, 2)

    badge_text = f" {label} [{x}, {y}, {w}x{h}] "
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(
        badge_text, font, font_scale, thickness
    )

    label_y = max(y - 6, text_h + 4)
    cv2.rectangle(
        frame_copy,
        (x, label_y - text_h - 4),
        (x + text_w, label_y + baseline),
        (15, 21, 37),
        -1,
    )
    cv2.rectangle(
        frame_copy,
        (x, label_y - text_h - 4),
        (x + text_w, label_y + baseline),
        color,
        1,
    )
    cv2.putText(
        frame_copy,
        badge_text,
        (x, label_y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    return frame_copy


def draw_tap_indicator(
    frame: np.ndarray,
    x: int,
    y: int,
    label: str = "TAP",
    color: Tuple[int, int, int] = (0, 242, 254),  # Electric Cyan BGR
) -> np.ndarray:
    """Draws a high-visibility target crosshair and coordinate badge overlay on the frame matrix."""
    if frame is None or frame.size == 0:
        return frame

    fh, fw = frame.shape[:2]
    if not (0 <= x < fw and 0 <= y < fh):
        return frame

    frame_copy = frame.copy()

    # Target concentric circles
    cv2.circle(frame_copy, (x, y), 18, color, 2, cv2.LINE_AA)
    cv2.circle(
        frame_copy, (x, y), 6, (0, 0, 255), -1, cv2.LINE_AA
    )  # Red inner center dot
    cv2.circle(frame_copy, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)

    # Crosshair reticle lines
    cv2.line(frame_copy, (x - 25, y), (x - 8, y), color, 2, cv2.LINE_AA)
    cv2.line(frame_copy, (x + 8, y), (x + 25, y), color, 2, cv2.LINE_AA)
    cv2.line(frame_copy, (x, y - 25), (x, y - 8), color, 2, cv2.LINE_AA)
    cv2.line(frame_copy, (x, y + 8), (x, y + 25), color, 2, cv2.LINE_AA)

    # Coordinate badge text box
    badge_text = f" 🎯 {label} [{x}, {y}] "
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(
        badge_text, font, font_scale, thickness
    )

    label_x = max(0, min(x + 20, fw - text_w - 5))
    label_y = max(text_h + 5, min(y - 10, fh - 5))

    # Dark background badge rectangle
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


class InteractiveCropCanvas(QLabel):
    """Interactive Qt Label widget enabling mouse drag-to-crop ROI selection with persistent overlay memory."""

    roi_selected = Signal(tuple)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "background-color: #06090f; border: 1px solid rgba(0, 242, 254, 0.3); border-radius: 6px;"
        )
        self.setMinimumSize(480, 270)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._rubber_band: Optional[QRubberBand] = None
        self._origin_point = QPoint()
        self._current_frame: Optional[np.ndarray] = None
        self._cropped_frame: Optional[np.ndarray] = None
        self._active_roi: Optional[Tuple[int, int, int, int]] = None

    def set_frame(self, frame: np.ndarray) -> None:
        """Replace the source image and invalidate all previous crop state."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return

        self._current_frame = frame.copy()
        self._cropped_frame = None
        self._active_roi = None

        self._update_display()

    def set_active_roi(
        self,
        roi: Optional[Tuple[int, int, int, int]],
    ) -> None:
        """Update the ROI and discard stale crop data when the ROI is cleared."""
        if roi is None:
            self._active_roi = None
            self._cropped_frame = None

        else:
            if self._current_frame is None or len(roi) != 4:
                return

            x, y, width, height = map(int, roi)

            frame_h, frame_w = self._current_frame.shape[:2]

            if (
                x < 0
                or y < 0
                or width <= 0
                or height <= 0
                or x + width > frame_w
                or y + height > frame_h
            ):
                return

            self._active_roi = (x, y, width, height)

            self._cropped_frame = self._current_frame[
                y : y + height,
                x : x + width,
            ].copy()

        self._update_display()

    def get_cropped_frame(self) -> Optional[np.ndarray]:
        """Returns cropped image matrix if ROI selected, otherwise full source frame."""
        return (
            self._cropped_frame
            if self._cropped_frame is not None
            else self._current_frame
        )

    def _update_display(self) -> None:
        if self._current_frame is None:
            return

        display_matrix = self._current_frame
        if self._active_roi is not None:
            display_matrix = draw_roi_overlay(
                self._current_frame, self._active_roi, label="CROP ROI"
            )

        pixmap = cv_frame_to_qpixmap(display_matrix, self.width(), self.height())
        self.setPixmap(pixmap)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_display()

    def mousePressEvent(self, event) -> None:
        if (
            event.button() == Qt.LeftButton
            and self.pixmap()
            and not self.pixmap().isNull()
        ):
            self._origin_point = event.pos()
            if not self._rubber_band:
                self._rubber_band = QRubberBand(QRubberBand.Rectangle, self)
            self._rubber_band.setGeometry(QRect(self._origin_point, QSize()))
            self._rubber_band.show()

    def mouseMoveEvent(self, event) -> None:
        if self._rubber_band and self._rubber_band.isVisible():
            self._rubber_band.setGeometry(
                QRect(self._origin_point, event.pos()).normalized()
            )

    def mouseReleaseEvent(self, event) -> None:
        if (
            event.button() == Qt.LeftButton
            and self._rubber_band
            and self._current_frame is not None
        ):
            self._rubber_band.hide()
            rect = QRect(self._origin_point, event.pos()).normalized()

            if rect.width() < 10 or rect.height() < 10:
                return

            pm = self.pixmap()
            if not pm or pm.isNull():
                return

            pm_w, pm_h = pm.width(), pm.height()
            lbl_w, lbl_h = self.width(), self.height()

            if pm_w <= 0 or pm_h <= 0:
                return

            offset_x = max(0, (lbl_w - pm_w) // 2)
            offset_y = max(0, (lbl_h - pm_h) // 2)

            adj_x1 = max(0, min(rect.left() - offset_x, pm_w))
            adj_y1 = max(0, min(rect.top() - offset_y, pm_h))
            adj_x2 = max(0, min(rect.right() - offset_x, pm_w))
            adj_y2 = max(0, min(rect.bottom() - offset_y, pm_h))

            crop_w = adj_x2 - adj_x1
            crop_h = adj_y2 - adj_y1

            if crop_w <= 0 or crop_h <= 0:
                return

            orig_h, orig_w = self._current_frame.shape[:2]
            scale_x = orig_w / float(pm_w)
            scale_y = orig_h / float(pm_h)

            real_x = min(int(adj_x1 * scale_x), orig_w - 1)
            real_y = min(int(adj_y1 * scale_y), orig_h - 1)
            real_w = max(1, min(int(crop_w * scale_x), orig_w - real_x))
            real_h = max(1, min(int(crop_h * scale_y), orig_h - real_y))

            self._cropped_frame = self._current_frame[
                real_y : real_y + real_h, real_x : real_x + real_w
            ].copy()

            self._active_roi = (real_x, real_y, real_w, real_h)
            self._update_display()
            self.roi_selected.emit(self._active_roi)


class InstanceCardWidget(QFrame):
    """Native widget card representing a single emulator instance with click-to-tap visual feed testing."""

    def __init__(self, serial: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.serial = serial
        self._is_running = False
        self._last_raw_frame_size: Optional[Tuple[int, int]] = None
        self.setObjectName("glassPanel")
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setFixedWidth(344)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Header Row
        header_widget = QWidget()
        header_widget.setFixedHeight(28)
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)

        self.title_label = QLabel(f"📱 {self.serial}")
        self.title_label.setStyleSheet(
            "color: #00f2fe; font-weight: bold; font-size: 13px;"
        )

        self.state_badge = QLabel("INIT")
        self.state_badge.setAlignment(Qt.AlignCenter)
        self.state_badge.setFixedHeight(22)
        self.state_badge.setStyleSheet(
            "background: rgba(0, 242, 254, 0.2); color: #00f2fe; "
            "border: 1px solid rgba(0, 242, 254, 0.5); border-radius: 4px; "
            "padding: 2px 8px; font-size: 10px; font-weight: bold;"
        )

        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.state_badge)
        layout.addWidget(header_widget)

        # Stream Canvas
        self.canvas = QLabel("Connecting Stream...")
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setFixedSize(320, 180)
        self.canvas.setStyleSheet(
            "background-color: #06090f; border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 6px;"
        )
        self.canvas.mousePressEvent = self._on_canvas_clicked
        layout.addWidget(self.canvas)

        # Controls Row
        self.btn_toggle = QPushButton("Start Instance")
        self.btn_toggle.setFixedHeight(32)
        self.btn_toggle.clicked.connect(self._on_toggle)
        layout.addWidget(self.btn_toggle)

    def _on_canvas_clicked(self, event) -> None:
        """Handles direct mouse clicks on preview cards to execute test taps and display visual overlays."""
        if event.button() != Qt.LeftButton or not self._last_raw_frame_size:
            return

        pm = self.canvas.pixmap()
        if not pm or pm.isNull():
            return

        click_pos = event.pos()
        lbl_w, lbl_h = self.canvas.width(), self.canvas.height()
        pm_w, pm_h = pm.width(), pm.height()

        offset_x = (lbl_w - pm_w) // 2
        offset_y = (lbl_h - pm_h) // 2

        rel_x = click_pos.x() - offset_x
        rel_y = click_pos.y() - offset_y

        if 0 <= rel_x < pm_w and 0 <= rel_y < pm_h:
            orig_w, orig_h = self._last_raw_frame_size
            scale_x = orig_w / float(pm_w)
            scale_y = orig_h / float(pm_h)

            real_x = int(rel_x * scale_x)
            real_y = int(rel_y * scale_y)

            window = self.window()
            if hasattr(window, "workers") and self.serial in window.workers:
                worker = window.workers[self.serial]
                worker.register_tap(real_x, real_y, label="MANUAL TEST TAP")
                signals.log_emitted.emit(
                    "INFO",
                    "ACTION",
                    f"👈 Manual test tap fired at [{real_x}, {real_y}]. Overlay updated.",
                    self.serial,
                )

    def update_frame(
        self,
        frame: np.ndarray,
        active_roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        """Renders live instance frame with remembered ROI bounding box overlay."""
        if frame is None or frame.size == 0:
            return

        self._last_raw_frame_size = (frame.shape[1], frame.shape[0])

        display_frame = frame
        if active_roi is not None:
            display_frame = draw_roi_overlay(frame, active_roi, label="ACTIVE ROI")

        pixmap = cv_frame_to_qpixmap(display_frame, 320, 180)
        if not pixmap.isNull():
            self.canvas.setPixmap(pixmap)

    def update_state(self, state_str: str) -> None:
        self.state_badge.setText(state_str)
        if state_str == "STOPPED":
            self._is_running = False
            self.btn_toggle.setText("Start Instance")
        else:
            self._is_running = True
            self.btn_toggle.setText("Stop Instance")

    def _on_toggle(self) -> None:
        window = self.window()
        if self._is_running:
            self._is_running = False
            self.btn_toggle.setText("Start Instance")
            if hasattr(window, "stop_worker"):
                window.stop_worker(self.serial)
        else:
            self._is_running = True
            self.btn_toggle.setText("Stop Instance")
            if hasattr(window, "start_worker"):
                window.start_worker(self.serial)


class MainWindow(QMainWindow):
    """Main Application Window with Thread-Safe Lifecycle Teardown and Liquid Glass Layout."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Ragnarok: The New World — Multi-Bot Native Launcher")
        self.resize(1400, 900)
        self.setStyleSheet(LiquidGlassTheme.MAIN_QSS)

        # Lifecycle State Flags & Active Targets
        self._is_shutting_down = False
        self.active_roi: Optional[Tuple[int, int, int, int]] = None

        # Initialize Core Backend Components
        self.adb_mgr = ADBManager()
        self.vision = VisionEngine(use_gpu=False)
        self.workers: Dict[str, InstanceWorker] = {}
        self.worker_cards: Dict[str, InstanceCardWidget] = {}
        self.worker_futures: Dict[str, concurrent.futures.Future] = {}

        self.idle_frame_futures: Dict[str, concurrent.futures.Future] = {}

        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=ADBConfig().max_instances + 4,
            thread_name_prefix="instance-worker",
        )
        self.scan_worker: Optional[ADBScanWorker] = None

        # Request IDs prevent stale background results from updating the UI.
        self._capture_request_id = 0
        self._ocr_request_id = 0

        # Connect Global Signals owned by MainWindow.
        signals.log_emitted.connect(self.append_log)
        signals.state_changed.connect(self.on_state_changed)
        signals.frame_ready.connect(self.on_frame_ready)
        signals.ui_task_finished.connect(self._on_ui_task_finished)

        self._build_ui()

        # Low-Frequency Timer (3000 ms) for Idle Frame Polling
        self.idle_timer = QTimer(self)
        self.idle_timer.timeout.connect(self._poll_idle_frames_async)
        self.idle_timer.start(3000)

        # Auto-rescan ADB instances asynchronously
        self.rescan_adb()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Header Quick Controls
        header_panel = QFrame()
        header_panel.setObjectName("glassPanel")
        header_layout = QHBoxLayout(header_panel)

        title_box = QVBoxLayout()
        title = QLabel("Ragnarok Multi-Bot Launcher")
        title.setObjectName("headerTitle")
        sub = QLabel("Liquid Glass Dark Mode Native Dashboard")
        sub.setObjectName("subText")
        title_box.addWidget(title)
        title_box.addWidget(sub)
        header_layout.addLayout(title_box)

        header_layout.addStretch()

        self.btn_rescan = QPushButton("🔄 Rescan ADB")
        self.btn_rescan.clicked.connect(self.rescan_adb)
        btn_start_all = QPushButton("▶ Start All")
        btn_start_all.clicked.connect(self.start_all)
        btn_stop_all = QPushButton("⏸ Pause All", objectName="dangerButton")
        btn_stop_all.clicked.connect(self.stop_all)

        header_layout.addWidget(self.btn_rescan)
        header_layout.addWidget(btn_start_all)
        header_layout.addWidget(btn_stop_all)
        main_layout.addWidget(header_panel)

        # Main Splitter Dashboard Layout
        splitter = QSplitter(Qt.Vertical)

        self.tabs = QTabWidget()

        # Tab 1: Live Instances Grid
        self.grid_scroll = QScrollArea()
        self.grid_scroll.setWidgetResizable(True)
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.grid_layout.setSpacing(16)
        self.grid_scroll.setWidget(self.grid_widget)
        self.tabs.addTab(self.grid_scroll, "📺 Live Instances")

        # Tab 2 & 3: Workbench and Inspector
        self.tabs.addTab(self._build_vision_bench(), "👁 Vision & OCR Bench")
        self.tabs.addTab(self._build_config_inspector(), "⚙ Config Inspector")

        # Automatically refresh dropdown items when switching to Vision Bench
        self.tabs.currentChanged.connect(self._on_tab_changed)

        splitter.addWidget(self.tabs)

        # Terminal Log Panel
        log_panel = QFrame()
        log_panel.setObjectName("glassPanel")
        log_layout = QVBoxLayout(log_panel)

        log_header = QHBoxLayout()
        log_title = QLabel("Terminal Telemetry Logs")
        log_title.setStyleSheet("color: #00f2fe; font-weight: bold;")
        log_header.addWidget(log_title)
        log_header.addStretch()

        btn_clear_log = QPushButton("Clear")
        btn_clear_log.clicked.connect(lambda: self.terminal_log.clear())
        log_header.addWidget(btn_clear_log)
        log_layout.addLayout(log_header)

        self.terminal_log = QTextEdit()
        self.terminal_log.setObjectName("terminalLog")
        self.terminal_log.setReadOnly(True)
        log_layout.addWidget(self.terminal_log)

        splitter.addWidget(log_panel)
        splitter.setSizes([600, 250])

        main_layout.addWidget(splitter)

    def _publish_ui_task_result(
        self,
        operation: str,
        request_id: int,
        serial: str,
        future: concurrent.futures.Future,
    ) -> None:
        """Publish background task results through the Qt signal bridge."""
        try:
            result = future.result()
            error_message = ""

        except Exception as exc:
            result = None
            error_message = str(exc)

        signals.ui_task_finished.emit(
            operation,
            request_id,
            serial,
            result,
            error_message,
        )

    @Slot(str, int, str, object, str)
    def _on_ui_task_finished(
        self,
        operation: str,
        request_id: int,
        serial: str,
        result: object,
        error_message: str,
    ) -> None:
        """Apply completed background operations on the GUI thread."""
        if self._is_shutting_down:
            return

        if operation == "capture":
            if request_id != self._capture_request_id:
                return

            if error_message:
                self.lbl_crop_info.setText(
                    f"❌ Frame capture failed for {serial}: {error_message}"
                )
                return

            if not isinstance(result, np.ndarray) or result.size == 0:
                self.lbl_crop_info.setText(f"❌ Failed to capture frame from {serial}.")
                return

            self.active_roi = None

            self.crop_canvas.set_frame(result)

            self.lbl_crop_preview.setText("Full Frame Active (Drag to crop region)")

            self.lbl_crop_preview.setPixmap(QPixmap())

            self.lbl_crop_info.setText(
                f"✅ Frame captured from {serial} "
                f"({result.shape[1]}x{result.shape[0]}px). "
                "Drag to crop region."
            )

            return

        if operation == "ocr":
            if request_id != self._ocr_request_id:
                return

            if error_message:
                self.ocr_bench_output.setPlainText(f"❌ OCR Error: {error_message}")
                return

            text_results = result if isinstance(result, list) else []

            if text_results:
                formatted = "\n".join(f"• {str(item)}" for item in text_results)

                self.ocr_bench_output.setPlainText(formatted)

            else:
                self.ocr_bench_output.setPlainText(
                    "⚠️ No Thai text detected in selected crop region."
                )

    def _on_tab_changed(self, index: int) -> None:
        """Triggers dynamic dropdown refresh when switching tabs."""
        if index == 1:  # Vision & OCR Bench tab
            self._update_instance_combo()

    def _update_instance_combo(self) -> None:
        """Populates instance selection dropdown with active ADB devices while preserving selection."""
        if not hasattr(self, "combo_instances"):
            return

        current_serial = self.combo_instances.currentData()
        self.combo_instances.blockSignals(True)
        self.combo_instances.clear()

        for serial in self.workers.keys():
            self.combo_instances.addItem(f"📱 {serial}", userData=serial)

        # Restore previous selection or default to first instance
        index_to_select = -1
        if current_serial:
            for i in range(self.combo_instances.count()):
                if self.combo_instances.itemData(i) == current_serial:
                    index_to_select = i
                    break

        if index_to_select >= 0:
            self.combo_instances.setCurrentIndex(index_to_select)
        elif self.combo_instances.count() > 0:
            self.combo_instances.setCurrentIndex(0)

        self.combo_instances.blockSignals(False)

    def _build_vision_bench(self) -> QWidget:
        """Constructs interactive Vision & OCR Workbench with target execution mode selection."""
        bench = QWidget()
        layout = QHBoxLayout(bench)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Left Panel: Interactive Crop Canvas
        left_box = QGroupBox("📸 Interactive Live Frame Selector & Cropper")
        left_box.setObjectName("glassPanel")
        left_layout = QVBoxLayout(left_box)

        # Controls Toolbar
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Instance Source:"))
        self.combo_instances = QComboBox()
        self.combo_instances.setMinimumWidth(160)
        toolbar.addWidget(self.combo_instances)

        btn_capture = QPushButton("📷 Snap Current Frame")
        btn_capture.clicked.connect(self._on_capture_vision_frame)
        toolbar.addWidget(btn_capture)

        btn_clear_roi = QPushButton("🗑 Clear Target", objectName="dangerButton")
        btn_clear_roi.clicked.connect(self._clear_saved_roi)
        toolbar.addWidget(btn_clear_roi)

        toolbar.addStretch()
        left_layout.addLayout(toolbar)

        # Canvas
        self.crop_canvas = InteractiveCropCanvas()
        self.crop_canvas.roi_selected.connect(self._on_roi_selected)
        left_layout.addWidget(self.crop_canvas)

        self.lbl_crop_info = QLabel(
            "💡 Tip: Drag to crop a button/dialog, then select an execution mode below."
        )
        self.lbl_crop_info.setStyleSheet("color: #94a3b8; font-size: 11px;")
        left_layout.addWidget(self.lbl_crop_info)

        layout.addWidget(left_box, stretch=3)

        # Right Panel: OCR & Target Dispatcher
        right_box = QGroupBox("👁 OCR & Target Dispatcher")
        right_box.setObjectName("glassPanel")
        right_layout = QVBoxLayout(right_box)

        right_layout.addWidget(QLabel("Selected Crop Region Preview:"))
        self.lbl_crop_preview = QLabel("No Region Selected")
        self.lbl_crop_preview.setAlignment(Qt.AlignCenter)
        self.lbl_crop_preview.setFixedHeight(110)
        self.lbl_crop_preview.setStyleSheet(
            "background-color: #06090f; border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 6px;"
        )
        right_layout.addWidget(self.lbl_crop_preview)

        # Target Mode Option Dropdown
        right_layout.addWidget(QLabel("Target Execution Mode:"))
        self.combo_target_mode = QComboBox()
        self.combo_target_mode.addItem(
            "⚡ One-Shot Click & Resume Questing", userData=True
        )
        self.combo_target_mode.addItem("🔄 Continuous Target Clicker", userData=False)
        right_layout.addWidget(self.combo_target_mode)

        # Dispatcher Button
        btn_set_target = QPushButton("🎯 Apply Target to All Instances")
        btn_set_target.setFixedHeight(36)
        btn_set_target.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00c6ff, stop:1 #0072ff);"
            "color: #ffffff; font-weight: bold; font-size: 12px; border-radius: 6px;"
        )
        btn_set_target.clicked.connect(self._on_set_target_all)
        right_layout.addWidget(btn_set_target)

        btn_run_ocr = QPushButton("🔤 Extract Thai OCR Text")
        btn_run_ocr.setFixedHeight(30)
        btn_run_ocr.clicked.connect(self._on_run_bench_ocr)
        right_layout.addWidget(btn_run_ocr)

        right_layout.addWidget(QLabel("Extracted Text Stream / Target Status:"))
        self.ocr_bench_output = QTextEdit()
        self.ocr_bench_output.setObjectName("terminalLog")
        right_layout.addWidget(self.ocr_bench_output)

        layout.addWidget(right_box, stretch=2)
        return bench

    def _on_set_target_all(self) -> None:
        """Broadcast a valid cropped template to all discovered instances."""
        cropped_matrix = self.crop_canvas.get_cropped_frame()

        if (
            self.active_roi is None
            or cropped_matrix is None
            or cropped_matrix.size == 0
        ):
            self.ocr_bench_output.setText(
                "⚠️ Please capture a frame and select a crop region first."
            )
            return

        if not self.workers:
            self.ocr_bench_output.setText(
                "⚠️ No discovered instances are available to assign the target."
            )
            return

        is_one_shot = bool(self.combo_target_mode.currentData())

        target_roi = tuple(self.active_roi)
        target_template = cropped_matrix.copy()

        for worker in list(self.workers.values()):
            worker.active_target_template = target_template
            worker.active_target_roi = target_roi
            worker.one_shot_target = is_one_shot
            worker.state = BotState.CHECK_QUEST

        count = len(self.workers)

        mode_str = "One-Shot & Resume" if is_one_shot else "Continuous"

        self.ocr_bench_output.setText(
            f"✅ Target ({mode_str}) assigned to {count} instance(s)."
        )

        self.lbl_crop_info.setText(
            f"🎯 Active Target Mode: {mode_str} across {count} instance(s)."
        )

        signals.log_emitted.emit(
            "INFO",
            "ACTION",
            f"Broadcast dynamic target ({mode_str}) to {count} instance(s).",
            "SYSTEM",
        )

    def _clear_saved_roi(self) -> None:
        """Clears remembered target coordinates and resets dynamic targets across all instances."""
        self.active_roi = None
        self._ocr_request_id += 1
        self.crop_canvas.set_active_roi(None)
        self.lbl_crop_preview.setText("No Region Selected")
        self.lbl_crop_preview.setPixmap(QPixmap())
        self.lbl_crop_info.setText("🗑 Saved ROI Overlay cleared.")

        for worker in self.workers.values():
            worker.active_target_template = None
            worker.active_target_roi = None

        self.ocr_bench_output.setText(
            "Cleared active target from all instance workers."
        )
        signals.log_emitted.emit(
            "WARN",
            "VISION",
            "Cleared active ROI and dynamic targets from all instances.",
            "SYSTEM",
        )

    def _on_capture_vision_frame(self) -> None:
        """Capture the selected emulator frame without blocking the GUI."""
        if self.combo_instances.count() == 0:
            self._update_instance_combo()

        if self.combo_instances.count() == 0:
            self.lbl_crop_info.setText(
                "⚠️ No active ADB instances found. Click 'Rescan ADB' first."
            )
            return

        target_serial = self.combo_instances.currentData()

        if not target_serial:
            target_serial = (
                self.combo_instances.currentText().replace("📱 ", "").strip()
            )

        if not target_serial or target_serial not in self.workers:
            self.lbl_crop_info.setText(
                "⚠️ Selected instance is invalid or disconnected."
            )
            return

        self._capture_request_id += 1
        self._ocr_request_id += 1

        request_id = self._capture_request_id

        # Invalidate the old crop while a new frame is being captured.
        self.active_roi = None
        self.crop_canvas.set_active_roi(None)

        self.lbl_crop_info.setText(f"⏳ Capturing frame from {target_serial}...")

        try:
            future = self.executor.submit(
                self.adb_mgr.capture_frame,
                target_serial,
            )

        except RuntimeError as exc:
            self.lbl_crop_info.setText(f"❌ Could not schedule frame capture: {exc}")
            return

        future.add_done_callback(
            lambda completed, rid=request_id, serial=target_serial: self._publish_ui_task_result(
                "capture",
                rid,
                serial,
                completed,
            )
        )

    @Slot(tuple)
    def _on_roi_selected(self, roi: Tuple[int, int, int, int]) -> None:
        """Handles ROI selection signal from canvas and updates crop preview label."""
        self.active_roi = roi
        self._ocr_request_id += 1
        x, y, w, h = roi
        self.lbl_crop_info.setText(
            f"📐 Saved ROI Overlay: X={x}, Y={y}, Width={w}px, Height={h}px"
        )

        cropped = self.crop_canvas.get_cropped_frame()
        if cropped is not None and cropped.size > 0:
            pixmap = cv_frame_to_qpixmap(
                cropped, self.lbl_crop_preview.width(), self.lbl_crop_preview.height()
            )
            self.lbl_crop_preview.setPixmap(pixmap)

        signals.log_emitted.emit(
            "INFO", "VISION", f"Active ROI overlay saved: {roi}", "SYSTEM"
        )

    def _on_run_bench_ocr(self) -> None:
        """Execute EasyOCR outside the GUI thread using a copied image."""
        if self.active_roi is None:
            self.ocr_bench_output.setPlainText(
                "⚠️ Please capture a frame and crop a region first."
            )
            return

        target_img = self.crop_canvas.get_cropped_frame()

        if target_img is None or target_img.size == 0:
            self.ocr_bench_output.setPlainText(
                "⚠️ Please capture a frame and crop a region first."
            )
            return

        self._ocr_request_id += 1

        request_id = self._ocr_request_id
        image_copy = target_img.copy()

        self.ocr_bench_output.setPlainText("⏳ Processing EasyOCR text recognition...")

        try:
            future = self.executor.submit(
                self.vision.read_thai_text,
                image_copy,
            )

        except RuntimeError as exc:
            self.ocr_bench_output.setPlainText(f"❌ Could not schedule OCR: {exc}")
            return

        future.add_done_callback(
            lambda completed, rid=request_id: self._publish_ui_task_result(
                "ocr",
                rid,
                "",
                completed,
            )
        )

    def _build_config_inspector(self) -> QWidget:
        inspector = QWidget()
        layout = QHBoxLayout(inspector)

        adb_box = QGroupBox("ADB Connection Parameters")
        adb_box.setObjectName("glassPanel")
        form = QFormLayout(adb_box)

        cfg = ADBConfig()
        form.addRow("Host IP:", QLineEdit(cfg.host))
        form.addRow("Default Port:", QSpinBox(value=cfg.default_port, maximum=65535))
        form.addRow("Custom ADB Path:", QLineEdit(cfg.custom_adb_path or ""))

        btn_save = QPushButton("Save Settings")
        form.addRow(btn_save)
        layout.addWidget(adb_box)

        return inspector

    @Slot(str, str, str, str)
    def append_log(
        self,
        level: str,
        tag: str,
        message: str,
        serial: str,
    ) -> None:
        """Append escaped log content to prevent HTML markup injection."""
        if self._is_shutting_down:
            return

        timestamp = time.strftime("%H:%M:%S")

        color_map = {
            "INFO": "#38bdf8",
            "WARN": "#fbbf24",
            "ERROR": "#f87171",
            "ADB": "#c084fc",
            "ACTION": "#a855f7",
        }

        level_text = str(level)
        color = color_map.get(
            level_text,
            "#f1f5f9",
        )

        safe_timestamp = html.escape(timestamp)
        safe_level = html.escape(level_text)
        safe_tag = html.escape(str(tag))
        safe_serial = html.escape(str(serial))
        safe_message = html.escape(str(message))

        formatted = (
            f'<span style="color:#64748b">'
            f"[{safe_timestamp}]</span> "
            f'<span style="color:{color}">'
            f"<b>[{safe_level}]</b> "
            f"[{safe_tag}] ({safe_serial}): "
            f"{safe_message}</span>"
        )

        self.terminal_log.append(formatted)

    @Slot(str, str)
    def on_state_changed(self, serial: str, state_name: str) -> None:
        if not self._is_shutting_down and serial in self.worker_cards:
            self.worker_cards[serial].update_state(state_name)

    @Slot(str, object)
    def on_frame_ready(self, serial: str, frame: np.ndarray) -> None:
        """Renders live instance frames with remembered ROI bounding box overlay."""
        if not self._is_shutting_down and serial in self.worker_cards:
            self.worker_cards[serial].update_frame(frame, self.active_roi)

    def rescan_adb(self) -> None:
        """Start one ADB scan without blocking the GUI."""
        if self._is_shutting_down:
            return

        if self.scan_worker is not None and self.scan_worker.isRunning():
            return

        self.btn_rescan.setEnabled(False)
        self.btn_rescan.setText("Scanning...")

        signals.log_emitted.emit(
            "INFO",
            "ADB",
            "Scanning local ADB instances in background...",
            "SYSTEM",
        )

        self.scan_worker = ADBScanWorker(self.adb_mgr)

        self.scan_worker.finished_scan.connect(self._on_scan_finished)

        self.scan_worker.failed_scan.connect(self._on_scan_failed)

        self.scan_worker.start()

    @Slot(str)
    def _on_scan_failed(self, message: str) -> None:
        """Restore the scan controls without deleting existing device cards."""
        if self._is_shutting_down:
            return

        self.btn_rescan.setEnabled(True)
        self.btn_rescan.setText("🔄 Rescan ADB")

        signals.log_emitted.emit(
            "ERROR",
            "ADB",
            f"Scan failed; existing instances were retained: {message}",
            "SYSTEM",
        )

    @Slot(list)
    def _on_scan_finished(self, devices: List[str]) -> None:
        """Handles scan completion by registering new instances and updating dropdowns."""
        if self._is_shutting_down:
            return

        self.btn_rescan.setEnabled(True)
        self.btn_rescan.setText("🔄 Rescan ADB")

        if not devices:
            signals.log_emitted.emit(
                "WARN",
                "ADB",
                "Scan completed, but no authorized local emulator was found. "
                "Check `adb devices -l`, the ADB executable path, and emulator ports.",
                "SYSTEM",
            )

        # Step 1: Remove cards for devices that were deduplicated or closed
        stale_serials = [s for s in list(self.workers.keys()) if s not in devices]
        for serial in stale_serials:
            if serial in self.workers:
                self.workers[serial].stop()
                del self.workers[serial]
            if serial in self.worker_cards:
                card = self.worker_cards.pop(serial)
                self.grid_layout.removeWidget(card)
                card.deleteLater()
                signals.log_emitted.emit(
                    "WARN",
                    "ADB",
                    f"Pruned duplicate/stale instance card {serial}",
                    serial,
                )

        # Step 2: Register new instance workers and cards
        for serial in devices:
            if serial not in self.workers:
                self.workers[serial] = InstanceWorker(serial, self.adb_mgr, self.vision)
                card = InstanceCardWidget(serial)
                self.worker_cards[serial] = card
                signals.log_emitted.emit(
                    "INFO", "ADB", f"Registered instance {serial}", serial
                )

        # Step 3: Reflow active cards and refresh Vision Bench instance selector
        self._relayout_cards()
        self._update_instance_combo()

    def _relayout_cards(self) -> None:
        """Re-arranges active instance cards dynamically in a 2-column grid."""
        for idx, (serial, card) in enumerate(self.worker_cards.items()):
            self.grid_layout.removeWidget(card)
            row, col = divmod(idx, 2)
            self.grid_layout.addWidget(card, row, col)

    def start_worker(self, serial: str) -> None:
        """Submit at most one active or pending worker execution per device."""
        if self._is_shutting_down:
            return

        worker = self.workers.get(serial)

        if worker is None:
            return

        existing = self.worker_futures.get(serial)

        if existing is not None and not existing.done():
            return

        if not worker.prepare_start():
            return

        try:
            future = self.executor.submit(worker.run)
            self.worker_futures[serial] = future

        except RuntimeError as exc:
            worker.cancel_start()

            signals.log_emitted.emit(
                "ERROR",
                "WORKER",
                f"Could not start worker: {exc}",
                serial,
            )

    def stop_worker(self, serial: str) -> None:
        """Stop an active worker or cancel its queued execution."""
        worker = self.workers.get(serial)

        if worker is None:
            return

        worker.stop()

        future = self.worker_futures.get(serial)

        if future is not None and not future.running():
            future.cancel()

    def start_all(self) -> None:
        for serial in self.workers:
            self.start_worker(serial)

    def stop_all(self) -> None:
        for serial in self.workers:
            self.stop_worker(serial)

    def _poll_idle_frames_async(self) -> None:
        """Prevent multiple outstanding idle-frame captures for one device."""
        if self._is_shutting_down:
            return

        for serial, worker in list(self.workers.items()):
            if worker.is_running:
                continue

            worker_future = self.worker_futures.get(serial)

            if worker_future is not None and not worker_future.done():
                continue

            idle_future = self.idle_frame_futures.get(serial)

            if idle_future is not None and not idle_future.done():
                continue

            try:
                self.idle_frame_futures[serial] = self.executor.submit(
                    self._capture_idle_frame_task,
                    serial,
                )

            except RuntimeError:
                return

    def _capture_idle_frame_task(self, serial: str) -> None:
        if self._is_shutting_down:
            return
        frame = self.adb_mgr.capture_frame(serial)
        if frame is not None and not self._is_shutting_down:
            signals.frame_ready.emit(serial, frame)

    def closeEvent(self, event) -> None:
        """Shut down timers, workers, background tasks, and signal connections."""
        self._is_shutting_down = True

        signal_slots = (
            (signals.log_emitted, self.append_log),
            (signals.state_changed, self.on_state_changed),
            (signals.frame_ready, self.on_frame_ready),
            (signals.ui_task_finished, self._on_ui_task_finished),
        )

        for signal, slot in signal_slots:
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass

        if hasattr(self, "idle_timer") and self.idle_timer.isActive():
            self.idle_timer.stop()

        self.stop_all()

        for future in self.worker_futures.values():
            if not future.running():
                future.cancel()

        for future in self.idle_frame_futures.values():
            if not future.running():
                future.cancel()

        # A custom QThread.run() method does not use QThread.exec().
        # Therefore quit() cannot interrupt the scan operation.
        if self.scan_worker is not None and self.scan_worker.isRunning():
            self.scan_worker.wait()

        self.executor.shutdown(
            wait=True,
            cancel_futures=True,
        )

        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
