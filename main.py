import sys
import time
import concurrent.futures
from typing import Dict, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFrame,
    QTabWidget,
    QGridLayout,
    QTextEdit,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QScrollArea,
    QSplitter,
    QSizePolicy,
)

from config import ADBConfig, BotConfig, VisionConfig
from core.adb_manager import ADBManager
from engine.instance_worker import InstanceWorker
from gui.signals import signals
from gui.styles import LiquidGlassTheme
from vision.vision import VisionEngine


def cv_frame_to_qpixmap(
    frame: np.ndarray, width: int = 320, height: int = 180
) -> QPixmap:
    """Converts OpenCV BGR image matrix to a scaled Qt QPixmap."""
    if frame is None or frame.size == 0:
        return QPixmap()
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(q_img)
    return pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class ADBScanWorker(QThread):
    """Background thread worker to discover ADB instances without freezing the UI."""

    finished_scan = Signal(list)

    def __init__(self, adb_mgr: ADBManager) -> None:
        super().__init__()
        self.adb_mgr = adb_mgr

    def run(self) -> None:
        devices = self.adb_mgr.scan_mumu_instances()
        self.finished_scan.emit(devices)


class InstanceCardWidget(QFrame):
    """Native widget card representing a single emulator instance with responsive layout."""

    def __init__(self, serial: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.serial = serial
        self.setObjectName("glassPanel")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setMinimumWidth(340)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Fixed-Height Header Row
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

        # Video Stream Canvas Container
        self.canvas = QLabel("Connecting Stream...")
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setMinimumSize(320, 180)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.canvas.setStyleSheet(
            "background-color: #06090f; border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 6px;"
        )
        layout.addWidget(self.canvas)

        # Controls Row
        self.btn_toggle = QPushButton("Start Instance")
        self.btn_toggle.setFixedHeight(32)
        self.btn_toggle.clicked.connect(self._on_toggle)
        layout.addWidget(self.btn_toggle)

    def update_frame(self, frame: np.ndarray) -> None:
        pixmap = cv_frame_to_qpixmap(frame, self.canvas.width(), self.canvas.height())
        if not pixmap.isNull():
            self.canvas.setPixmap(pixmap)

    def update_state(self, state_str: str) -> None:
        self.state_badge.setText(state_str)

    def _on_toggle(self) -> None:
        is_running = self.btn_toggle.text() == "Stop Instance"
        window = self.window()
        if is_running:
            self.btn_toggle.setText("Start Instance")
            if hasattr(window, "stop_worker"):
                window.stop_worker(self.serial)
        else:
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

        # Lifecycle State Flags
        self._is_shutting_down = False

        # Initialize Core Backend Components
        self.adb_mgr = ADBManager()
        self.vision = VisionEngine(use_gpu=False)
        self.workers: Dict[str, InstanceWorker] = {}
        self.worker_cards: Dict[str, InstanceCardWidget] = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.scan_worker: Optional[ADBScanWorker] = None

        # Connect Global Signals
        signals.log_emitted.connect(self.append_log)
        signals.state_changed.connect(self.on_state_changed)
        signals.frame_ready.connect(self.on_frame_ready)

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

        tabs = QTabWidget()

        # Tab 1: Live Instances Grid
        self.grid_scroll = QScrollArea()
        self.grid_scroll.setWidgetResizable(True)
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.grid_layout.setSpacing(16)
        self.grid_scroll.setWidget(self.grid_widget)
        tabs.addTab(self.grid_scroll, "📺 Live Instances")

        # Tab 2 & 3: Workbench and Inspector
        tabs.addTab(self._build_vision_bench(), "👁 Vision & OCR Bench")
        tabs.addTab(self._build_config_inspector(), "⚙ Config Inspector")

        splitter.addWidget(tabs)

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

    def _build_vision_bench(self) -> QWidget:
        bench = QWidget()
        layout = QHBoxLayout(bench)

        left_box = QGroupBox("Template & ROI Testing")
        left_box.setObjectName("glassPanel")
        form = QFormLayout(left_box)

        form.addRow("Template File:", QLineEdit("templates/quest_button.png"))
        form.addRow(
            "Matching Threshold:",
            QDoubleSpinBox(value=0.75, maximum=1.0, singleStep=0.05),
        )
        btn_run_match = QPushButton("Test Template Match")
        form.addRow(btn_run_match)

        layout.addWidget(left_box)

        right_box = QGroupBox("EasyOCR Thai Text Parsing")
        right_box.setObjectName("glassPanel")
        ocr_layout = QVBoxLayout(right_box)
        ocr_layout.addWidget(QLabel("Extracted String Stream:"))
        ocr_text = QTextEdit()
        ocr_text.setObjectName("terminalLog")
        ocr_layout.addWidget(ocr_text)

        layout.addWidget(right_box)
        return bench

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
    def append_log(self, level: str, tag: str, message: str, serial: str) -> None:
        if self._is_shutting_down:
            return
        timestamp = time.strftime("%H:%M:%S")
        color_map = {
            "INFO": "#38bdf8",
            "WARN": "#fbbf24",
            "ERROR": "#f87171",
            "ADB": "#c084fc",
        }
        color = color_map.get(level, "#f1f5f9")
        formatted = f'<span style="color:#64748b">[{timestamp}]</span> <span style="color:{color}"><b>[{level}]</b> [{tag}] ({serial}): {message}</span>'
        self.terminal_log.append(formatted)

    @Slot(str, str)
    def on_state_changed(self, serial: str, state_name: str) -> None:
        if not self._is_shutting_down and serial in self.worker_cards:
            self.worker_cards[serial].update_state(state_name)

    @Slot(str, object)
    def on_frame_ready(self, serial: str, frame: np.ndarray) -> None:
        if not self._is_shutting_down and serial in self.worker_cards:
            self.worker_cards[serial].update_frame(frame)

    def rescan_adb(self) -> None:
        """Launches ADB scanning asynchronously to prevent main thread blocking."""
        if self._is_shutting_down:
            return
        self.btn_rescan.setEnabled(False)
        self.btn_rescan.setText("Scanning...")
        signals.log_emitted.emit(
            "INFO", "ADB", "Scanning local ADB instances in background...", "SYSTEM"
        )

        self.scan_worker = ADBScanWorker(self.adb_mgr)
        self.scan_worker.finished_scan.connect(self._on_scan_finished)
        self.scan_worker.start()

    @Slot(list)
    def _on_scan_finished(self, devices: List[str]) -> None:
        if self._is_shutting_down:
            return

        self.btn_rescan.setEnabled(True)
        self.btn_rescan.setText("🔄 Rescan ADB")

        for serial in devices:
            if serial not in self.workers:
                self.workers[serial] = InstanceWorker(serial, self.adb_mgr, self.vision)
                card = InstanceCardWidget(serial)
                self.worker_cards[serial] = card

                idx = len(self.worker_cards) - 1
                row, col = divmod(idx, 2)
                self.grid_layout.addWidget(card, row, col)
                signals.log_emitted.emit(
                    "INFO", "ADB", f"Registered instance {serial}", serial
                )

    def start_worker(self, serial: str) -> None:
        if self._is_shutting_down:
            return
        if serial in self.workers and not self.workers[serial].is_running:
            worker = self.workers[serial]
            try:
                self.executor.submit(worker.run)
            except RuntimeError:
                pass

    def stop_worker(self, serial: str) -> None:
        if serial in self.workers and self.workers[serial].is_running:
            self.workers[serial].stop()

    def start_all(self) -> None:
        for serial in self.workers:
            self.start_worker(serial)

    def stop_all(self) -> None:
        for serial in self.workers:
            self.stop_worker(serial)

    def _poll_idle_frames_async(self) -> None:
        """Captures idle frames in background thread pool with defensive exception boundaries."""
        if self._is_shutting_down:
            return

        for serial, worker in list(self.workers.items()):
            if not worker.is_running:
                try:
                    self.executor.submit(self._capture_idle_frame_task, serial)
                except RuntimeError:
                    break

    def _capture_idle_frame_task(self, serial: str) -> None:
        if self._is_shutting_down:
            return
        frame = self.adb_mgr.capture_frame(serial)
        if frame is not None and not self._is_shutting_down:
            signals.frame_ready.emit(serial, frame)

    def closeEvent(self, event) -> None:
        """Performs graceful shutdown of timers, scan threads, and thread pools."""
        self._is_shutting_down = True

        # Stop periodic timers
        if hasattr(self, "idle_timer") and self.idle_timer.isActive():
            self.idle_timer.stop()

        # Stop ADB scanning worker thread if active
        if self.scan_worker and self.scan_worker.isRunning():
            self.scan_worker.quit()
            self.scan_worker.wait(1000)

        # Signal workers to stop
        self.stop_all()

        # Shutdown executor pool without raising scheduling errors
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
