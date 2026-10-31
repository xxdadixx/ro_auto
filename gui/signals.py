from PySide6.QtCore import QObject, Signal
import numpy as np


class AppSignals(QObject):
    """Global signal bridge facilitating thread-safe communication across workers and UI."""

    # Signal payload: (serial, state_name)
    state_changed = Signal(str, str)

    # Signal payload: (serial, frame_numpy_array)
    frame_ready = Signal(str, object)

    # Signal payload: (level, tag, message, serial)
    log_emitted = Signal(str, str, str, str)

    # Signal payload: list_of_discovered_serials
    instances_scanned = Signal(list)

    # Signal payload: (operation, request_id, serial, result, error_message)
    ui_task_finished = Signal(str, int, str, object, str)


# Shared Global Signal Dispatcher
signals = AppSignals()