import math
from typing import Tuple, List
from core.adb_manager import ADBManager
from navigation.pathfinder import AStarPathfinder
from navigation.minimap_tracker import MinimapTracker


class AutomaticNavigator:
    """Executes automated movements towards map goals."""

    def __init__(
        self,
        adb_mgr: ADBManager,
        tracker: MinimapTracker,
        minimap_roi: Tuple[int, int, int, int],
        joystick_center: Tuple[int, int] = (200, 500),
    ) -> None:
        self.adb = adb_mgr
        self.tracker = tracker
        self.minimap_roi = minimap_roi
        self.joystick_center = joystick_center
        self.pathfinder = AStarPathfinder(grid_w=100, grid_h=100)

    def tap_minimap_target(
        self, device_serial: str, target_minimap_offset: Tuple[int, int]
    ) -> None:
        """Issues ADB tap command to a target point on the minimap HUD."""
        x_offset, y_offset, _, _ = self.minimap_roi
        tap_x = x_offset + target_minimap_offset[0]
        tap_y = y_offset + target_minimap_offset[1]
        self.adb.tap(device_serial, tap_x, tap_y)

    def move_joystick_towards(
        self, device_serial: str, current_pos: Tuple[int, int], target_pos: Tuple[int, int]
    ) -> None:
        """Calculates direction angle and performs a swipe on the virtual joystick."""
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        angle_rad = math.atan2(-dy, dx)  # Screen coordinates inversion

        cx, cy = self.joystick_center
        distance = 70  # Drag length in pixels
        end_x = int(cx + distance * math.cos(angle_rad))
        end_y = int(cy - distance * math.sin(angle_rad))

        self.adb.swipe(device_serial, cx, cy, end_x, end_y, duration_ms=400)