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
        self,
        device_serial: str,
        target_minimap_offset: Tuple[int, int],
    ) -> None:
        """Tap a target only when it is inside the minimap rectangle."""
        x_offset, y_offset, width, height = map(
            int,
            self.minimap_roi,
        )

        target_x, target_y = map(
            int,
            target_minimap_offset,
        )

        if (
            x_offset < 0
            or y_offset < 0
            or width <= 0
            or height <= 0
        ):
            raise ValueError(
                "minimap_roi must be a positive in-frame rectangle"
            )

        if not (
            0 <= target_x < width
            and 0 <= target_y < height
        ):
            raise ValueError(
                "target_minimap_offset must lie inside minimap_roi"
            )

        self.adb.tap(
            device_serial,
            x_offset + target_x,
            y_offset + target_y,
        )

    def move_joystick_towards(
        self,
        device_serial: str,
        current_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
    ) -> None:
        """Move the joystick toward a target without swiping at zero distance."""
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]

        if dx == 0 and dy == 0:
            return

        angle_rad = math.atan2(-dy, dx)

        center_x, center_y = self.joystick_center

        distance = 70

        end_x = int(
            round(center_x + distance * math.cos(angle_rad))
        )

        end_y = int(
            round(center_y - distance * math.sin(angle_rad))
        )

        self.adb.swipe(
            device_serial,
            center_x,
            center_y,
            end_x,
            end_y,
            duration_ms=400,
        )