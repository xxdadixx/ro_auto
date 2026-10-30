import cv2
import numpy as np
from typing import Tuple, Optional
from vision.vision import VisionEngine


class MinimapTracker:
    """Tracks player marker position and parses numerical map coordinates."""

    def __init__(self, vision_engine: VisionEngine) -> None:
        self.vision = vision_engine

    def locate_player_on_minimap(
        self, frame: np.ndarray, minimap_roi: Tuple[int, int, int, int]
    ) -> Optional[Tuple[int, int]]:
        """Detects center coordinates of player location marker in minimap ROI."""
        x, y, w, h = minimap_roi
        crop = frame[y : y + h, x : x + w]

        # Extract player arrow icon via HSV color range (bright yellow/green marker)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower_marker = np.array([20, 120, 120])
        upper_marker = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower_marker, upper_marker)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) + x
                cy = int(M["m01"] / M["m00"]) + y
                return (cx, cy)
        return None

    def read_player_coords(
        self, frame: np.ndarray, coord_roi: Tuple[int, int, int, int]
    ) -> Optional[Tuple[int, int]]:
        """Parses printed coordinate string (e.g. '142, 210') using OCR."""
        text_list = self.vision.read_thai_text(frame, roi=coord_roi)
        for raw in text_list:
            cleaned = "".join([c for c in raw if c.isdigit() or c in ",: "])
            parts = cleaned.replace(":", ",").split(",")
            if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                return (int(parts[0].strip()), int(parts[1].strip()))
        return None