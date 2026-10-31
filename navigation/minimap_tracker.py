import cv2
import numpy as np
from typing import Tuple, Optional
from vision.vision import VisionEngine


class MinimapTracker:
    """Tracks player marker position and parses numerical map coordinates."""

    def __init__(self, vision_engine: VisionEngine) -> None:
        self.vision = vision_engine

    def locate_player_on_minimap(
        self,
        frame: np.ndarray,
        minimap_roi: Tuple[int, int, int, int],
    ) -> Optional[Tuple[int, int]]:
        """Locate the yellow minimap marker and return frame coordinates."""
        if (
            frame is None
            or not isinstance(frame, np.ndarray)
            or frame.size == 0
        ):
            return None

        if frame.ndim == 2:
            frame_bgr = cv2.cvtColor(
                frame,
                cv2.COLOR_GRAY2BGR,
            )

        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame_bgr = cv2.cvtColor(
                frame,
                cv2.COLOR_BGRA2BGR,
            )

        elif frame.ndim == 3 and frame.shape[2] == 3:
            frame_bgr = frame

        else:
            return None

        if len(minimap_roi) != 4:
            return None

        x, y, width, height = map(int, minimap_roi)

        frame_h, frame_w = frame_bgr.shape[:2]

        if (
            x < 0
            or y < 0
            or width <= 0
            or height <= 0
            or x + width > frame_w
            or y + height > frame_h
        ):
            return None

        crop = frame_bgr[
            y:y + height,
            x:x + width,
        ]

        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2HSV,
        )

        lower_marker = np.array(
            [20, 120, 120],
            dtype=np.uint8,
        )

        upper_marker = np.array(
            [40, 255, 255],
            dtype=np.uint8,
        )

        mask = cv2.inRange(
            hsv,
            lower_marker,
            upper_marker,
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        largest = max(
            contours,
            key=cv2.contourArea,
        )

        moments = cv2.moments(largest)

        if moments["m00"] <= 0:
            return None

        center_x = int(
            round(moments["m10"] / moments["m00"])
        ) + x

        center_y = int(
            round(moments["m01"] / moments["m00"])
        ) + y

        return center_x, center_y

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