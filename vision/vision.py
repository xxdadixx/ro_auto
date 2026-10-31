import os
import threading
from typing import Dict, List, Optional, Tuple
import cv2
import easyocr
import numpy as np

try:
    import pytesseract

    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False

from config import VisionConfig


class VisionEngine:
    """High-precision dynamic computer vision engine featuring OCR text anchor extraction and multi-scale template matching."""

    def __init__(
        self, use_gpu: bool = False, config: VisionConfig = VisionConfig()
    ) -> None:
        self.config = config
        self.use_gpu = use_gpu
        self._reader: Optional[easyocr.Reader] = None
        self._ocr_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._template_cache: Dict[str, np.ndarray] = {}

    @property
    def reader(self) -> easyocr.Reader:
        """Instantiates EasyOCR/PyTorch lazily on demand with thread-safe double-checked locking."""
        if self._reader is None:
            with self._ocr_lock:
                if self._reader is None:
                    self._reader = easyocr.Reader(
                        list(self.config.ocr_languages), gpu=self.use_gpu
                    )
        return self._reader

    def find_text_anchor(
        self,
        frame: np.ndarray,
        keywords: List[str],
        roi: Optional[Tuple[int, int, int, int]] = None,
        min_confidence: float = 0.35,
    ) -> Optional[Tuple[Tuple[int, int], str]]:
        """Dynamically locates text keywords on screen using OCR and returns exact bounding-box centroid coordinates."""
        if frame is None or frame.size == 0 or not keywords:
            return None

        target_img = frame
        offset_x, offset_y = 0, 0
        if roi:
            x, y, w, h = roi
            fh, fw = frame.shape[:2]
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > fw or y + h > fh:
                return None
            target_img = frame[y : y + h, x : x + w]
            offset_x, offset_y = x, y

        processed_img = self.preprocess_for_ocr(target_img)

        with self._ocr_lock:
            try:
                ocr_results = self.reader.readtext(
                    processed_img,
                    paragraph=False,
                    detail=1,
                    contrast_ths=0.1,
                    adjust_contrast=0.5,
                )
                for bbox, text, prob in ocr_results:
                    if prob < min_confidence:
                        continue
                    cleaned = text.strip()
                    for kw in keywords:
                        if kw.lower() in cleaned.lower():
                            # Calculate bounding box centroid inside upscaled image
                            pts = np.array(bbox, dtype=np.float32)
                            # Bounding box is extracted from 2.5x upscaled processed_img
                            cx_upscaled = float(np.mean(pts[:, 0]))
                            cy_upscaled = float(np.mean(pts[:, 1]))

                            # Convert coordinates back to original frame space
                            real_cx = int(cx_upscaled / 2.5) + offset_x
                            real_cy = int(cy_upscaled / 2.5) + offset_y
                            return ((real_cx, real_cy), cleaned)
            except Exception:
                pass

        return None

    def match_template_multiscale(
        self,
        frame: np.ndarray,
        template: np.ndarray,
        threshold: float = 0.70,
        roi: Optional[Tuple[int, int, int, int]] = None,
        scales: Tuple[float, ...] = (0.75, 0.88, 1.0, 1.15, 1.30),
    ) -> Optional[Tuple[int, int]]:
        """Finds center coordinates of an in-memory template matrix using scale-invariant pyramid matching."""
        if frame is None or frame.size == 0 or template is None or template.size == 0:
            return None

        # Normalize channels to 3-channel BGR
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if len(template.shape) == 3 and template.shape[2] == 4:
            template = cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)

        target_frame = frame
        offset_x, offset_y = 0, 0
        if roi:
            x, y, w, h = roi
            fh, fw = frame.shape[:2]
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > fw or y + h > fh:
                return None
            target_frame = frame[y : y + h, x : x + w]
            offset_x, offset_y = x, y

        best_max_val = -1.0
        best_center = None

        t_h_orig, t_w_orig = template.shape[:2]

        for scale in scales:
            sw = int(t_w_orig * scale)
            sh = int(t_h_orig * scale)

            if (
                sw <= 0
                or sh <= 0
                or target_frame.shape[0] < sh
                or target_frame.shape[1] < sw
            ):
                continue

            resized_template = cv2.resize(
                template, (sw, sh), interpolation=cv2.INTER_AREA
            )

            try:
                result = cv2.matchTemplate(
                    target_frame, resized_template, cv2.TM_CCOEFF_NORMED
                )
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                if max_val > best_max_val and max_val >= threshold:
                    best_max_val = max_val
                    center_x = max_loc[0] + (sw // 2) + offset_x
                    center_y = max_loc[1] + (sh // 2) + offset_y
                    best_center = (center_x, center_y)
            except cv2.Error:
                continue

        return best_center

    def detect_cutscene_skip_button(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> bool:
        """Strict visual recognition for top-right cutscene skip button using structural feature analysis."""
        if frame is None or frame.size == 0:
            return False

        target_img = frame
        if roi:
            x, y, w, h = roi
            fh, fw = frame.shape[:2]
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > fw or y + h > fh:
                return False
            target_img = frame[y : y + h, x : x + w]

        gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(target_img, cv2.COLOR_BGR2HSV)

        lower_green = np.array([35, 140, 140])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        contours, _ = cv2.findContours(
            green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for c in contours:
            area = cv2.contourArea(c)
            if 60 <= area <= 1500:
                bx, by, bw, bh = cv2.boundingRect(c)
                aspect_ratio = float(bw) / float(bh)
                if 0.7 <= aspect_ratio <= 1.4:
                    green_crop_gray = gray[by : by + bh, bx : bx + bw]
                    if green_crop_gray.size > 0:
                        _, white_mask = cv2.threshold(
                            green_crop_gray, 200, 255, cv2.THRESH_BINARY
                        )
                        white_pixels = cv2.countNonZero(white_mask)
                        white_ratio = white_pixels / float(bw * bh)

                        if 0.12 <= white_ratio <= 0.55:
                            return True

        return False

    def preprocess_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """Upscales crop 2.5x and applies CLAHE enhancement without ruining Thai glyphs."""
        h, w = img.shape[:2]
        upscaled = cv2.resize(
            img, (int(w * 2.5), int(h * 2.5)), interpolation=cv2.INTER_CUBIC
        )
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size,
        )
        return clahe.apply(gray)
