import os
import threading
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
import easyocr
from config import VisionConfig


class VisionEngine:
    """Handles template matching and Thai OCR image processing with lazy model loading."""

    def __init__(self, use_gpu: bool = False, config: VisionConfig = VisionConfig()) -> None:
        self.config = config
        self.use_gpu = use_gpu
        self._reader: Optional[easyocr.Reader] = None  # Lazy PyTorch initialization
        self._ocr_lock = threading.Lock()
        self._template_cache: Dict[str, np.ndarray] = {}

    @property
    def reader(self) -> easyocr.Reader:
        """Instantiates EasyOCR/PyTorch only on demand, saving ~700 MB RAM at startup."""
        if self._reader is None:
            self._reader = easyocr.Reader(list(self.config.ocr_languages), gpu=self.use_gpu)
        return self._reader

    def _get_template(self, template_path: str) -> Optional[np.ndarray]:
        """Loads and caches template images in memory to prevent repeated disk I/O."""
        if template_path not in self._template_cache:
            if not os.path.exists(template_path):
                return None
            template = cv2.imread(template_path, cv2.IMREAD_COLOR)
            if template is None:
                return None
            self._template_cache[template_path] = template
        return self._template_cache[template_path]

    def match_template(
        self,
        frame: np.ndarray,
        template_path: str,
        threshold: Optional[float] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[Tuple[int, int]]:
        """Finds center coordinates of a target image template within a frame."""
        if frame is None or frame.size == 0:
            return None

        template = self._get_template(template_path)
        if template is None:
            return None

        effective_threshold = (
            threshold if threshold is not None else self.config.template_matching_threshold
        )
        target_frame = frame
        offset_x, offset_y = 0, 0

        if roi:
            x, y, w, h = roi
            fh, fw = frame.shape[:2]
            if x + w > fw or y + h > fh:
                return None
            target_frame = frame[y : y + h, x : x + w]
            offset_x, offset_y = x, y

        if target_frame.shape[0] < template.shape[0] or target_frame.shape[1] < template.shape[1]:
            return None

        result = cv2.matchTemplate(target_frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= effective_threshold:
            t_h, t_w, _ = template.shape
            center_x = max_loc[0] + (t_w // 2) + offset_x
            center_y = max_loc[1] + (t_h // 2) + offset_y
            return (center_x, center_y)

        return None

    def preprocess_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """Applies grayscale conversion, CLAHE contrast enhancement, and Otsu thresholding."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size,
        )
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def read_thai_text(
        self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None
    ) -> List[str]:
        """Extracts text strings from specified frame region with thread synchronization."""
        if frame is None or frame.size == 0:
            return []

        target_img = frame
        if roi:
            x, y, w, h = roi
            fh, fw = frame.shape[:2]
            if x + w > fw or y + h > fh:
                return []
            target_img = frame[y : y + h, x : x + w]

        processed_img = self.preprocess_for_ocr(target_img)

        # Synchronize OCR calls and trigger lazy PyTorch load via self.reader
        with self._ocr_lock:
            ocr_results = self.reader.readtext(processed_img, detail=0)

        return [text.strip() for text in ocr_results if text.strip()]