import os
import re
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
    """High-precision computer vision and Thai EasyOCR/Tesseract engine with sentence reconstruction."""

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
        """Instantiates EasyOCR/PyTorch lazily on demand."""
        if self._reader is None:
            self._reader = easyocr.Reader(
                list(self.config.ocr_languages), gpu=self.use_gpu
            )
        return self._reader

    def _get_template(self, template_path: str) -> Optional[np.ndarray]:
        """Loads and caches template images with thread safety."""
        with self._cache_lock:
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
            threshold
            if threshold is not None
            else self.config.template_matching_threshold
        )
        target_frame = frame
        offset_x, offset_y = 0, 0

        if roi:
            x, y, w, h = roi
            fh, fw = frame.shape[:2]
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > fw or y + h > fh:
                return None
            target_frame = frame[y : y + h, x : x + w]
            offset_x, offset_y = x, y

        if (
            target_frame.shape[0] < template.shape[0]
            or target_frame.shape[1] < template.shape[1]
        ):
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
        """Upscales crop 2.5x and applies non-destructive CLAHE enhancement without ruining Thai glyphs."""
        h, w = img.shape[:2]

        # 1. Bicubic Upscaling 2.5x to preserve fine stroke structures for tone marks
        upscaled = cv2.resize(
            img, (int(w * 2.5), int(h * 2.5)), interpolation=cv2.INTER_CUBIC
        )

        # 2. Convert to Grayscale
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

        # 3. Apply CLAHE contrast enhancement (preserves upper/lower Thai glyph tones)
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size,
        )
        enhanced = clahe.apply(gray)
        return enhanced

    def match_template_matrix(
        self,
        frame: np.ndarray,
        template: np.ndarray,
        threshold: Optional[float] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[Tuple[int, int]]:
        """Finds center coordinates of an in-memory template matrix within a target frame."""
        if frame is None or frame.size == 0 or template is None or template.size == 0:
            return None

        effective_threshold = (
            threshold
            if threshold is not None
            else self.config.template_matching_threshold
        )
        target_frame = frame
        offset_x, offset_y = 0, 0

        if roi:
            x, y, w, h = roi
            fh, fw = frame.shape[:2]
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > fw or y + h > fh:
                return None
            target_frame = frame[y : y + h, x : x + w]
            offset_x, offset_y = x, y

        if (
            target_frame.shape[0] < template.shape[0]
            or target_frame.shape[1] < template.shape[1]
        ):
            return None

        result = cv2.matchTemplate(target_frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= effective_threshold:
            t_h, t_w = template.shape[:2]
            center_x = max_loc[0] + (t_w // 2) + offset_x
            center_y = max_loc[1] + (t_h // 2) + offset_y
            return (center_x, center_y)

        return None

    def read_thai_text(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
        paragraph: bool = True,
    ) -> List[str]:
        """Extracts complete Thai sentences from image regions using dual-pass EasyOCR and PyTesseract."""
        if frame is None or frame.size == 0:
            return []

        target_img = frame
        if roi:
            x, y, w, h = roi
            fh, fw = frame.shape[:2]
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > fw or y + h > fh:
                return []
            target_img = frame[y : y + h, x : x + w]

        processed_img = self.preprocess_for_ocr(target_img)

        results_list: List[str] = []

        # Pass 1: EasyOCR with Paragraph Sentence Reconstruction
        with self._ocr_lock:
            try:
                ocr_results = self.reader.readtext(
                    processed_img,
                    paragraph=paragraph,
                    detail=0,
                    contrast_ths=0.1,
                    adjust_contrast=0.5,
                )
                for text in ocr_results:
                    cleaned = text.strip()
                    if cleaned:
                        results_list.append(cleaned)
            except Exception:
                pass

        if results_list:
            return results_list

        # Pass 2: Fallback to Raw Image EasyOCR without preprocessing
        with self._ocr_lock:
            try:
                raw_results = self.reader.readtext(
                    target_img, paragraph=paragraph, detail=0
                )
                for text in raw_results:
                    cleaned = text.strip()
                    if cleaned:
                        results_list.append(cleaned)
            except Exception:
                pass

        if results_list:
            return results_list

        # Pass 3: Tesseract Fallback (if installed)
        if HAS_PYTESSERACT:
            try:
                tess_text = pytesseract.image_to_string(
                    processed_img, lang="tha+eng", config="--psm 6"
                )
                lines = [
                    line.strip() for line in tess_text.splitlines() if line.strip()
                ]
                if lines:
                    return lines
            except Exception:
                pass

        return []
