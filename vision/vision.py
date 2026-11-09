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
        """Locate a keyword using EasyOCR and return its center in frame coordinates."""
        if (
            frame is None
            or not isinstance(frame, np.ndarray)
            or frame.size == 0
        ):
            return None

        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.ndim != 3 or frame.shape[2] != 3:
            return None

        target_img = frame
        offset_x = 0
        offset_y = 0

        if roi is not None:
            if len(roi) != 4:
                raise ValueError("roi must contain x, y, width, height")

            x, y, width, height = map(int, roi)
            frame_h, frame_w = frame.shape[:2]

            if (
                x < 0
                or y < 0
                or width <= 0
                or height <= 0
                or x + width > frame_w
                or y + height > frame_h
            ):
                return None

            target_img = frame[y:y + height, x:x + width]
            offset_x = x
            offset_y = y

        normalized_keywords = [
            str(keyword).casefold()
            for keyword in keywords
            if str(keyword).strip()
        ]

        if not normalized_keywords:
            return None

        processed_img = self.preprocess_for_ocr(target_img)

        if processed_img is None or processed_img.size == 0:
            return None

        # Resolve the lazy-loaded reader before acquiring the inference lock.
        # The reader property also acquires _ocr_lock during initialization.
        try:
            reader = self.reader

            with self._ocr_lock:
                ocr_results = reader.readtext(
                    processed_img,
                    paragraph=False,
                    detail=1,
                    contrast_ths=0.1,
                    adjust_contrast=0.5,
                )

        except Exception as exc:
            raise RuntimeError(
                "EasyOCR failed while locating a text anchor"
            ) from exc

        # Calculate the actual preprocessing scale rather than assuming
        # that both image dimensions always use an identical scale factor.
        scale_x = processed_img.shape[1] / target_img.shape[1]
        scale_y = processed_img.shape[0] / target_img.shape[0]

        if scale_x <= 0 or scale_y <= 0:
            return None

        for result in ocr_results:
            if not isinstance(result, (tuple, list)) or len(result) != 3:
                continue

            bbox, recognized_text, probability = result

            cleaned = str(recognized_text).strip()

            try:
                confidence = float(probability)
                points = np.asarray(bbox, dtype=np.float32)
            except (TypeError, ValueError):
                continue

            if (
                not np.isfinite(confidence)
                or confidence < min_confidence
                or not cleaned
            ):
                continue

            if not any(
                keyword in cleaned.casefold()
                for keyword in normalized_keywords
            ):
                continue

            if (
                points.ndim != 2
                or points.shape[0] == 0
                or points.shape[1] < 2
                or not np.all(np.isfinite(points))
            ):
                continue

            center_x = int(
                round(float(points[:, 0].mean()) / scale_x)
            ) + offset_x

            center_y = int(
                round(float(points[:, 1].mean()) / scale_y)
            ) + offset_y

            return (center_x, center_y), cleaned

        return None

    def read_thai_text(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
        min_confidence: float = 0.0,
    ) -> List[str]:
        """Recognize Thai and English text in a frame or a selected ROI."""
        if (
            frame is None
            or not isinstance(frame, np.ndarray)
            or frame.size == 0
        ):
            return []

        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.ndim != 3 or frame.shape[2] != 3:
            return []

        target_img = frame

        if roi is not None:
            if len(roi) != 4:
                raise ValueError("roi must contain x, y, width, height")

            x, y, width, height = map(int, roi)
            frame_h, frame_w = frame.shape[:2]

            if (
                x < 0
                or y < 0
                or width <= 0
                or height <= 0
                or x + width > frame_w
                or y + height > frame_h
            ):
                return []

            target_img = frame[y:y + height, x:x + width]

        processed_img = self.preprocess_for_ocr(target_img)

        if processed_img is None or processed_img.size == 0:
            return []

        try:
            reader = self.reader

            with self._ocr_lock:
                ocr_results = reader.readtext(
                    processed_img,
                    paragraph=False,
                    detail=1,
                    contrast_ths=0.1,
                    adjust_contrast=0.5,
                )

        except Exception as exc:
            raise RuntimeError(
                "EasyOCR failed while reading text"
            ) from exc

        recognized: List[str] = []

        for result in ocr_results:
            if not isinstance(result, (tuple, list)) or len(result) != 3:
                continue

            text = str(result[1]).strip()

            try:
                confidence = float(result[2])
            except (TypeError, ValueError):
                continue

            if (
                text
                and np.isfinite(confidence)
                and confidence >= min_confidence
            ):
                recognized.append(text)

        return recognized

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
    ) -> Optional[Tuple[int, int]]:
        """Detects top-right cutscene skip button using dual-pass OCR and icon color heuristics.
        Returns (x, y) tap coordinates if detected, otherwise None."""
        if frame is None or frame.size == 0:
            return None

        fh, fw = frame.shape[:2]
        target_img = frame
        offset_x, offset_y = 0, 0

        if roi:
            x, y, w, h = roi
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > fw or y + h > fh:
                return None
            target_img = frame[y : y + h, x : x + w]
            offset_x, offset_y = x, y

        # Pass 1: Direct Grayscale Thresholding for Small Thai Text ("แตะเพื่อข้าม", "ข้าม")
        gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY) if target_img.ndim == 3 else target_img
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        # Bypassing heavy CLAHE to preserve thin Thai diacritics
        try:
            with self._ocr_lock:
                results = self.reader.readtext(thresh, paragraph=False, detail=1)
                for bbox, text, prob in results:
                    clean_text = str(text).casefold().replace(" ", "")
                    if any(kw in clean_text for kw in ["ข้าม", "แตะเพื่อ", "skip", "pass"]):
                        points = np.asarray(bbox, dtype=np.float32)
                        cx = int(round(float(points[:, 0].mean()))) + offset_x
                        cy = int(round(float(points[:, 1].mean()))) + offset_y
                        return (cx, cy)
        except Exception:
            pass

        # Pass 2: Neon-Green Fast-Forward Chevron Icon Detection (⏩)
        if target_img.ndim == 3:
            hsv = cv2.cvtColor(target_img, cv2.COLOR_BGR2HSV)
            # Expanded HSV range to capture neon green/cyan chevron icons
            lower_green = np.array([30, 60, 90])
            upper_green = np.array([100, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)

            contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if 10 <= area <= 1000:
                    bx, by, bw, bh = cv2.boundingRect(c)
                    if 0.5 <= (float(bw) / float(bh)) <= 2.0:
                        center_x = offset_x + bx + (bw // 2)
                        center_y = offset_y + by + (bh // 2)
                        return (center_x, center_y)

        return None

    def detect_dialogue_or_cutscene(
        self,
        frame: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        """Secondary fail-safe: Detects active dialogue choices or NPC text banners when skip button is absent.
        Returns tap coordinates for the dialogue prompt or screen center."""
        if frame is None or frame.size == 0:
            return None

        fh, fw = frame.shape[:2]
        
        # Lower half region where NPC names, dialogues, and choices ('💬') appear
        dialogue_roi = (0, int(fh * 0.55), fw, int(fh * 0.40))
        x, y, w, h = dialogue_roi
        crop = frame[y : y + h, x : x + w]

        # Search for interactive dialogue options (marked with speech bubble icon '💬' or text questions)
        ocr_texts = self.read_thai_text(crop)
        if ocr_texts:
            anchor = self.find_text_anchor(crop, keywords=["?", "💬", "คือ", "อะไร", "ไป", "ตื่น"])
            if anchor:
                (local_x, local_y), _ = anchor
                return (local_x + x, local_y + y)

            # Fallback: Tap right-hand side dialogue selection area
            return (int(fw * 0.75), int(fh * 0.75))

        return None

    def preprocess_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """Adaptively rescales ROI crops and applies CLAHE enhancement for optimal EasyOCR text recognition."""
        if img is None or img.size == 0:
            return img

        h, w = img.shape[:2]
        
        # Target an optimal character height range (~150px height) to prevent matrix bloat
        target_h = 150.0
        if h < target_h:
            scale = min(target_h / float(h), 2.5)  # Cap maximum upscale factor at 2.5x
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            img_scaled = img

        gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY) if img_scaled.ndim == 3 else img_scaled
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size,
        )
        return clahe.apply(gray)
