import cv2
import numpy as np
from typing import Optional, Tuple, List
import easyocr


class VisionEngine:
    """Handles template matching and Thai OCR image processing."""

    def __init__(self, use_gpu: bool = False) -> None:
        # Initialize EasyOCR with Thai and English language support
        self.reader = easyocr.Reader(["th", "en"], gpu=use_gpu)

    def match_template(
        self,
        frame: np.ndarray,
        template_path: str,
        threshold: float = 0.8,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[Tuple[int, int]]:
        """Finds center coordinates of a target image template within a frame."""
        template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        if template is None or frame is None:
            return None

        target_frame = frame
        offset_x, offset_y = 0, 0

        # Apply ROI (x, y, width, height) slicing if provided
        if roi:
            x, y, w, h = roi
            target_frame = frame[y : y + h, x : x + w]
            offset_x, offset_y = x, y

        result = cv2.matchTemplate(target_frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            t_h, t_w, _ = template.shape
            center_x = max_loc[0] + (t_w // 2) + offset_x
            center_y = max_loc[1] + (t_h // 2) + offset_y
            return (center_x, center_y)

        return None

    def preprocess_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """Applies grayscale conversion, contrast enlargement, and binarization for Thai text parsing."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Increase contrast via CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Otsu thresholding
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def read_thai_text(
        self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None
    ) -> List[str]:
        """Extracts Thai text strings from specified frame region."""
        target_img = frame
        if roi:
            x, y, w, h = roi
            target_img = frame[y : y + h, x : x + w]

        processed_img = self.preprocess_for_ocr(target_img)
        ocr_results = self.reader.readtext(processed_img, detail=0)
        return [text.strip() for text in ocr_results if text.strip()]