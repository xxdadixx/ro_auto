from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class VisionConfig:
    """Configuration parameters for OpenCV computer vision and EasyOCR processing."""

    template_matching_threshold: float = 0.70
    ocr_languages: Tuple[str, ...] = ("th", "en")
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)


@dataclass(frozen=True)
class ADBConfig:
    """Configuration for ADB connection and emulator scanning parameters."""

    host: str = "127.0.0.1"
    default_port: int = 7555
    start_port: int = 16384
    port_stride: int = 32
    max_instances: int = 32
    connection_timeout: float = 0.3

    # Custom ADB path
    custom_adb_path: Optional[str] = None


@dataclass(frozen=True)
class BotConfig:
    """Operational regions of interest defined as relative percentage bounds (xmin, ymin, width, height) in range [0.0, 1.0]."""

    # Relative ROIs: (x_ratio, y_ratio, w_ratio, h_ratio)
    quest_hud_rel_roi: Tuple[float, float, float, float] = (0.0, 0.12, 0.45, 0.55)  # Left 45% HUD region
    cutscene_rel_roi: Tuple[float, float, float, float] = (0.75, 0.0, 0.25, 0.15)    # Top-right 25% HUD region
    quest_gui_go_rel_roi: Tuple[float, float, float, float] = (0.50, 0.60, 0.50, 0.38) # Bottom-right 50% GUI region
    
    # Relative Fallback Tap Points (x_ratio, y_ratio)
    top_main_quest_rel_point: Tuple[float, float] = (0.30, 0.28)
    go_button_rel_point: Tuple[float, float] = (0.70, 0.86)
    skip_button_rel_point: Tuple[float, float] = (0.95, 0.05)

    loop_delay_sec: float = 0.5