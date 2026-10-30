from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class VisionConfig:
    """Configuration parameters for OpenCV computer vision and EasyOCR processing."""

    template_matching_threshold: float = 0.75
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
    custom_adb_path: Optional[str] = (
        r"E:\Program Files\Netease\MuMuPlayer\nx_device\15.0\shell\adb.exe"
    )


@dataclass(frozen=True)
class BotConfig:
    """Operational regions of interest (ROI) and timing settings for instances."""

    quest_roi: Tuple[int, int, int, int] = (10, 200, 200, 150)  # (x, y, width, height)
    minimap_roi: Tuple[int, int, int, int] = (800, 20, 200, 200)
    joystick_center: Tuple[int, int] = (200, 500)
    loop_delay_sec: float = 0.5
