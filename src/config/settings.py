from pathlib import Path
from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"


@dataclass(frozen=True)
class VisionConfig:
    face_cascade_path: Path = MODELS_DIR / "haarcascade_frontalface_default.xml"
    eye_cascade_path: Path = MODELS_DIR / "haarcascade_eye_tree_eyeglasses.xml"

    # Process a reduced frame for detection to keep real-time performance usable.
    scale_factor: float = 0.5
    face_scale_factor: float = 1.1
    face_min_neighbors: int = 5
    face_min_size: tuple[int, int] = (60, 60)

    # Eye detection is the more expensive stage. Run it every N frames and
    # reuse the latest result between detection frames.
    eye_detection_interval: int = 2
    eye_scale_factor: float = 1.08
    eye_min_neighbors: int = 4
    eye_min_size: tuple[int, int] = (15, 15)

    # Both eyes must be detected continuously for this many seconds.
    min_eyes_for_engagement: int = 2
    engagement_seconds: float = 5.0

    # Temporary face tracking settings for multi-person sessions.
    track_max_distance: int = 100
    track_timeout: float = 1.5

    face_box_color: tuple[int, int, int] = (255, 180, 50)
    eye_box_color: tuple[int, int, int] = (50, 220, 90)
    engaged_box_color: tuple[int, int, int] = (0, 255, 255)
    box_thickness: int = 2
