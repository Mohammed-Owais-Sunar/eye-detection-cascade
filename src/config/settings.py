from pathlib import Path
from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"


@dataclass(frozen=True)
class VisionConfig:
    face_cascade_path: Path = MODELS_DIR / "haarcascade_frontalface_default.xml"
    eye_cascade_path: Path = MODELS_DIR / "haarcascade_eye_tree_eyeglasses.xml"

    scale_factor: float = 0.5
    face_scale_factor: float = 1.1
    face_min_neighbors: int = 5
    face_min_size: tuple[int, int] = (60, 60)

    eye_scale_factor: float = 1.08
    eye_min_neighbors: int = 4
    eye_min_size: tuple[int, int] = (15, 15)

    # "Engaged" = both eyes detected, not just one
    min_eyes_for_engagement: int = 2

    face_box_color: tuple[int, int, int] = (255, 180, 50)
    eye_box_color: tuple[int, int, int] = (50, 220, 90)
    box_thickness: int = 2