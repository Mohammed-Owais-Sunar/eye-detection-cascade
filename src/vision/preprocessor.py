import cv2
import numpy as np


class FramePreprocessor:
    """Handles grayscale conversion and contrast enhancement."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def enhance_contrast(self, gray_roi: np.ndarray) -> np.ndarray:
        return self.clahe.apply(gray_roi)