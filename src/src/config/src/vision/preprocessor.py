import cv2
import numpy as np


class FramePreprocessor:
    """Handles image normalization, grayscale conversion, and contrast enhancement."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        """Converts an RGB/BGR frame to single-channel 8-bit grayscale."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def enhance_contrast(self, gray_roi: np.ndarray) -> np.ndarray:
        """Applies CLAHE to suppress specular reflections and balance dark shadows."""
        return self.clahe.apply(gray_roi)