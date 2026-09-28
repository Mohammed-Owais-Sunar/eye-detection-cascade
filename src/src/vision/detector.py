import time
from dataclasses import dataclass
import cv2
import numpy as np

from src.config.settings import VisionConfig
from src.vision.preprocessor import FramePreprocessor


@dataclass
class DetectionResult:
    annotated_frame: np.ndarray
    face_count: int
    eye_count: int
    latency_ms: float


class EyeDetector:
    """Two-stage Face and Eye detector using OpenCV Haar Cascades."""

    def __init__(self, config: VisionConfig | None = None):
        self.config = config or VisionConfig()
        self.preprocessor = FramePreprocessor()

        self.face_cascade = cv2.CascadeClassifier(str(self.config.face_cascade_path))
        self.eye_cascade = cv2.CascadeClassifier(str(self.config.eye_cascade_path))

        if self.face_cascade.empty():
            raise FileNotFoundError(f"Missing face model: {self.config.face_cascade_path}")
        if self.eye_cascade.empty():
            raise FileNotFoundError(f"Missing eye model: {self.config.eye_cascade_path}")

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        start_time = time.perf_counter()
        annotated = frame.copy()

        # Step 1: Downscale frame for fast inference
        h, w = frame.shape[:2]
        small_frame = cv2.resize(frame, (0, 0), fx=self.config.scale_factor, fy=self.config.scale_factor)
        gray = self.preprocessor.to_grayscale(small_frame)

        # Step 2: Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.config.face_scale_factor,
            minNeighbors=self.config.face_min_neighbors,
            minSize=self.config.face_min_size,
        )

        inv_scale = 1.0 / self.config.scale_factor
        total_eyes = 0

        # Step 3: Localize eyes inside each face ROI
        for (fx, fy, fw, fh) in faces:
            # Map face bounds back to original coordinate system
            orig_fx, orig_fy = int(fx * inv_scale), int(fy * inv_scale)
            orig_fw, orig_fh = int(fw * inv_scale), int(fh * inv_scale)

            cv2.rectangle(
                annotated,
                (orig_fx, orig_fy),
                (orig_fx + orig_fw, orig_fy + orig_fh),
                self.config.face_box_color,
                self.config.box_thickness,
            )

            # Restrict eye search to upper 60% of the face (removes mouth/nose false positives)
            eye_roi_height = int(fh * 0.60)
            face_roi_gray = gray[fy : fy + eye_roi_height, fx : fx + fw]
            
            # Apply CLAHE to mitigate glare from eyeglass lenses
            enhanced_face_roi = self.preprocessor.enhance_contrast(face_roi_gray)

            eyes = self.eye_cascade.detectMultiScale(
                enhanced_face_roi,
                scaleFactor=self.config.eye_scale_factor,
                minNeighbors=self.config.eye_min_neighbors,
                minSize=self.config.eye_min_size,
            )

            total_eyes += len(eyes)

            # Draw bounding boxes for detected eyes
            for (ex, ey, ew, eh) in eyes:
                orig_ex = int((fx + ex) * inv_scale)
                orig_ey = int((fy + ey) * inv_scale)
                orig_ew = int(ew * inv_scale)
                orig_eh = int(eh * inv_scale)

                cv2.rectangle(
                    annotated,
                    (orig_ex, orig_ey),
                    (orig_ex + orig_ew, orig_ey + orig_eh),
                    self.config.eye_box_color,
                    self.config.box_thickness,
                )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return DetectionResult(
            annotated_frame=annotated,
            face_count=len(faces),
            eye_count=total_eyes,
            latency_ms=round(latency_ms, 2),
        )