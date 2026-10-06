import time
from dataclasses import dataclass

import cv2
import numpy as np

from src.config.settings import VisionConfig
from src.vision.preprocessor import FramePreprocessor


@dataclass
class PersonDetection:
    track_id: int
    bbox: tuple[int, int, int, int]
    eye_count: int
    looking: bool
    dwell_time: float
    engaged: bool


@dataclass
class DetectionResult:
    annotated_frame: np.ndarray
    face_count: int
    eye_count: int
    engaged: bool
    latency_ms: float
    persons: list[PersonDetection]


class EyeDetector:
    """Multi-person face/eye detector using temporary spatial tracking only.

    Privacy mode deliberately avoids face signatures, embeddings, photographs,
    or biometric re-identification. Track IDs exist only in RAM for the
    current camera processor session.
    """

    def __init__(self, config: VisionConfig | None = None):
        self.config = config or VisionConfig()
        self.preprocessor = FramePreprocessor()
        self.face_cascade = cv2.CascadeClassifier(str(self.config.face_cascade_path))
        self.eye_cascade = cv2.CascadeClassifier(str(self.config.eye_cascade_path))

        if self.face_cascade.empty():
            raise FileNotFoundError(f"Missing face model: {self.config.face_cascade_path}")
        if self.eye_cascade.empty():
            raise FileNotFoundError(f"Missing eye model: {self.config.eye_cascade_path}")

        self.next_track_id = 1
        self.tracks: dict[int, dict] = {}
        self.frame_index = 0

    def _new_track(self, center: tuple[int, int], now: float) -> int:
        track_id = self.next_track_id
        self.next_track_id += 1
        self.tracks[track_id] = {
            "center": center,
            "last_seen": now,
            "looking_started": None,
            "dwell_time": 0.0,
            "engaged": False,
            "eye_count": 0,
            "looking": False,
            "eye_boxes": [],
        }
        return track_id

    def _match_track(
        self,
        center: tuple[int, int],
        now: float,
        used: set[int],
    ) -> int:
        best_id = None
        best_distance = float("inf")

        for track_id, track in self.tracks.items():
            if track_id in used:
                continue
            if now - track["last_seen"] > self.config.track_timeout:
                continue

            tx, ty = track["center"]
            distance = ((center[0] - tx) ** 2 + (center[1] - ty) ** 2) ** 0.5

            if distance < best_distance and distance <= self.config.track_max_distance:
                best_distance = distance
                best_id = track_id

        return best_id if best_id is not None else self._new_track(center, now)

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        start_time = time.perf_counter()
        now = time.time()
        self.frame_index += 1
        annotated = frame.copy()

        small_frame = cv2.resize(
            frame,
            (0, 0),
            fx=self.config.scale_factor,
            fy=self.config.scale_factor,
        )
        gray = self.preprocessor.to_grayscale(small_frame)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.config.face_scale_factor,
            minNeighbors=self.config.face_min_neighbors,
            minSize=self.config.face_min_size,
        )

        inv_scale = 1.0 / self.config.scale_factor
        detections = []
        used_tracks: set[int] = set()
        total_eyes = 0
        run_eye_detection = self.frame_index % self.config.eye_detection_interval == 0

        for (fx, fy, fw, fh) in faces:
            orig_fx, orig_fy = int(fx * inv_scale), int(fy * inv_scale)
            orig_fw, orig_fh = int(fw * inv_scale), int(fh * inv_scale)
            center = (orig_fx + orig_fw // 2, orig_fy + orig_fh // 2)

            track_id = self._match_track(center, now, used_tracks)
            used_tracks.add(track_id)
            track = self.tracks[track_id]
            track["center"] = center
            track["last_seen"] = now

            if run_eye_detection:
                eye_roi_height = int(fh * 0.60)
                face_roi_gray = gray[fy:fy + eye_roi_height, fx:fx + fw]
                enhanced_face_roi = self.preprocessor.enhance_contrast(face_roi_gray)

                eyes = self.eye_cascade.detectMultiScale(
                    enhanced_face_roi,
                    scaleFactor=self.config.eye_scale_factor,
                    minNeighbors=self.config.eye_min_neighbors,
                    minSize=self.config.eye_min_size,
                )

                track["eye_count"] = len(eyes)
                track["eye_boxes"] = list(eyes)

            eye_count = track["eye_count"]
            total_eyes += eye_count
            looking = eye_count >= self.config.min_eyes_for_engagement
            track["looking"] = looking

            if looking:
                if track["looking_started"] is None:
                    track["looking_started"] = now
                track["dwell_time"] = now - track["looking_started"]

                if track["dwell_time"] >= self.config.engagement_seconds:
                    track["engaged"] = True
            else:
                track["looking_started"] = None
                track["dwell_time"] = 0.0

            box_color = (
                self.config.engaged_box_color
                if track["engaged"]
                else self.config.face_box_color
            )

            cv2.rectangle(
                annotated,
                (orig_fx, orig_fy),
                (orig_fx + orig_fw, orig_fy + orig_fh),
                box_color,
                self.config.box_thickness,
            )

            # Reuse the latest eye detections on skipped frames. This avoids
            # running the expensive eye cascade on every incoming frame while
            # keeping the visual overlay continuous.
            for (ex, ey, ew, eh) in track["eye_boxes"]:
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

            status = "ENGAGED" if track["engaged"] else (
                "LOOKING" if looking else "SEEN"
            )
            label = f"Person {track_id} | {track['dwell_time']:.1f}s | {status}"

            cv2.putText(
                annotated,
                label,
                (orig_fx, max(orig_fy - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2,
                cv2.LINE_AA,
            )

            detections.append(
                PersonDetection(
                    track_id=track_id,
                    bbox=(orig_fx, orig_fy, orig_fw, orig_fh),
                    eye_count=eye_count,
                    looking=looking,
                    dwell_time=track["dwell_time"],
                    engaged=track["engaged"],
                )
            )

        self.tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if now - track["last_seen"] <= self.config.track_timeout
        }

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return DetectionResult(
            annotated_frame=annotated,
            face_count=len(faces),
            eye_count=total_eyes,
            engaged=any(person.engaged for person in detections),
            latency_ms=round(latency_ms, 2),
            persons=detections,
        )
