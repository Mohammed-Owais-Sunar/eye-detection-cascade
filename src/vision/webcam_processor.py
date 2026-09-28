import threading
import time

import av
import cv2
from streamlit_webrtc import VideoProcessorBase

from src.data_layer.logger import DetectionLogger
from src.vision.detector import EyeDetector


class AdPulseVideoProcessor(VideoProcessorBase):
    """WebRTC processor: frames are analyzed in memory and never written to disk."""

    def __init__(self, location: str, campaign: str, db_path: str):
        self.location = location
        self.campaign = campaign
        self.logger = DetectionLogger(db_path)
        self.detector = EyeDetector()
        self.lock = threading.Lock()

        self.started_at = time.time()
        self.last_frame_at = time.time()
        self.frames = 0
        self.last_latency_ms = 0.0
        self.current_fps = 0.0

        self.seen_track_ids = set()
        self.engaged_track_ids = set()
        self.logged_track_ids = set()
        self.peak_people = 0
        self.peak_looking = 0
        self.latest_persons = []

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        result = self.detector.process_frame(image)

        now = time.time()
        with self.lock:
            self.frames += 1
            self.last_latency_ms = result.latency_ms
            elapsed = max(now - self.started_at, 1e-6)
            self.current_fps = self.frames / elapsed
            self.last_frame_at = now

            for person in result.persons:
                self.seen_track_ids.add(person.track_id)

                if person.engaged:
                    self.engaged_track_ids.add(person.track_id)
                    if person.track_id not in self.logged_track_ids:
                        self.logged_track_ids.add(person.track_id)
                        self.logger.log_interaction(
                            self.location,
                            self.campaign,
                            person.dwell_time,
                            True,
                            False,
                        )

            looking = sum(1 for person in result.persons if person.looking)
            self.peak_people = max(self.peak_people, len(result.persons))
            self.peak_looking = max(self.peak_looking, looking)
            self.latest_persons = list(result.persons)

        return av.VideoFrame.from_ndarray(result.annotated_frame, format="bgr24")

    def snapshot(self):
        with self.lock:
            looking = sum(1 for person in self.latest_persons if person.looking)
            engaged = sum(1 for person in self.latest_persons if person.engaged)
            tracks = []

            for person in self.latest_persons:
                state = "ENGAGED" if person.engaged else (
                    "LOOKING" if person.looking else "SEEN"
                )
                css_class = "track engaged" if person.engaged else "track"
                tracks.append(
                    f'<div class="{css_class}">'
                    f'P{person.track_id} / {state}'
                    f'<span class="time">{person.dwell_time:.1f}s</span>'
                    f'</div>'
                )

            return {
                "unique_people": len(self.seen_track_ids),
                "engagements": len(self.engaged_track_ids),
                "looking_now": looking,
                "engaged_now": engaged,
                "peak_people": self.peak_people,
                "peak_looking": self.peak_looking,
                "latency_ms": self.last_latency_ms,
                "fps": self.current_fps,
                "tracks_html": "".join(tracks),
            }
