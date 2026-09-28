import queue
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "detections.db"

@dataclass
class LogEntry:
    timestamp: datetime
    location: str
    face_count: int
    eyes_detected: int
    inference_latency_ms: float

class DetectionLogger:
    """Non-blocking asynchronous SQLite logger with location tagging."""
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.queue: queue.Queue = queue.Queue()
        self.running = True
        self._init_db()
        self.worker = threading.Thread(target=self._process_queue, daemon=True)
        self.worker.start()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS detection_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    location TEXT NOT NULL,
                    face_count INTEGER NOT NULL,
                    eyes_detected INTEGER NOT NULL,
                    inference_latency_ms REAL NOT NULL
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_log_timestamp ON detection_logs (timestamp)")
            conn.commit()

    def log_event(self, location: str, face_count: int, eyes_detected: int, latency_ms: float) -> None:
        entry = LogEntry(
            timestamp=datetime.utcnow(),
            location=location,
            face_count=face_count,
            eyes_detected=eyes_detected,
            inference_latency_ms=latency_ms,
        )
        self.queue.put(entry)

    def get_location_analytics(self) -> pd.DataFrame:
        """Aggregates data to calculate the Engagement Rate per location."""
        query = """
            SELECT 
                location,
                SUM(face_count) as total_foot_traffic,
                SUM(eyes_detected) as total_engagement,
                COUNT(id) as tracking_cycles
            FROM detection_logs
            GROUP BY location
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn)
                if not df.empty:
                    df["engagement_rate_%"] = ((df["total_engagement"] / (df["total_foot_traffic"] * 2)) * 100).fillna(0).round(1)
                return df
        except sqlite3.Error:
            return pd.DataFrame()

    def _process_queue(self) -> None:
        while self.running:
            try:
                entry: LogEntry = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO detection_logs 
                        (timestamp, location, face_count, eyes_detected, inference_latency_ms)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            entry.timestamp.isoformat(),
                            entry.location,
                            entry.face_count,
                            entry.eyes_detected,
                            entry.inference_latency_ms,
                        ),
                    )
                    conn.commit()
            except sqlite3.Error:
                pass
            finally:
                self.queue.task_done()

    def shutdown(self) -> None:
        self.running = False
        if self.worker.is_alive():
            self.worker.join(timeout=2.0)