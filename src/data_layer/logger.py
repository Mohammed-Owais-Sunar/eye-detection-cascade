import sqlite3
import time
from pathlib import Path

class DetectionLogger:
    def __init__(self, db_path="analytics.db"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS ad_analytics (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp REAL,
                            location_tag TEXT,
                            campaign TEXT,
                            dwell_time REAL,
                            engaged BOOLEAN,
                            smiled BOOLEAN)''')
            
    def log_interaction(self, location, campaign, dwell_time, engaged, smiled):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO ad_analytics (timestamp, location_tag, campaign, dwell_time, engaged, smiled) VALUES (?, ?, ?, ?, ?, ?)",
                         (time.time(), location, campaign, dwell_time, engaged, smiled))