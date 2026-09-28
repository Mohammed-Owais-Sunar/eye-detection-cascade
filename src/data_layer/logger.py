import sqlite3
import time


class DetectionLogger:
    def __init__(self, db_path="analytics.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS ad_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                event_id INTEGER,
                event_name TEXT,
                location_tag TEXT,
                campaign TEXT,
                dwell_time REAL,
                engaged BOOLEAN,
                smiled BOOLEAN
            )""")

            # Upgrade databases created by older ADPULSE versions.
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(ad_analytics)")
            }
            if "event_id" not in columns:
                conn.execute("ALTER TABLE ad_analytics ADD COLUMN event_id INTEGER")
            if "event_name" not in columns:
                conn.execute("ALTER TABLE ad_analytics ADD COLUMN event_name TEXT")

            conn.execute("""CREATE TABLE IF NOT EXISTS ad_locations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                building TEXT DEFAULT '',
                active BOOLEAN DEFAULT 1
            )""")

            default_locations = [
                ("Library Entrance", "Library"),
                ("O Building", "Academic Block"),
                ("Canteen", "Student Center"),
                ("Hostel Gate", "Hostel"),
                ("Sports Complex", "Sports"),
            ]
            for location_name, building in default_locations:
                conn.execute(
                    "INSERT OR IGNORE INTO ad_locations (name, building, active) VALUES (?, ?, 1)",
                    (location_name, building),
                )

            conn.execute("""CREATE TABLE IF NOT EXISTS ad_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                campaign TEXT NOT NULL,
                active BOOLEAN DEFAULT 1,
                created_at REAL
            )""")

            conn.execute("""CREATE TABLE IF NOT EXISTS event_locations (
                event_id INTEGER NOT NULL,
                location_id INTEGER NOT NULL,
                PRIMARY KEY (event_id, location_id),
                FOREIGN KEY (event_id) REFERENCES ad_events(id),
                FOREIGN KEY (location_id) REFERENCES ad_locations(id)
            )""")

    def log_interaction(self, event_id, event_name, location, campaign, dwell_time, engaged, smiled):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO ad_analytics
                (timestamp, event_id, event_name, location_tag, campaign, dwell_time, engaged, smiled)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    time.time(),
                    event_id,
                    event_name,
                    location,
                    campaign,
                    dwell_time,
                    engaged,
                    smiled,
                ),
            )

    def create_location(self, name, building=""):
        name = name.strip()
        building = building.strip()
        if not name:
            raise ValueError("Location name is required.")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO ad_locations (name, building, active) VALUES (?, ?, 1)",
                (name, building),
            )

    def list_locations(self, active_only=True):
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT id, name, building, active FROM ad_locations"
            if active_only:
                query += " WHERE active = 1"
            query += " ORDER BY name"
            return conn.execute(query).fetchall()

    def create_event(self, name, start_date, end_date, campaign, location_ids):
        name = name.strip()
        campaign = campaign.strip()
        if not name:
            raise ValueError("Event name is required.")
        if start_date > end_date:
            raise ValueError("End date cannot be before start date.")
        if not location_ids:
            raise ValueError("Select at least one location.")

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """INSERT INTO ad_events
                (name, start_date, end_date, campaign, active, created_at)
                VALUES (?, ?, ?, ?, 1, ?)""",
                (name, start_date, end_date, campaign, time.time()),
            )
            event_id = cur.lastrowid
            conn.executemany(
                "INSERT INTO event_locations (event_id, location_id) VALUES (?, ?)",
                [(event_id, int(location_id)) for location_id in location_ids],
            )
            return event_id

    def list_events(self, active_only=False):
        with sqlite3.connect(self.db_path) as conn:
            query = """SELECT e.id, e.name, e.start_date, e.end_date, e.campaign, e.active,
                              GROUP_CONCAT(l.name, ' • ')
                       FROM ad_events e
                       LEFT JOIN event_locations el ON e.id = el.event_id
                       LEFT JOIN ad_locations l ON l.id = el.location_id"""
            if active_only:
                query += " WHERE e.active = 1"
            query += " GROUP BY e.id ORDER BY e.start_date DESC, e.id DESC"
            return conn.execute(query).fetchall()

    def get_event_locations(self, event_id):
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                """SELECT l.id, l.name, l.building
                   FROM ad_locations l
                   JOIN event_locations el ON el.location_id = l.id
                   WHERE el.event_id = ? AND l.active = 1
                   ORDER BY l.name""",
                (event_id,),
            ).fetchall()

    def toggle_event(self, event_id, active):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE ad_events SET active = ? WHERE id = ?",
                (1 if active else 0, event_id),
            )

    def toggle_location(self, location_id, active):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE ad_locations SET active = ? WHERE id = ?",
                (1 if active else 0, location_id),
            )
