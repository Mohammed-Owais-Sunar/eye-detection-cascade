import streamlit as st
import cv2
import pandas as pd
import sqlite3
import time
import sys
from pathlib import Path

# Pathing fix so Streamlit Cloud can find the src modules
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

from src.vision.detector import EyeDetector
from src.data_layer.logger import DetectionLogger

st.set_page_config(
    page_title="Campus Ad Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styles ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.block-container { padding-top: 2rem; max-width: 1400px; }

h1 {
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}

/* Card look for metric blocks */
div[data-testid="stMetric"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 1rem;
}
div[data-testid="stMetricValue"] { color: #38bdf8; font-weight: 700; }

/* Video frame container */
div[data-testid="stImage"] img {
    border-radius: 14px;
    border: 1px solid #334155;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0b1220;
    border-right: 1px solid #1e293b;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    color: #0f172a;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.5rem 1.25rem;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(56, 189, 248, 0.4);
}

/* Hide chrome */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------- State ----------
if "running" not in st.session_state:
    st.session_state.running = False
if "session_impressions" not in st.session_state:
    st.session_state.session_impressions = 0
if "session_engagements" not in st.session_state:
    st.session_state.session_engagements = 0
if "last_face_seen" not in st.session_state:
    st.session_state.last_face_seen = 0.0
if "seen_track_ids" not in st.session_state:
    st.session_state.seen_track_ids = set()
if "engaged_track_ids" not in st.session_state:
    st.session_state.engaged_track_ids = set()

logger = DetectionLogger()

# ---------- Header ----------
st.title("🎯 Spatial Ad Analytics Engine")
st.caption("Edge-to-cloud pipeline for real-time physical advertising conversion metrics.")

tab_edge, tab_cloud = st.tabs(["🔴 Edge Sensor (Local)", "📈 Cloud Analytics (Global)"])

# =========================================================
# EDGE SENSOR TAB
# =========================================================
with tab_edge:
    left, right = st.columns([3, 1])

    with right:
        st.markdown("### Deployment")
        location = st.selectbox("Campus Zone", [
            "Library Entrance", "Tech Block", "Canteen", "Hostel Gate", "Sports Complex"
        ])
        campaign = st.selectbox("Active Campaign", [
            "Tech Symposium Ad", "Hackathon Poster", "Campus Election", "Club Recruitment"
        ])

        st.markdown("### Sensor control")
        if not st.session_state.running:
            if st.button("▶ Activate Edge Camera", use_container_width=True):
                st.session_state.running = True
                st.rerun()
        else:
            if st.button("⏹ Stop Sensor", use_container_width=True):
                st.session_state.running = False
                st.rerun()

        st.markdown("### Session stats")
        imp_ph = st.empty()
        eng_ph = st.empty()
        rate_ph = st.empty()

    with left:
        stframe = st.empty()
        fps_ph = st.empty()

    if st.session_state.running:
        detector = EyeDetector()
        cap = cv2.VideoCapture(0)
        prev_time = time.time()

        try:
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera unavailable.")
                    st.session_state.running = False
                    break

                frame = cv2.flip(frame, 1)
                result = detector.process_frame(frame)

                # FPS
                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                # Count each tracked person once as an impression.
                for person in result.persons:
                    if person.track_id not in st.session_state.seen_track_ids:
                        st.session_state.seen_track_ids.add(person.track_id)
                        st.session_state.session_impressions += 1

                    # Engagement is counted only once after continuous 5-second looking.
                    if (
                        person.engaged
                        and person.track_id not in st.session_state.engaged_track_ids
                    ):
                        st.session_state.engaged_track_ids.add(person.track_id)
                        st.session_state.session_engagements += 1
                        logger.log_interaction(
                            location,
                            campaign,
                            person.dwell_time,
                            True,
                            False,
                        )

                if result.face_count > 0:
                    st.session_state.last_face_seen = now

                # Render
                stframe.image(
                    cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )
                fps_ph.metric("FPS", f"{fps:.1f}")

                imp_ph.metric("Impressions (session)", st.session_state.session_impressions)
                eng_ph.metric("Engagements (session)", st.session_state.session_engagements)
                rate = (
                    st.session_state.session_engagements
                    / max(st.session_state.session_impressions, 1)
                    * 100
                )
                rate_ph.metric("Engagement Rate", f"{rate:.1f}%")
        finally:
            cap.release()

    else:
        stframe.info("Sensor is idle. Click **Activate Edge Camera** to begin collecting impressions.")
        imp_ph.metric("Impressions (session)", st.session_state.session_impressions)
        eng_ph.metric("Engagements (session)", st.session_state.session_engagements)

# =========================================================
# CLOUD ANALYTICS TAB
# =========================================================
with tab_cloud:
    st.markdown("### Location A/B Testing & Engagement Metrics")

    try:
        conn = sqlite3.connect("analytics.db")
        df = pd.read_sql_query("SELECT * FROM ad_analytics", conn)
        conn.close()

        if df.empty:
            st.info("No sensor data logged yet. Run the Edge Sensor to collect initial data.")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

            # KPI row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Impressions", len(df))
            c2.metric("Engagements", int(df["engaged"].sum()))
            c3.metric("Engagement Rate", f"{df['engaged'].mean() * 100:.1f}%")
            c4.metric(
                "Avg Dwell (s)",
                f"{df['dwell_time'].mean():.1f}" if "dwell_time" in df else "—",
            )

            # Filters
            st.markdown("### Filters")
            f1, f2 = st.columns(2)
            loc_filter = f1.multiselect(
                "Locations", sorted(df["location_tag"].unique()),
                default=sorted(df["location_tag"].unique()),
            )
            camp_filter = f2.multiselect(
                "Campaigns", sorted(df["campaign"].unique()),
                default=sorted(df["campaign"].unique()),
            )
            filtered = df[df["location_tag"].isin(loc_filter) & df["campaign"].isin(camp_filter)]

            # Charts
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Engagement Rate by Location")
                loc_stats = filtered.groupby("location_tag")["engaged"].mean() * 100
                st.bar_chart(loc_stats)

            with col_b:
                st.subheader("Engagement Rate by Campaign")
                camp_stats = filtered.groupby("campaign")["engaged"].mean() * 100
                st.bar_chart(camp_stats)

            st.subheader("Impressions over Time")
            filtered_time = filtered.set_index("timestamp")
            st.line_chart(filtered_time["engaged"].resample("1min").count())

            st.subheader("Raw Telemetry Logs")
            st.dataframe(filtered.tail(50), use_container_width=True)

            # Export
            st.download_button(
                "⬇ Download CSV",
                filtered.reset_index().to_csv(index=False),
                "ad_analytics.csv",
                "text/csv",
            )

    except Exception as e:
        st.warning(f"Database uninitialized or unreadable: {e}")