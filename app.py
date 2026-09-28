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

# ---------- Visual system ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
:root {color-scheme:dark}
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {
  background: radial-gradient(ellipse 75% 55% at 78% -10%,rgba(99,102,241,.19),transparent 75%),
              radial-gradient(ellipse 60% 45% at 5% 30%,rgba(6,182,212,.09),transparent 80%),#080d1b;
  color:#e8efff;font-family:'DM Sans',sans-serif;
}
[data-testid="stHeader"] {background:transparent}
.block-container {max-width:1480px;padding-top:1.8rem;padding-bottom:4rem}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif;letter-spacing:-.035em}
h1 {font-size:clamp(2.1rem,4vw,3.5rem)!important;line-height:1.1!important;
background:linear-gradient(105deg,#fff 20%,#9bd7ff 55%,#a5a1ff 95%);
-webkit-background-clip:text;-webkit-text-fill-color:transparent}
h3 {color:#f3f6ff}
p, label, [data-testid="stCaptionContainer"] {color:#a7b5d2}
[data-testid="stSidebar"] {background:#0c1427;border-right:1px solid #253351}
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:12px;border-bottom:1px solid #283450;padding-bottom:8px}
[data-testid="stTabs"] button[role="tab"] {border-radius:12px;padding:10px 22px;color:#9faecc;
transition:background .25s,color .25s,transform .25s}
[data-testid="stTabs"] button[role="tab"]:hover {background:#1c2a47;color:#fff;transform:translateY(-2px)}
[data-testid="stTabs"] button[aria-selected="true"] {color:#8fe6ff!important;background:#172746!important}
[data-testid="stMetric"] {background:linear-gradient(135deg,rgba(26,41,72,.94),rgba(13,23,45,.95));
border:1px solid #304466;border-radius:20px;padding:20px 22px;
box-shadow:0 10px 36px #0003;transition:transform .25s,border-color .25s,box-shadow .25s}
[data-testid="stMetric"]:hover {transform:translateY(-4px);border-color:#47b9e8;box-shadow:0 16px 40px #38bdf822}
[data-testid="stMetricLabel"] {color:#9eb0d0}
[data-testid="stMetricValue"] {color:#f5f8ff;font-family:'Space Grotesk',sans-serif}
[data-testid="stImage"] img {border-radius:22px;border:1px solid #344b70;box-shadow:0 20px 60px #0008}
.stButton>button, [data-testid="stDownloadButton"] button {border-radius:13px!important;
background:linear-gradient(110deg,#26d3ed,#6d7cff)!important;border:1px solid #72dfff55!important;
color:#061329!important;font-weight:800!important;letter-spacing:.015em;
box-shadow:0 7px 22px #3a8efa33;transition:transform .23s,box-shadow .23s,filter .23s!important}
.stButton>button:hover, [data-testid="stDownloadButton"] button:hover {transform:translateY(-3px) scale(1.015);
filter:brightness(1.12);box-shadow:0 12px 30px #38bdf866}
.stButton>button:active {transform:translateY(0) scale(.98)}
[data-testid="stForm"] {background:#101c32;border:1px solid #304466;border-radius:20px;padding:20px}
[data-baseweb="select"]>div, [data-baseweb="input"]>div {background:#101c32!important;
border-color:#354c70!important;border-radius:12px!important}
[data-testid="stAlert"] {border-radius:16px}
[data-testid="stDataFrame"], [data-testid="stVegaLiteChart"] {border:1px solid #304466;border-radius:16px;overflow:hidden}
.hero-card {padding:25px 28px;margin-bottom:24px;border:1px solid #304466;border-radius:24px;
background:linear-gradient(120deg,rgba(31,52,93,.86),rgba(20,28,55,.72));
box-shadow:0 18px 55px #0004;position:relative;overflow:hidden}
.hero-card:after {content:"";position:absolute;width:220px;height:220px;border-radius:50%;
background:#50baff22;filter:blur(55px);right:-50px;top:-80px;pointer-events:none}
.eyebrow {font-size:12px;letter-spacing:.18em;text-transform:uppercase;font-weight:800;color:#74e5f9}
.hero-card p {font-size:15px;margin:8px 0 0;color:#b8c8e4}
.pill {display:inline-block;padding:7px 13px;border-radius:100px;background:#113b42;
border:1px solid #236c70;color:#84f6d8;font-size:12px;font-weight:800;margin-bottom:12px}
.section-hint {color:#a5b5d3;font-size:13px;margin-top:-8px;margin-bottom:18px}
#MainMenu,footer {visibility:hidden}
@media(max-width:720px){.block-container{padding:1rem}.hero-card{padding:18px}}
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
st.markdown("""
<div class="hero-card">
  <div class="pill">● VISION ENGINE · LIVE ANALYTICS</div>
  <div class="eyebrow">CAMPUS INTELLIGENCE PLATFORM / V2.0</div>
  <h1>Spatial Ad Analytics</h1>
  <p>Understand attention in real time. Track multi-person engagement, discover campaign insights, and measure what matters.</p>
</div>
""", unsafe_allow_html=True)

tab_edge, tab_cloud = st.tabs(["🔴 Edge Sensor (Local)", "📈 Cloud Analytics (Global)"])

# =========================================================
# EDGE SENSOR TAB
# =========================================================
with tab_edge:
    left, right = st.columns([3, 1])

    with right:
        st.markdown("### ⚙️ Deployment")
        location = st.selectbox("Campus Zone", [
            "Library Entrance", "Tech Block", "Canteen", "Hostel Gate", "Sports Complex"
        ])
        campaign = st.selectbox("Active Campaign", [
            "Tech Symposium Ad", "Hackathon Poster", "Campus Election", "Club Recruitment"
        ])

        st.markdown("### 📷 Camera controls")
        if not st.session_state.running:
            if st.button("▶ Activate Edge Camera", use_container_width=True):
                st.session_state.running = True
                st.rerun()
        else:
            if st.button("⏹ Stop Sensor", use_container_width=True):
                st.session_state.running = False
                st.rerun()

        st.markdown("### 📊 Session pulse")
        imp_ph = st.empty()
        eng_ph = st.empty()
        rate_ph = st.empty()

    with left:
        st.markdown("### ◉ Live vision feed")
        st.caption("People are tracked independently · Engagement requires 5 seconds of continuous eye detection")
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
    st.markdown("### 📈 Analytics command center")
    st.markdown('<div class="section-hint">Explore campaign performance and manage your locally stored telemetry.</div>', unsafe_allow_html=True)

    # Clear local analytics and current-session counters only after confirmation.
    with st.expander("🗑️ Data management · clear analytics", expanded=False):
        st.warning("This permanently deletes all rows from the local analytics database and resets session counters. This cannot be undone.")
        confirm_clear = st.checkbox("I understand and want to delete all collected analytics", key="confirm_clear")
        if st.button("🗑️ Clear all data", disabled=not confirm_clear, key="clear_analytics"):
            if st.session_state.running:
                st.error("Stop the camera before clearing data.")
            else:
                with sqlite3.connect(logger.db_path) as clear_conn:
                    clear_conn.execute("DELETE FROM ad_analytics")
                st.session_state.session_impressions = 0
                st.session_state.session_engagements = 0
                st.session_state.seen_track_ids = set()
                st.session_state.engaged_track_ids = set()
                st.session_state.last_face_seen = 0.0
                st.success("All local analytics cleared and session counters reset.")
                st.rerun()

    try:
        conn = sqlite3.connect(logger.db_path)
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