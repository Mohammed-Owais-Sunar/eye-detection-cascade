import sys
from pathlib import Path
import time
import cv2
import streamlit as st
import pandas as pd

# Add the project root to Python's path so Streamlit Cloud can find the 'src' folder
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

from src.data_layer.logger import DetectionLogger
from src.vision.detector import EyeDetector

st.set_page_config(page_title="Campus Ad Analytics", page_icon="📊", layout="wide")

# ... (Keep the rest of your app.py code exactly the same below this line) ...

@st.cache_resource
def get_detector() -> EyeDetector:
    return EyeDetector()

@st.cache_resource
def get_logger() -> DetectionLogger:
    return DetectionLogger()

def main() -> None:
    st.title("📊 Spatial Ad Analytics Engine")
    st.caption("A/B testing physical campus locations by converting foot traffic into visual engagement metrics.")

    detector = get_detector()
    logger = get_logger()

    tab_sensor, tab_analytics = st.tabs(["📸 Live Sensor Deployment", "📈 Location Analytics"])

    with tab_sensor:
        col_controls, col_video = st.columns([1, 3])
        
        with col_controls:
            st.subheader("Sensor Config")
            location_name = st.text_input("Deployment Location", value="Main Library Foyer", help="Where is this screen currently located?")
            run_stream = st.toggle("Activate Sensor", value=False)
            
            st.divider()
            st.metric("Current Target", location_name)
            metric_fps = st.empty()
            metric_traffic = st.empty()
            metric_looks = st.empty()

        with col_video:
            video_placeholder = st.empty()
            if not run_stream:
                video_placeholder.info("Sensor inactive. Set location and toggle 'Activate Sensor'.")

        if run_stream:
            cap = cv2.VideoCapture(0)
            last_log_time = time.time()
            prev_frame_time = time.time()

            try:
                while run_stream:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.flip(frame, 1)
                    result = detector.process_frame(frame)

                    current_time = time.time()
                    fps = 1.0 / max((current_time - prev_frame_time), 1e-5)
                    prev_frame_time = current_time

                    rgb_frame = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                    metric_fps.metric(label="System FPS", value=f"{fps:.1f}")
                    metric_traffic.metric(label="Live Foot Traffic (Faces)", value=str(result.face_count))
                    metric_looks.metric(label="Active Viewers (Eyes)", value=str(result.eye_count))

                    # Log to database every 2 seconds
                    if current_time - last_log_time >= 2.0 and (result.face_count > 0 or result.eye_count > 0):
                        logger.log_event(
                            location=location_name,
                            face_count=result.face_count,
                            eyes_detected=result.eye_count,
                            latency_ms=result.latency_ms,
                        )
                        last_log_time = current_time
            finally:
                cap.release()

    with tab_analytics:
        st.subheader("Advertising Location Performance")
        st.markdown("Compare the engagement rates of different campus deployment zones to determine the highest ROI for digital signage.")
        
        df = logger.get_location_analytics()
        
        if df is None or df.empty:
            st.warning("No data collected yet. Deploy the sensor to gather analytics.")
        else:
            st.dataframe(
                df,
                column_config={
                    "location": "Campus Location",
                    "total_foot_traffic": "Total Foot Traffic (Faces)",
                    "total_engagement": "Total Eye Contact",
                    "engagement_rate_%": st.column_config.ProgressColumn(
                        "Engagement Rate (%)",
                        help="Percentage of maximum possible eye contact.",
                        format="%f%%",
                        min_value=0,
                        max_value=100,
                    ),
                },
                hide_index=True,
                use_container_width=True
            )
            
            st.bar_chart(data=df, x="location", y="engagement_rate_%", color="#38bdf8")

if __name__ == "__main__":
    main()