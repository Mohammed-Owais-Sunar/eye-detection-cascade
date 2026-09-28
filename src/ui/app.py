import time
import streamlit as st
import cv2
import pandas as pd
import sys
import sqlite3
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

from src.vision.detector import EyeDetector
from src.data_layer.logger import DetectionLogger

st.set_page_config(page_title="Campus Ad Analytics", page_icon="🎯", layout="wide")

st.markdown("""
    
""", unsafe_allow_html=True)

logger = DetectionLogger()

st.title("Spatial Ad Analytics Engine 🎯")
st.write("Edge-to-cloud pipeline for real-time physical advertising conversion metrics.")

tab1, tab2 = st.tabs(["🔴 Edge Sensor (Local)", "📈 Cloud Analytics (Global)"])

with tab1:
    st.markdown("### Live Campus Location Deployment")
    col1, col2 = st.columns(2)
    location = col1.selectbox("Campus Zone", ["Library Entrance", "Tech Block", "Canteen"])
    campaign = col2.selectbox("Active Campaign", ["Tech Symposium Ad", "Hackathon Poster", "Campus Election"])
    
    run_sensor = st.checkbox("Activate Edge Camera Sensor")
    
    if run_sensor:
        stframe = st.empty()
        detector = EyeDetector()
        cap = cv2.VideoCapture(0)
        
        is_person_present = False
        total_eye_contact_time = 0.0
        session_smiled = False
        last_frame_time = time.time()
        
        while run_sensor:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            processed_frame, faces, engagements, smiles = detector.process_frame(frame)
            
            current_time = time.time()
            delta_time = current_time - last_frame_time
            last_frame_time = current_time
            
            if faces > 0:
                if not is_person_present:
                    is_person_present = True
                    total_eye_contact_time = 0.0
                    session_smiled = False
                
                if engagements > 0:
                    total_eye_contact_time += delta_time
                    
                if smiles > 0:
                    session_smiled = True
                    
                if total_eye_contact_time > 0:
                    color = (0, 255, 0) if total_eye_contact_time >= 5.0 else (0, 255, 255)
                    cv2.putText(processed_frame, f"Viewing Time: {round(total_eye_contact_time, 1)}s", 
                                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    if session_smiled:
                        cv2.putText(processed_frame, "Status: DELIGHTED", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                if is_person_present:
                    if total_eye_contact_time >= 5.0:
                        logger.log_interaction(location, campaign, round(total_eye_contact_time, 2), True, session_smiled)
                    is_person_present = False
                    total_eye_contact_time = 0.0
                    session_smiled = False
                
            stframe.image(processed_frame, channels="BGR", use_container_width=True)
            
        cap.release()

with tab2:
    st.markdown("### Location A/B Testing & Engagement Metrics")
    try:
        conn = sqlite3.connect("analytics.db")
        df = pd.read_sql_query("SELECT * FROM ad_analytics", conn)
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Impressions", len(df))
            c2.metric("Total Engagements", df['engaged'].sum())
            c3.metric("Engagement Rate", f"{(df['engaged'].mean() * 100):.1f}%")
            c4.metric("Delight Score (Smiles)", df['smiled'].sum())
            
            st.subheader("Performance by Location")
            loc_stats = df.groupby('location_tag')['engaged'].mean() * 100
            st.bar_chart(loc_stats)
            
            st.subheader("Raw Telemetry Logs")
            st.dataframe(df.tail(10))
            
            st.divider()
            
            # CSV Download Button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Analytics Report (CSV)", data=csv, file_name='ad_metrics.csv', mime='text/csv')
            
            if st.button("🗑️ Clear All Database Logs (Viva Reset)"):
                with sqlite3.connect("analytics.db") as conn:
                    conn.execute("DELETE FROM ad_analytics")
                    conn.commit()
                st.success("Database wiped clean!")
                st.rerun()
                
        else:
            st.info("No sensor data logged yet. Run the Edge Sensor locally to collect initial data.")
    except Exception as e:
        st.warning("Database uninitialized. Start the edge sensor first.")