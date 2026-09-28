import streamlit as st
import cv2
import pandas as pd
import sys
import sqlite3
from pathlib import Path

# Pathing fix so Streamlit Cloud can find the src modules
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

from src.vision.detector import EyeDetector
from src.data_layer.logger import DetectionLogger

st.set_page_config(page_title="Campus Ad Analytics", page_icon="🎯", layout="wide")

# Apply modern dark theme aesthetics
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
        
        while run_sensor:
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame, faces, engagements = detector.process_frame(frame)
            
            # Log data dynamically if someone enters the frame
            if faces > 0:
                logger.log_interaction(location, campaign, 0.5, engagements > 0)
                
            stframe.image(processed_frame, channels="BGR", use_column_width=True)
            
        cap.release()

with tab2:
    st.markdown("### Location A/B Testing & Engagement Metrics")
    try:
        conn = sqlite3.connect("analytics.db")
        df = pd.read_sql_query("SELECT * FROM ad_analytics", conn)
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Impressions", len(df))
            c2.metric("Total Engagements (Eye Contact)", df['engaged'].sum())
            c3.metric("Avg Engagement Rate", f"{(df['engaged'].mean() * 100):.1f}%")
            
            st.subheader("Performance by Location")
            # Calculate engagement conversion rate per physical zone
            loc_stats = df.groupby('location_tag')['engaged'].mean() * 100
            st.bar_chart(loc_stats)
            
            st.subheader("Raw Telemetry Logs")
            st.dataframe(df.tail(10))
        else:
            st.info("No sensor data logged yet. Run the Edge Sensor locally to collect initial data.")
    except Exception as e:
        st.warning("Database uninitialized. Start the edge sensor first.")