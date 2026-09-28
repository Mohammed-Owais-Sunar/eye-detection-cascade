import streamlit as st
import cv2
import pandas as pd
import sqlite3
import time
import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

from src.vision.detector import EyeDetector
from src.data_layer.logger import DetectionLogger

st.set_page_config(
    page_title="VISIONGUARD // CONTROL ROOM",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# VISIONGUARD // CONTROL ROOM UI
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root{
  color-scheme:dark;
  --bg:#050607;
  --panel:#090c0d;
  --line:#1c2420;
  --text:#e7eee9;
  --muted:#748078;
  --acid:#b7ff38;
  --cyan:#55e7ff;
  --orange:#ff8b3d;
  --red:#ff4d4d;
}

html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"]{
  background:
    linear-gradient(rgba(183,255,56,.018) 1px,transparent 1px),
    linear-gradient(90deg,rgba(183,255,56,.018) 1px,transparent 1px),
    var(--bg);
  background-size:32px 32px;
  color:var(--text);
  font-family:'Space Grotesk',sans-serif;
}
[data-testid="stHeader"]{background:rgba(5,6,7,.88)}
.block-container{max-width:1540px;padding:18px 30px 70px}
[data-testid="stSidebar"]{
  background:#070909;
  border-right:1px solid var(--line);
}
[data-testid="stSidebar"] *{font-family:'Space Grotesk',sans-serif}
#MainMenu,footer{visibility:hidden}

.commandbar{
  display:flex;justify-content:space-between;align-items:center;
  border-top:1px solid #263029;border-bottom:1px solid #263029;
  padding:9px 0;margin-bottom:22px;
  font-family:'IBM Plex Mono',monospace;font-size:11px;
  letter-spacing:.13em;color:#7d8b82;
}
.live-dot{color:var(--acid)}
.brand{
  display:grid;grid-template-columns:auto 1fr;gap:20px;align-items:end;
  margin:8px 0 26px;
}
.brand-mark{
  width:70px;height:70px;border:1px solid var(--acid);
  display:flex;align-items:center;justify-content:center;
  color:var(--acid);font:600 28px 'IBM Plex Mono';
  box-shadow:inset 0 0 25px rgba(183,255,56,.06),0 0 24px rgba(183,255,56,.05);
}
.brand h1{
  margin:0!important;font-size:clamp(35px,5vw,72px)!important;
  letter-spacing:-.065em!important;line-height:.86!important;
  color:#eef4ef!important;-webkit-text-fill-color:#eef4ef!important;
  background:none!important;
}
.brand-sub{
  margin-top:10px;color:#7d8b82;font:12px 'IBM Plex Mono';
  letter-spacing:.18em;text-transform:uppercase;
}
.hud{
  border:1px solid #27312b;background:rgba(8,11,10,.92);
  position:relative;padding:18px;overflow:hidden;
}
.hud:before,.hud:after{
  content:"";position:absolute;width:18px;height:18px;pointer-events:none;
}
.hud:before{left:-1px;top:-1px;border-left:2px solid var(--acid);border-top:2px solid var(--acid)}
.hud:after{right:-1px;bottom:-1px;border-right:2px solid var(--acid);border-bottom:2px solid var(--acid)}
.hud-title{
  font:11px 'IBM Plex Mono';letter-spacing:.16em;color:#738078;
  text-transform:uppercase;border-bottom:1px solid #202923;padding-bottom:10px;margin-bottom:14px;
}
.feed-wrap{
  position:relative;background:#020303;border:1px solid #27312b;
  padding:5px;box-shadow:0 0 0 1px #090d0b;
}
.feed-wrap:after{
  content:"";position:absolute;left:5px;right:5px;height:1px;
  background:rgba(183,255,56,.28);top:18%;
  box-shadow:0 90px rgba(183,255,56,.08),0 180px rgba(183,255,56,.08),0 270px rgba(183,255,56,.08);
  pointer-events:none;animation:scan 4s linear infinite;
}
@keyframes scan{0%{transform:translateY(0)}100%{transform:translateY(420px)}}
[data-testid="stImage"] img{
  border-radius:0!important;border:0!important;box-shadow:none!important;
  filter:contrast(1.04) saturate(.86);
}
.track{
  border-left:2px solid var(--acid);padding:8px 10px;margin:7px 0;
  background:#0d120e;font:12px 'IBM Plex Mono';color:#d9e6dc;
}
.track.engaged{border-color:var(--orange);color:#fff0df}
.track .time{float:right;color:var(--acid)}
.track.engaged .time{color:var(--orange)}
.signal{
  display:flex;align-items:center;justify-content:space-between;
  padding:12px 0;border-bottom:1px solid #1d251f;
  font-family:'IBM Plex Mono';
}
.signal:last-child{border-bottom:0}
.signal b{font-size:22px;color:#edf5ef}
.signal span{font-size:10px;letter-spacing:.1em;color:#69766e;text-transform:uppercase}
.big-number{
  font:600 clamp(38px,5vw,68px) 'IBM Plex Mono';
  letter-spacing:-.07em;color:var(--acid);line-height:1;
}
.micro{font:10px 'IBM Plex Mono';color:#69766e;letter-spacing:.1em;text-transform:uppercase}
.status-online{color:var(--acid)}
.status-idle{color:#69766e}
.stButton>button,[data-testid="stDownloadButton"] button{
  border-radius:0!important;border:1px solid #344139!important;
  background:#0b0f0c!important;color:#dce8df!important;
  font-family:'IBM Plex Mono'!important;font-size:11px!important;
  letter-spacing:.08em!important;text-transform:uppercase;
  box-shadow:none!important;transition:all .18s ease!important;
}
.stButton>button:hover,[data-testid="stDownloadButton"] button:hover{
  border-color:var(--acid)!important;color:var(--acid)!important;
  background:#111811!important;transform:none!important;
  box-shadow:inset 0 0 18px rgba(183,255,56,.06)!important;
}
[data-testid="stMetric"]{
  background:#080b09;border:0;border-top:1px solid #2a352e;border-radius:0;
  padding:12px 4px;box-shadow:none;transition:border-color .2s;
}
[data-testid="stMetric"]:hover{border-top-color:var(--acid);transform:none;box-shadow:none}
[data-testid="stMetricLabel"]{color:#6f7d74;font:10px 'IBM Plex Mono';text-transform:uppercase}
[data-testid="stMetricValue"]{color:#edf5ef;font:600 25px 'IBM Plex Mono'}
[data-testid="stTabs"] [data-baseweb="tab-list"]{
  gap:0;border-bottom:1px solid #27312b;margin-bottom:22px;
}
[data-testid="stTabs"] button[role="tab"]{
  border-radius:0!important;padding:11px 20px;color:#657168;
  font:11px 'IBM Plex Mono';letter-spacing:.12em;text-transform:uppercase;
}
[data-testid="stTabs"] button[role="tab"]:hover{color:#fff;background:#0c100d}
[data-testid="stTabs"] button[aria-selected="true"]{
  color:var(--acid)!important;background:#0b100b!important;
  box-shadow:inset 0 -2px var(--acid);
}
[data-baseweb="select"]>div{
  background:#090d0b!important;border:1px solid #27312b!important;border-radius:0!important;
}
[data-testid="stExpander"]{border:1px solid #27312b!important;border-radius:0!important;background:#080b09}
[data-testid="stAlert"]{border-radius:0!important}
[data-testid="stDataFrame"],[data-testid="stVegaLiteChart"]{border:1px solid #27312b;border-radius:0;overflow:hidden}
input{font-family:'IBM Plex Mono'!important}
.sectionline{
  display:flex;align-items:center;gap:12px;margin:18px 0 10px;
  color:#7b887f;font:10px 'IBM Plex Mono';letter-spacing:.15em;text-transform:uppercase;
}
.sectionline:after{content:"";height:1px;background:#202922;flex:1}
.notice{
  border:1px dashed #39453d;padding:13px 15px;color:#8d9b92;
  font:11px 'IBM Plex Mono';line-height:1.7;background:#080b09;
}
.footerline{
  margin-top:35px;padding-top:12px;border-top:1px solid #202922;
  display:flex;justify-content:space-between;color:#4e5a53;font:9px 'IBM Plex Mono';
  letter-spacing:.12em;text-transform:uppercase;
}
@media(max-width:900px){.block-container{padding:12px}.brand{grid-template-columns:1fr}.brand-mark{display:none}}
</style>
""", unsafe_allow_html=True)

# =========================================================
# STATE
# =========================================================
defaults={
    "running":False,"session_impressions":0,"session_engagements":0,
    "last_face_seen":0.0,"seen_track_ids":set(),"engaged_track_ids":set(),
    "session_started_at":None,"total_frames":0,"last_latency":0.0
}
for key,value in defaults.items():
    if key not in st.session_state:
        st.session_state[key]=value

logger=DetectionLogger()

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="commandbar">
  <span><span class="live-dot">●</span> VISIONGUARD / LOCAL NODE 01</span>
  <span>EDGE COMPUTER VISION // REAL-TIME ATTENTION SYSTEM</span>
  <span>BUILD 2.4.0</span>
</div>
<div class="brand">
  <div class="brand-mark">◉</div>
  <div>
    <h1>VISIONGUARD</h1>
    <div class="brand-sub">Spatial attention intelligence / control room</div>
  </div>
</div>
""",unsafe_allow_html=True)

tab_edge,tab_cloud=st.tabs(["◉  LIVE CONTROL ROOM","⌁  INTELLIGENCE LOG"])

# =========================================================
# LIVE CONTROL ROOM
# =========================================================
with tab_edge:
    left,mid,right=st.columns([6,2.25,2.0],gap="medium")

    with mid:
        st.markdown('<div class="hud"><div class="hud-title">deployment vector</div>',unsafe_allow_html=True)
        location=st.selectbox("ZONE",["Library Entrance","Tech Block","Canteen","Hostel Gate","Sports Complex"],label_visibility="collapsed")
        campaign=st.selectbox("CAMPAIGN",["Tech Symposium Ad","Hackathon Poster","Campus Election","Club Recruitment"],label_visibility="collapsed")
        st.markdown('<div class="sectionline">engine</div>',unsafe_allow_html=True)
        if not st.session_state.running:
            if st.button("▶  INITIALIZE ENGINE",use_container_width=True):
                st.session_state.running=True
                st.session_state.session_started_at=time.time()
                st.session_state.seen_track_ids=set()
                st.session_state.engaged_track_ids=set()
                st.rerun()
        else:
            if st.button("■  TERMINATE FEED",use_container_width=True):
                st.session_state.running=False
                st.rerun()
        if st.button("↻  RESET SESSION",use_container_width=True):
            if not st.session_state.running:
                st.session_state.session_impressions=0
                st.session_state.session_engagements=0
                st.session_state.seen_track_ids=set()
                st.session_state.engaged_track_ids=set()
                st.session_state.session_started_at=None
                st.rerun()
            else:
                st.warning("Terminate the feed before resetting.")
        st.markdown('</div>',unsafe_allow_html=True)

        st.markdown('<div class="hud" style="margin-top:12px"><div class="hud-title">detection protocol</div><div class="notice">FACE TRACK → TWO EYES → CONTINUOUS 05.00 SEC → ENGAGEMENT<br><br>Each tracked person is counted once. Looking timer resets when eye detection is lost.</div></div>',unsafe_allow_html=True)

    with right:
        st.markdown('<div class="hud"><div class="hud-title">system telemetry</div>',unsafe_allow_html=True)
        status="ONLINE" if st.session_state.running else "STANDBY"
        status_cls="status-online" if st.session_state.running else "status-idle"
        st.markdown(f'<div class="signal"><span>engine</span><b class="{status_cls}">{status}</b></div>',unsafe_allow_html=True)
        imp_ph=st.empty(); eng_ph=st.empty(); rate_ph=st.empty(); live_ph=st.empty(); latency_ph=st.empty(); session_ph=st.empty()
        st.markdown('</div>',unsafe_allow_html=True)
        st.markdown('<div class="hud" style="margin-top:12px"><div class="hud-title">active tracks</div>',unsafe_allow_html=True)
        people_ph=st.empty()
        st.markdown('</div>',unsafe_allow_html=True)

    with left:
        st.markdown('<div class="hud"><div class="hud-title">camera / attention field</div>',unsafe_allow_html=True)
        stframe=st.empty()
        fps_ph=st.empty()
        st.markdown('</div>',unsafe_allow_html=True)

    if st.session_state.running:
        detector=EyeDetector()
        cap=cv2.VideoCapture(0)
        prev_time=time.time()
        try:
            while st.session_state.running:
                ret,frame=cap.read()
                if not ret:
                    st.error("Camera unavailable.")
                    st.session_state.running=False
                    break
                frame=cv2.flip(frame,1)
                result=detector.process_frame(frame)
                st.session_state.total_frames+=1
                st.session_state.last_latency=result.latency_ms
                now=time.time()
                fps=1.0/max(now-prev_time,1e-6)
                prev_time=now

                for person in result.persons:
                    if person.track_id not in st.session_state.seen_track_ids:
                        st.session_state.seen_track_ids.add(person.track_id)
                        st.session_state.session_impressions+=1
                    if person.engaged and person.track_id not in st.session_state.engaged_track_ids:
                        st.session_state.engaged_track_ids.add(person.track_id)
                        st.session_state.session_engagements+=1
                        logger.log_interaction(location,campaign,person.dwell_time,True,False)

                if result.face_count>0:
                    st.session_state.last_face_seen=now

                stframe.image(cv2.cvtColor(result.annotated_frame,cv2.COLOR_BGR2RGB),use_container_width=True)

                looking=sum(1 for p in result.persons if p.looking)
                engaged=sum(1 for p in result.persons if p.engaged)
                rate=st.session_state.session_engagements/max(st.session_state.session_impressions,1)*100
                imp_ph.markdown(f'<div class="micro">UNIQUE PEOPLE</div><div class="big-number">{st.session_state.session_impressions:02d}</div>',unsafe_allow_html=True)
                eng_ph.markdown(f'<div class="micro">ENGAGEMENTS</div><div class="big-number">{st.session_state.session_engagements:02d}</div>',unsafe_allow_html=True)
                rate_ph.markdown(f'<div class="micro">CONVERSION</div><div class="big-number">{rate:04.1f}%</div>',unsafe_allow_html=True)
                live_ph.markdown(f'<div class="signal"><span>looking now</span><b>{looking:02d}</b></div><div class="signal"><span>engaged now</span><b>{engaged:02d}</b></div>',unsafe_allow_html=True)
                latency_ph.metric("LATENCY",f"{result.latency_ms:.1f} ms")
                fps_ph.metric("FRAME RATE",f"{fps:.1f} FPS")
                if st.session_state.session_started_at:
                    elapsed=int(time.time()-st.session_state.session_started_at)
                    session_ph.metric("SESSION",f"{elapsed//60:02d}:{elapsed%60:02d}")

                if result.persons:
                    rows=[]
                    for p in result.persons:
                        state="ENGAGED" if p.engaged else ("LOOKING" if p.looking else "SEEN")
                        cls="track engaged" if p.engaged else "track"
                        rows.append(f'<div class="{cls}">P{p.track_id} / {state}<span class="time">{p.dwell_time:.1f}s</span></div>')
                    people_ph.markdown("".join(rows),unsafe_allow_html=True)
                else:
                    people_ph.markdown('<div class="micro">NO ACTIVE TRACKS</div>',unsafe_allow_html=True)
        finally:
            cap.release()
    else:
        stframe.markdown('<div class="notice" style="height:430px;display:flex;align-items:center;justify-content:center;text-align:center">SYSTEM STANDBY<br><br>INITIALIZE ENGINE TO OPEN THE ATTENTION FIELD</div>',unsafe_allow_html=True)
        fps_ph.metric("FRAME RATE","—")
        imp_ph.markdown(f'<div class="micro">UNIQUE PEOPLE</div><div class="big-number">{st.session_state.session_impressions:02d}</div>',unsafe_allow_html=True)
        eng_ph.markdown(f'<div class="micro">ENGAGEMENTS</div><div class="big-number">{st.session_state.session_engagements:02d}</div>',unsafe_allow_html=True)
        rate=st.session_state.session_engagements/max(st.session_state.session_impressions,1)*100
        rate_ph.markdown(f'<div class="micro">CONVERSION</div><div class="big-number">{rate:04.1f}%</div>',unsafe_allow_html=True)
        live_ph.markdown('<div class="micro">ENGINE STANDBY</div>',unsafe_allow_html=True)
        latency_ph.metric("LATENCY","—")
        session_ph.metric("SESSION","00:00")
        people_ph.markdown('<div class="micro">NO ACTIVE TRACKS</div>',unsafe_allow_html=True)

# =========================================================
# INTELLIGENCE LOG
# =========================================================
with tab_cloud:
    st.markdown('<div class="sectionline">telemetry archive / intelligence log</div>',unsafe_allow_html=True)

    # ---------------------------------------------------------
    # LOCAL DATA WIPE
    # ---------------------------------------------------------
    with sqlite3.connect(logger.db_path) as conn:
        archive_count = conn.execute("SELECT COUNT(*) FROM ad_analytics").fetchone()[0]

    st.markdown(
        f'<div class="hud"><div class="hud-title">local archive / data control</div>'
        f'<div class="notice">ARCHIVE STATUS: <b>{archive_count:,} EVENTS</b><br>'
        'CAMERA FRAMES ARE NEVER STORED. ONLY LOCAL TELEMETRY EVENTS ARE RETAINED.</div></div>',
        unsafe_allow_html=True,
    )

    with st.expander("⌫  OPEN DATA DESTRUCTION CONSOLE"):
        st.warning("PERMANENT ACTION — this deletes every stored engagement event from analytics.db and clears the current session counters.")
        confirm=st.checkbox("I UNDERSTAND — CONFIRM PERMANENT DELETION",key="confirm_clear")
        if st.button("☒  DELETE ALL TELEMETRY",disabled=not confirm,use_container_width=True):
            if st.session_state.running:
                st.error("TERMINATE THE LIVE ENGINE BEFORE DELETING TELEMETRY.")
            else:
                with sqlite3.connect(logger.db_path) as conn:
                    conn.execute("DELETE FROM ad_analytics")
                    conn.commit()
                st.session_state.session_impressions=0
                st.session_state.session_engagements=0
                st.session_state.seen_track_ids=set()
                st.session_state.engaged_track_ids=set()
                st.session_state.session_started_at=None
                st.session_state.last_face_seen=0.0
                st.session_state.total_frames=0
                st.session_state.last_latency=0.0
                st.success("TELEMETRY PURGED — LOCAL ARCHIVE IS NOW EMPTY.")
                time.sleep(0.6)
                st.rerun()

    try:
        with sqlite3.connect(logger.db_path) as conn:
            df=pd.read_sql_query("SELECT * FROM ad_analytics",conn)
        if df.empty:
            st.markdown('<div class="notice">ARCHIVE EMPTY // INITIALIZE THE LIVE CONTROL ROOM TO GENERATE TELEMETRY</div>',unsafe_allow_html=True)
        else:
            df["timestamp"]=pd.to_datetime(df["timestamp"],unit="s")
            total=len(df); engagements=int(df["engaged"].sum()); rate=df["engaged"].mean()*100
            a,b,c,d=st.columns(4)
            a.metric("EVENTS",total); b.metric("ENGAGEMENTS",engagements); c.metric("RATE",f"{rate:.1f}%"); d.metric("AVG DWELL",f"{df['dwell_time'].mean():.1f}s")

            st.markdown('<div class="sectionline">performance vectors</div>',unsafe_allow_html=True)
            loc=df.groupby("location_tag")["engaged"].mean().mul(100).sort_values(ascending=False)
            camp=df.groupby("campaign")["engaged"].mean().mul(100).sort_values(ascending=False)
            x,y=st.columns(2)
            with x:
                st.caption("LOCATION / ENGAGEMENT %")
                st.bar_chart(loc)
            with y:
                st.caption("CAMPAIGN / ENGAGEMENT %")
                st.bar_chart(camp)

            st.markdown('<div class="sectionline">attention timeline</div>',unsafe_allow_html=True)
            timeline=df.set_index("timestamp")["engaged"].resample("1min").agg(["count","sum"])
            timeline.columns=["events","engagements"]
            st.area_chart(timeline)

            st.markdown('<div class="sectionline">raw event stream</div>',unsafe_allow_html=True)
            st.dataframe(df.tail(100),use_container_width=True)
            st.download_button("EXPORT TELEMETRY / CSV",df.to_csv(index=False),"visionguard_telemetry.csv","text/csv")
    except Exception as e:
        st.error(f"Telemetry archive error: {e}")

st.markdown('<div class="footerline"><span>VISIONGUARD / LOCAL-FIRST COMPUTER VISION</span><span>NO CAMERA FRAMES STORED</span><span>5 SEC ATTENTION PROTOCOL</span></div>',unsafe_allow_html=True)
