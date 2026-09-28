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
.metric-chart{
  border:1px solid #27312b;background:#060908;padding:14px 16px;
  margin-top:4px;overflow:hidden;
}
.metric-chart-title{
  color:#738078;font:10px 'IBM Plex Mono';letter-spacing:.12em;
  text-transform:uppercase;margin-bottom:14px;
}
.hud-bars{display:flex;flex-direction:column;gap:9px}
.hud-bar-row{
  display:grid;grid-template-columns:145px 1fr 58px;gap:10px;
  align-items:center;font:10px 'IBM Plex Mono';color:#b8c5bc;
}
.hud-bar-label{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hud-bar-track{height:8px;background:#101611;border:1px solid #202a23;position:relative}
.hud-bar-fill{height:100%;background:linear-gradient(90deg,#5d8f24,#b7ff38)}
.hud-bar-value{text-align:right;color:#b7ff38}
.hud-timeline{
  display:flex;align-items:flex-end;gap:3px;height:145px;
  padding:12px 6px 4px;border-top:1px solid #202922;
}
.hud-time-bar{
  flex:1;min-width:3px;background:#55e7ff;opacity:.72;
  box-shadow:0 0 8px rgba(85,231,255,.10);
}
.hud-time-bar:hover{opacity:1;background:#b7ff38}
.hud-empty{color:#536058;font:10px 'IBM Plex Mono';padding:18px 0}
.event-grid{
  border:1px solid #27312b;
  background:#060908;
  overflow:hidden;
}
.event-grid-scroll{
  max-height:390px;
  overflow:auto;
  scrollbar-width:thin;
  scrollbar-color:#344239 #080b09;
}
.event-grid table{
  width:100%;
  min-width:900px;
  border-collapse:collapse;
  table-layout:fixed;
  font:10px 'IBM Plex Mono';
  color:#cbd6ce;
}
.event-grid th{
  position:sticky;
  top:0;
  z-index:2;
  background:#0c120e;
  color:#8fa097;
  border-bottom:1px solid #344139;
  padding:10px 12px;
  text-align:left;
  letter-spacing:.10em;
  font-weight:500;
  white-space:nowrap;
}
.event-grid td{
  padding:9px 12px;
  border-bottom:1px solid #18211b;
  text-align:left;
  vertical-align:middle;
  white-space:nowrap;
  overflow:hidden;
  text-overflow:ellipsis;
}
.event-grid tr:hover td{
  background:#0d140f;
  color:#eff7f1;
}
.event-grid th:nth-child(1),.event-grid td:nth-child(1){width:105px;color:var(--acid)}
.event-grid th:nth-child(2),.event-grid td:nth-child(2){width:185px}
.event-grid th:nth-child(3),.event-grid td:nth-child(3){width:145px}
.event-grid th:nth-child(4),.event-grid td:nth-child(4){width:205px}
.event-grid th:nth-child(5),.event-grid td:nth-child(5){width:90px;text-align:right}
.event-grid th:nth-child(6),.event-grid td:nth-child(6),
.event-grid th:nth-child(7),.event-grid td:nth-child(7){width:90px;text-align:center}
.footerline{
  margin-top:35px;padding-top:12px;border-top:1px solid #202922;
  display:flex;justify-content:space-between;color:#4e5a53;font:9px 'IBM Plex Mono';
  letter-spacing:.12em;text-transform:uppercase;
}
.security-strip{
  display:grid;grid-template-columns:2fr 1fr 1.25fr 1.1fr;
  gap:0;border:1px solid #27312b;background:#070a08;
  margin:4px 0 12px;min-height:78px;
}
.security-strip>div{padding:14px 16px;border-right:1px solid #202922}
.security-strip>div:last-child{border-right:0}
.security-kicker,.security-stat span{
  display:block;color:#657269;font:9px 'IBM Plex Mono';
  letter-spacing:.14em;text-transform:uppercase;
}
.security-main{margin-top:7px;color:var(--acid);font:600 20px 'IBM Plex Mono';letter-spacing:.02em}
.security-stat{display:flex;flex-direction:column;justify-content:center;gap:7px}
.security-stat b{color:#dfe9e2;font:500 13px 'IBM Plex Mono';letter-spacing:.04em}
.danger-panel{
  border:1px solid #4a2929;background:#120909;padding:15px 17px;
  margin-bottom:14px;
}
.danger-title{color:#ff7777;font:600 11px 'IBM Plex Mono';letter-spacing:.14em}
.danger-copy{color:#9c8b8b;font:11px 'IBM Plex Mono';line-height:1.7;margin-top:9px}
.danger-copy b{color:#e2cccc}
@media(max-width:900px){
  .security-strip{grid-template-columns:1fr 1fr}
  .security-strip>div:nth-child(2){border-right:0}
}
/* FINAL HUD POLISH */
[data-testid="stExpander"] summary{
  background:#0a0f0c!important;
  color:#dce8df!important;
}
[data-testid="stExpander"] summary:hover{
  background:#101711!important;
  color:var(--acid)!important;
}
[data-testid="stCheckbox"] label{
  color:#cfd9d2!important;
}
[data-testid="stCheckbox"] [data-baseweb="checkbox"]{
  border-color:#4c5b50!important;
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
    "session_started_at":None,"total_frames":0,"last_latency":0.0,
    "peak_people":0,"peak_looking":0
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
        location=st.selectbox("ZONE",["Library Entrance","O Building","Canteen","Hostel Gate","Sports Complex"],label_visibility="collapsed")
        campaign=st.selectbox("CAMPAIGN",["Mid-Sem Exam Timetable","Hackathon Poster","Campus Election","Club Recruitment"],label_visibility="collapsed")
        st.markdown('<div class="sectionline">engine</div>',unsafe_allow_html=True)
        if not st.session_state.running:
            if st.button("▶  INITIALIZE ENGINE",use_container_width=True):
                st.session_state.running=True
                st.session_state.session_started_at=time.time()
                st.session_state.seen_track_ids=set()
                st.session_state.engaged_track_ids=set()
                st.session_state.peak_people=0
                st.session_state.peak_looking=0
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
                st.session_state.peak_people=0
                st.session_state.peak_looking=0
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
        imp_ph=st.empty(); eng_ph=st.empty(); rate_ph=st.empty(); live_ph=st.empty(); latency_ph=st.empty(); session_ph=st.empty(); peak_ph=st.empty()
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
                st.session_state.peak_people=max(st.session_state.peak_people,len(result.persons))
                st.session_state.peak_looking=max(st.session_state.peak_looking,looking)
                rate=st.session_state.session_engagements/max(st.session_state.session_impressions,1)*100
                imp_ph.markdown(f'<div class="micro">UNIQUE PEOPLE</div><div class="big-number">{st.session_state.session_impressions:02d}</div>',unsafe_allow_html=True)
                eng_ph.markdown(f'<div class="micro">ENGAGEMENTS</div><div class="big-number">{st.session_state.session_engagements:02d}</div>',unsafe_allow_html=True)
                rate_ph.markdown(f'<div class="micro">CONVERSION</div><div class="big-number">{rate:04.1f}%</div>',unsafe_allow_html=True)
                live_ph.markdown(f'<div class="signal"><span>looking now</span><b>{looking:02d}</b></div><div class="signal"><span>engaged now</span><b>{engaged:02d}</b></div>',unsafe_allow_html=True)
                peak_ph.markdown(f'<div class="signal"><span>peak audience</span><b>{st.session_state.peak_people:02d}</b></div><div class="signal"><span>peak looking</span><b>{st.session_state.peak_looking:02d}</b></div>',unsafe_allow_html=True)
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
        peak_ph.markdown(f'<div class="signal"><span>peak audience</span><b>{st.session_state.peak_people:02d}</b></div><div class="signal"><span>peak looking</span><b>{st.session_state.peak_looking:02d}</b></div>',unsafe_allow_html=True)
        latency_ph.metric("LATENCY","—")
        session_ph.metric("SESSION","00:00")
        people_ph.markdown('<div class="micro">NO ACTIVE TRACKS</div>',unsafe_allow_html=True)

# =========================================================
# INTELLIGENCE LOG
# =========================================================
with tab_cloud:
    st.markdown('<div class="sectionline">telemetry archive / intelligence log</div>',unsafe_allow_html=True)

    # ---------------------------------------------------------
    # PRIVACY / DATA SECURITY CONSOLE
    # ---------------------------------------------------------
    with sqlite3.connect(logger.db_path) as conn:
        archive_count = conn.execute("SELECT COUNT(*) FROM ad_analytics").fetchone()[0]

    archive_label = "ARCHIVE EMPTY" if archive_count == 0 else f"{archive_count:,} TELEMETRY EVENTS"
    archive_state = "● CLEAR" if archive_count == 0 else "● ACTIVE"

    st.markdown(
        f'''
        <div class="security-strip">
          <div>
            <div class="security-kicker">LOCAL DATA VAULT</div>
            <div class="security-main">{archive_state}</div>
          </div>
          <div class="security-stat">
            <span>STORED EVENTS</span><b>{archive_count:,}</b>
          </div>
          <div class="security-stat">
            <span>CAMERA FRAMES</span><b>NOT STORED</b>
          </div>
          <div class="security-stat">
            <span>DATA LOCATION</span><b>LOCAL ONLY</b>
          </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    with st.expander("⌫  DANGER ZONE / PERMANENT DATA WIPE"):
        st.markdown(
            '<div class="danger-panel">'
            '<div class="danger-title">⚠ PERMANENT DELETION PROTOCOL</div>'
            '<div class="danger-copy">'
            'This operation permanently removes every stored engagement event from '
            '<b>analytics.db</b> and resets the current session telemetry. '
            'Camera frames are not stored.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        confirm=st.checkbox(
            "I UNDERSTAND — DELETE THE COMPLETE LOCAL TELEMETRY ARCHIVE",
            key="confirm_clear",
        )

        if st.button(
            "☒  PURGE ALL TELEMETRY",
            disabled=not confirm,
            use_container_width=True,
        ):
            if st.session_state.running:
                st.error("TERMINATE THE LIVE ENGINE BEFORE PURGING TELEMETRY.")
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
                st.session_state.peak_people=0
                st.session_state.peak_looking=0

                st.success("✓ TELEMETRY PURGED — LOCAL DATA VAULT IS CLEAR.")
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

            def hud_bar_chart(series, title):
                if series.empty:
                    return f'<div class="metric-chart"><div class="metric-chart-title">{title}</div><div class="hud-empty">NO TELEMETRY AVAILABLE</div></div>'
                max_value=max(float(series.max()),1.0)
                rows=[]
                for label,value in series.items():
                    width=max(0,min(float(value)/max_value*100,100))
                    rows.append(
                        f'<div class="hud-bar-row">'
                        f'<div class="hud-bar-label">{label}</div>'
                        f'<div class="hud-bar-track"><div class="hud-bar-fill" style="width:{width:.1f}%"></div></div>'
                        f'<div class="hud-bar-value">{float(value):.1f}%</div>'
                        f'</div>'
                    )
                return f'<div class="metric-chart"><div class="metric-chart-title">{title}</div><div class="hud-bars">{"".join(rows)}</div></div>'

            x,y=st.columns(2)
            with x:
                st.markdown(hud_bar_chart(loc,"LOCATION / ENGAGEMENT %"),unsafe_allow_html=True)
            with y:
                st.markdown(hud_bar_chart(camp,"CAMPAIGN / ENGAGEMENT %"),unsafe_allow_html=True)

            st.markdown('<div class="sectionline">attention timeline</div>',unsafe_allow_html=True)
            timeline=df.set_index("timestamp")["engaged"].resample("1min").agg(["count","sum"])
            timeline.columns=["events","engagements"]

            if timeline.empty:
                st.markdown('<div class="metric-chart"><div class="hud-empty">NO TIMELINE DATA</div></div>',unsafe_allow_html=True)
            else:
                peak=max(int(timeline["events"].max()),1)
                bars=[]
                for _,row in timeline.tail(60).iterrows():
                    height=max(4,min(float(row["events"])/peak*100,100))
                    bars.append(f'<div class="hud-time-bar" style="height:{height:.1f}%" title="Events: {int(row["events"])} | Engagements: {int(row["engagements"])}"></div>')
                st.markdown(
                    '<div class="metric-chart">'
                    '<div class="metric-chart-title">EVENT DENSITY / 1 MINUTE</div>'
                    f'<div class="hud-timeline">{"".join(bars)}</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="sectionline">raw event stream</div>',unsafe_allow_html=True)

            # Keep database IDs private from the presentation layer.
            # Generate stable, readable event IDs for the visible report.
            events=df.tail(100).copy().reset_index(drop=True)
            events.insert(0,"event_id",[f"VG-E{i:03d}" for i in range(1,len(events)+1)])

            display_cols=[
                "event_id","timestamp","location_tag","campaign",
                "dwell_time","engaged","smiled"
            ]
            events=events[display_cols].rename(columns={
                "event_id":"EVENT",
                "timestamp":"TIMESTAMP",
                "location_tag":"LOCATION",
                "campaign":"CAMPAIGN",
                "dwell_time":"DWELL",
                "engaged":"ENGAGED",
                "smiled":"SMILED",
            })
            events["TIMESTAMP"]=events["TIMESTAMP"].dt.strftime("%d %b %Y  %H:%M:%S")
            events["DWELL"]=events["DWELL"].map(lambda x:f"{x:.1f}s")
            events["ENGAGED"]=events["ENGAGED"].map(lambda x:"YES" if bool(x) else "NO")
            events["SMILED"]=events["SMILED"].map(lambda x:"YES" if bool(x) else "NO")

            table_html=events.to_html(index=False,escape=True,border=0)
            st.markdown(
                f'<div class="event-grid"><div class="event-grid-scroll">{table_html}</div></div>',
                unsafe_allow_html=True,
            )
            st.download_button("EXPORT TELEMETRY / CSV",df.to_csv(index=False),"visionguard_telemetry.csv","text/csv")
    except Exception as e:
        st.error(f"Telemetry archive error: {e}")

st.markdown('<div class="footerline"><span>VISIONGUARD / LOCAL-FIRST COMPUTER VISION</span><span>NO CAMERA FRAMES STORED</span><span>5 SEC ATTENTION PROTOCOL</span></div>',unsafe_allow_html=True)
