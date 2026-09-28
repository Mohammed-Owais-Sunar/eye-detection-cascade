import streamlit as st
import cv2
import pandas as pd
import sqlite3
import time
import sys
from pathlib import Path
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from src.vision.webcam_processor import AdPulseVideoProcessor

root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

from src.data_layer.logger import DetectionLogger

st.set_page_config(
    page_title="ADPULSE // CONTROL ROOM",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# ADPULSE // CONTROL ROOM UI
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
[data-testid="stExpander"] details > summary,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] button{
  background:#0a0f0c!important;
  background-color:#0a0f0c!important;
  color:#dce8df!important;
  border-color:#27312b!important;
}
[data-testid="stExpander"] details > summary:hover,
[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] button:hover{
  background:#101711!important;
  background-color:#101711!important;
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
  <span><span class="live-dot">●</span> ADPULSE / PRIVACY EDGE NODE</span>
  <span>EDGE AD ANALYTICS // REAL-TIME ATTENTION SYSTEM</span>
  <span>BUILD 2.4.0</span>
</div>
<div class="brand">
  <div class="brand-mark">◉</div>
  <div>
    <h1>ADPULSE</h1>
    <div class="brand-sub">Spatial attention intelligence / control room</div>
  </div>
</div>
""",unsafe_allow_html=True)

tab_edge,tab_cloud,tab_manager=st.tabs(["◉  LIVE CONTROL ROOM","⌁  INTELLIGENCE LOG","⚙  EVENT MANAGER"])

# =========================================================
# LIVE CONTROL ROOM
# =========================================================
with tab_edge:
    left,mid,right=st.columns([6,2.25,2.0],gap="medium")

    with mid:
        st.markdown('<div class="hud"><div class="hud-title">deployment vector</div>',unsafe_allow_html=True)
        active_events=logger.list_events(active_only=True)
        if active_events:
            event_labels={event[0]: f"{event[1]}  //  {event[2]} → {event[3]}" for event in active_events}
            selected_event_id=st.selectbox("EVENT",list(event_labels.keys()),format_func=lambda event_id: event_labels[event_id])
            selected_event=next(event for event in active_events if event[0] == selected_event_id)
            event_id=selected_event[0]
            event_name=selected_event[1]
            campaign=selected_event[4]
            event_locations=logger.get_event_locations(event_id)
            location_labels={loc[0]: f"{loc[1]}  //  {loc[2]}" if loc[2] else loc[1] for loc in event_locations}
            if location_labels:
                selected_location_id=st.selectbox("ZONE",list(location_labels.keys()),format_func=lambda location_id: location_labels[location_id])
                selected_location=next(loc for loc in event_locations if loc[0] == selected_location_id)
                location=selected_location[1]
            else:
                location=None
                st.warning("NO LOCATIONS ASSIGNED TO THIS EVENT. ADD ONE IN EVENT MANAGER.")
        else:
            event_id=None
            event_name=None
            campaign=None
            location=None
            st.warning("NO ACTIVE EVENTS. CREATE AN EVENT IN EVENT MANAGER BEFORE STARTING.")
        if active_events and location:
            st.markdown(f'<div class="notice">EVENT // {event_name}<br>CAMPAIGN // {campaign}<br>ZONE // {location}</div>',unsafe_allow_html=True)
        st.markdown('<div class="sectionline">engine</div>',unsafe_allow_html=True)
        if not st.session_state.running:
            if st.button("▶  INITIALIZE ENGINE",use_container_width=True,disabled=not bool(active_events and location)):
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

        st.markdown('<div class="hud" style="margin-top:12px"><div class="hud-title">detection protocol</div><div class="notice">FACE TRACK → TWO EYES → CONTINUOUS 05.00 SEC → AD ENGAGEMENT<br><br>Each tracked person is counted once. No camera frames or face images are stored. Temporary video processing is used only during the active session.</div></div>',unsafe_allow_html=True)

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
        fps_ph_left=st.empty()

        if st.session_state.running:
            consent = st.checkbox(
                "I CONSENT TO TEMPORARY CAMERA PROCESSING FOR THIS SESSION",
                key="camera_consent",
            )
            if not consent:
                stframe.markdown(
                    '<div class="notice" style="height:430px;display:flex;align-items:center;justify-content:center;text-align:center">'
                    'CAMERA PAUSED<br><br>GRANT SESSION CONSENT TO START BROWSER CAMERA'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                # Render WebRTC INSIDE the left camera panel so the browser
                # video stays next to deployment and telemetry.
                host = str(st.context.headers.get("host", "")).lower()
                is_local = (
                    host.startswith("localhost")
                    or host.startswith("127.0.0.1")
                    or host.startswith("[::1]")
                )

                webrtc_kwargs = {
                    "key": "adpulse-camera",
                    "mode": WebRtcMode.SENDRECV,
                    "video_processor_factory": lambda: AdPulseVideoProcessor(
                        event_id=event_id,
                        event_name=event_name,
                        location=location,
                        campaign=campaign,
                        db_path=logger.db_path,
                    ),
                    "media_stream_constraints": {"video": True, "audio": False},
                    "async_processing": True,
                    "video_html_attrs": {
                        "style": {"width": "100%", "height": "430px", "objectFit": "contain"},
                        "controls": False,
                        "autoPlay": True,
                        "muted": True,
                    },
                }

                if not is_local:
                    webrtc_kwargs["rtc_configuration"] = {
                        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                    }

                ctx = webrtc_streamer(**webrtc_kwargs)

                if ctx.state.playing:
                    st.markdown(
                        '<div class="notice">PRIVACY EDGE ACTIVE // VIDEO FRAMES ARE PROCESSED IN MEMORY '
                        'AND ARE NOT SAVED AS PHOTOS OR VIDEO.</div>',
                        unsafe_allow_html=True,
                    )

                processor = ctx.video_processor
                if processor is not None:
                    snapshot = processor.snapshot()
                    st.session_state.session_impressions = snapshot["unique_people"]
                    st.session_state.session_engagements = snapshot["engagements"]
                    st.session_state.peak_people = max(st.session_state.peak_people, snapshot["peak_people"])
                    st.session_state.peak_looking = max(st.session_state.peak_looking, snapshot["peak_looking"])

                    fps_ph_left.metric("FRAME RATE", f'{snapshot["fps"]:.1f} FPS')

                    latency_ph.metric("LATENCY", f'{snapshot["latency_ms"]:.1f} ms')
                    fps_ph.metric("FRAME RATE", f'{snapshot["fps"]:.1f} FPS')
                    rate = st.session_state.session_engagements / max(st.session_state.session_impressions, 1) * 100
                    imp_ph.markdown(f'<div class="micro">UNIQUE VIEWERS</div><div class="big-number">{st.session_state.session_impressions:02d}</div>', unsafe_allow_html=True)
                    eng_ph.markdown(f'<div class="micro">AD ENGAGEMENTS</div><div class="big-number">{st.session_state.session_engagements:02d}</div>', unsafe_allow_html=True)
                    rate_ph.markdown(f'<div class="micro">ENGAGEMENT RATE</div><div class="big-number">{rate:04.1f}%</div>', unsafe_allow_html=True)
                    live_ph.markdown(
                        f'<div class="signal"><span>looking now</span><b>{snapshot["looking_now"]:02d}</b></div>'
                        f'<div class="signal"><span>engaged now</span><b>{snapshot["engaged_now"]:02d}</b></div>',
                        unsafe_allow_html=True,
                    )
                    peak_ph.markdown(
                        f'<div class="signal"><span>peak audience</span><b>{snapshot["peak_people"]:02d}</b></div>'
                        f'<div class="signal"><span>peak looking</span><b>{snapshot["peak_looking"]:02d}</b></div>',
                        unsafe_allow_html=True,
                    )
                    if snapshot["tracks_html"]:
                        people_ph.markdown(snapshot["tracks_html"], unsafe_allow_html=True)
                    else:
                        people_ph.markdown('<div class="micro">NO ACTIVE TRACKS</div>', unsafe_allow_html=True)
                else:
                    fps_ph_left.metric("FRAME RATE", "0.0 FPS")
        else:
            fps_ph_left.metric("FRAME RATE","—")

        st.markdown('</div>',unsafe_allow_html=True)

    if not st.session_state.running:
        imp_ph.markdown(f'<div class="micro">UNIQUE VIEWERS</div><div class="big-number">{st.session_state.session_impressions:02d}</div>',unsafe_allow_html=True)
        eng_ph.markdown(f'<div class="micro">ENGAGEMENTS</div><div class="big-number">{st.session_state.session_engagements:02d}</div>',unsafe_allow_html=True)
        rate=st.session_state.session_engagements/max(st.session_state.session_impressions,1)*100
        rate_ph.markdown(f'<div class="micro">ENGAGEMENT RATE</div><div class="big-number">{rate:04.1f}%</div>',unsafe_allow_html=True)
        live_ph.markdown('<div class="micro">ENGINE STANDBY</div>',unsafe_allow_html=True)
        peak_ph.markdown(f'<div class="signal"><span>peak audience</span><b>{st.session_state.peak_people:02d}</b></div><div class="signal"><span>peak looking</span><b>{st.session_state.peak_looking:02d}</b></div>',unsafe_allow_html=True)
        latency_ph.metric("LATENCY","—")
        session_ph.metric("SESSION","00:00")
        people_ph.markdown('<div class="micro">NO ACTIVE TRACKS</div>',unsafe_allow_html=True)

# =========================================================
# INTELLIGENCE LOG
# =========================================================
with tab_cloud:
    st.markdown('<div class="sectionline">ad telemetry archive / intelligence log</div>',unsafe_allow_html=True)

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
            <div class="security-kicker">TELEMETRY VAULT</div>
            <div class="security-main">{archive_state}</div>
          </div>
          <div class="security-stat">
            <span>STORED EVENTS</span><b>{archive_count:,}</b>
          </div>
          <div class="security-stat">
            <span>CAMERA FRAMES</span><b>NOT STORED</b>
          </div>
          <div class="security-stat">
            <span>CAMERA STORAGE</span><b>NONE</b>
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
            'Camera frames are processed transiently for detection and are not stored as photos or video.'
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
            x,y=st.columns(2)
            with x:
                st.caption("LOCATION / ENGAGEMENT %")
                st.bar_chart(loc,use_container_width=True)
            with y:
                st.caption("AD CAMPAIGN / ENGAGEMENT %")
                st.bar_chart(camp,use_container_width=True)

            st.markdown('<div class="sectionline">attention timeline</div>',unsafe_allow_html=True)
            timeline=df.set_index("timestamp")["engaged"].resample("1min").agg(["count","sum"])
            timeline.columns=["events","engagements"]
            st.area_chart(timeline,use_container_width=True)

            st.markdown('<div class="sectionline">raw event stream</div>',unsafe_allow_html=True)

            # Keep database IDs private from the presentation layer.
            # Generate stable, readable event IDs for the visible report.
            events=df.tail(100).copy().reset_index(drop=True)
            events.insert(0,"display_event_id",[f"AP-E{i:03d}" for i in range(1,len(events)+1)])

            display_cols=[
                "display_event_id","event_id","event_name","timestamp","location_tag","campaign",
                "dwell_time","engaged","smiled"
            ]
            events=events[display_cols].rename(columns={
                "display_event_id":"EVENT",
                "event_id":"EVENT ID",
                "event_name":"EVENT NAME",
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
            st.download_button("EXPORT TELEMETRY / CSV",df.to_csv(index=False),"adpulse_telemetry.csv","text/csv")
    except Exception as e:
        st.error(f"Telemetry archive error: {e}")


# =========================================================
# EVENT MANAGER
# =========================================================
with tab_manager:
    st.markdown('<div class="sectionline">campaign operations / event registry</div>',unsafe_allow_html=True)
    st.markdown('<div class="notice">CREATE REAL AD CAMPAIGNS, ASSIGN MULTIPLE PHYSICAL ZONES, AND THEN LAUNCH THE LIVE CONTROL ROOM FROM A SELECTED EVENT.</div>',unsafe_allow_html=True)

    create_col, list_col = st.columns([1.05, 1.4], gap="large")

    with create_col:
        st.markdown('<div class="hud"><div class="hud-title">create event</div>',unsafe_allow_html=True)
        event_name_input=st.text_input("EVENT NAME",placeholder="e.g. Tech Fest 2026")
        date_a,date_b=st.columns(2)
        with date_a:
            start_date=st.date_input("START DATE")
        with date_b:
            end_date=st.date_input("END DATE")
        campaign_input=st.selectbox("AD CAMPAIGN",["New Product Launch","Limited-Time Offer","Festival Campaign","Brand Awareness"])
        locations=logger.list_locations(active_only=True)
        location_map={loc[0]: loc[1] for loc in locations}
        selected_location_ids=st.multiselect(
            "ASSIGNED LOCATIONS",
            list(location_map.keys()),
            format_func=lambda location_id: next((f"{loc[1]}  //  {loc[2]}" if loc[2] else loc[1]) for loc in locations if loc[0] == location_id),
        )
        if st.button("＋  CREATE EVENT",use_container_width=True):
            try:
                new_id=logger.create_event(event_name_input,start_date.isoformat(),end_date.isoformat(),campaign_input,selected_location_ids)
                st.success(f"EVENT CREATED // AP-EVENT-{new_id:03d}")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        st.markdown('</div>',unsafe_allow_html=True)

        st.markdown('<div class="hud" style="margin-top:12px"><div class="hud-title">location registry</div>',unsafe_allow_html=True)
        location_name_input=st.text_input("LOCATION NAME",placeholder="e.g. Auditorium Gate")
        building_input=st.text_input("BUILDING / AREA",placeholder="e.g. Main Block")
        if st.button("＋  ADD LOCATION",use_container_width=True):
            try:
                logger.create_location(location_name_input,building_input)
                st.success("LOCATION ADDED")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        st.markdown('</div>',unsafe_allow_html=True)

    with list_col:
        st.markdown('<div class="hud"><div class="hud-title">event registry</div>',unsafe_allow_html=True)
        registry=logger.list_events(active_only=False)
        if registry:
            registry_df=pd.DataFrame(registry,columns=["ID","EVENT","START","END","CAMPAIGN","ACTIVE","LOCATIONS"])
            registry_df["ID"]=registry_df["ID"].map(lambda x:f"EV-{x:03d}")
            registry_df["ACTIVE"]=registry_df["ACTIVE"].map(lambda x:"ONLINE" if x else "OFFLINE")
            st.dataframe(registry_df,use_container_width=True,hide_index=True)
            st.markdown('<div class="sectionline">event controls</div>',unsafe_allow_html=True)
            control_events={row[0]:f"{row[1]}  //  {row[2]} → {row[3]}" for row in registry}
            control_id=st.selectbox("SELECT EVENT",list(control_events.keys()),format_func=lambda x:control_events[x],key="event_control_id")
            control_row=next(row for row in registry if row[0]==control_id)
            current_active=bool(control_row[5])
            if st.button("■  DEACTIVATE EVENT" if current_active else "▶  ACTIVATE EVENT",use_container_width=True):
                logger.toggle_event(control_id,not current_active)
                st.rerun()
        else:
            st.markdown('<div class="hud-empty">EVENT REGISTRY EMPTY</div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

        st.markdown('<div class="hud" style="margin-top:12px"><div class="hud-title">location registry</div>',unsafe_allow_html=True)
        location_registry=logger.list_locations(active_only=False)
        if location_registry:
            loc_df=pd.DataFrame(location_registry,columns=["ID","LOCATION","BUILDING","ACTIVE"])
            loc_df["ID"]=loc_df["ID"].map(lambda x:f"LOC-{x:03d}")
            loc_df["ACTIVE"]=loc_df["ACTIVE"].map(lambda x:"ONLINE" if x else "OFFLINE")
            st.dataframe(loc_df,use_container_width=True,hide_index=True)
        st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="sectionline">how it flows</div>',unsafe_allow_html=True)
    st.markdown('<div class="notice">EVENT MANAGER → CREATE EVENT → ASSIGN LOCATIONS → LIVE CONTROL ROOM → SELECT EVENT + ZONE → INITIALIZE CAMERA → ENGAGEMENTS ARE LOGGED WITH EVENT CONTEXT.</div>',unsafe_allow_html=True)

st.markdown('<div class="footerline"><span>ADPULSE / PRIVACY-FIRST AD ATTENTION</span><span>NO CAMERA FRAMES STORED / TRANSIENT PROCESSING ONLY</span><span>5 SEC ATTENTION PROTOCOL</span></div>',unsafe_allow_html=True)
