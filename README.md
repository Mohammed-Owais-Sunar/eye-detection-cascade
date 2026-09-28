# ADPULSE — Advertisement Attention Analytics

ADPULSE is a privacy-conscious computer-vision analytics prototype for measuring **advertisement attention** from a browser webcam.

It detects faces and eyes with OpenCV Haar cascades, maintains temporary spatial tracks for multiple people, and records an engagement when a tracked person is estimated to look toward the camera continuously for **5 seconds or more**.

## What ADPULSE does

- 📷 Browser webcam input through Streamlit WebRTC
- 👥 Temporary multi-person spatial tracking
- 👀 Eye-based attention estimation
- ⏱️ 5-second continuous attention threshold
- 1️⃣ Counts each temporary track as an engagement only once
- 📊 Live unique-viewer, engagement, rate and dwell-time analytics
- 📍 Event, campaign and location management
- 🧠 Intelligence Log for recorded engagement events
- 🔒 Privacy-first design: no webcam frames, face photos or biometric embeddings are intentionally persisted

## Important limitation

ADPULSE is an **attention estimation system**, not a biometric identification system and not a precise gaze tracker.

The current detector uses face/eye detection plus temporary spatial tracking. Detecting two eyes is treated as an approximation of looking toward the camera. Lighting, glasses, head angle, occlusion, camera quality and Haar-cascade limitations can affect accuracy.

A future V2 can use head-pose or gaze-estimation models for stronger attention estimation without introducing identity recognition.

## Architecture

```
Browser Webcam
      │
      ▼
Streamlit WebRTC
      │
      ▼
AdPulseVideoProcessor
      │
      ├── OpenCV Face Cascade
      ├── OpenCV Eye Cascade
      ├── Temporary Spatial Tracking
      └── 5-second Engagement Logic
      │
      ▼
Analytics / Event Logger
      │
      ├── Live Control Room
      ├── Intelligence Log
      └── Event Manager
```

## Tech stack

- Python
- Streamlit
- streamlit-webrtc
- OpenCV
- NumPy
- Pandas
- Plotly
- SQLite

## Run locally

### 1. Clone

```bash
git clone https://github.com/Mohammed-Owais-Sunar/eye-detection-cascade.git
cd eye-detection-cascade
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Start ADPULSE

```powershell
python -m streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

## Demo test

1. Create/select an event.
2. Select a location and campaign.
3. Give camera consent.
4. Start the browser camera.
5. Keep one person facing the camera for at least 5 seconds.
6. Verify one engagement is recorded.
7. Continue looking — the same temporary track should not create repeated engagements.
8. Test two people simultaneously.
9. Test the system again with different lighting and head angles.

See [TESTING.md](TESTING.md) for the full validation checklist.

## Privacy

ADPULSE is designed around aggregate analytics rather than identity recognition.

The application does not intentionally save webcam video, face photographs or face embeddings. Webcam frames are processed transiently by the vision pipeline while the session is running. Analytics contain event/location/campaign context and engagement measurements.

For a real-world deployment, camera consent, signage, data-retention rules and applicable privacy requirements should be reviewed before collecting audience analytics.

## Deployment

The project can be deployed on Streamlit Community Cloud from the `main` branch with `app.py` as the entry point.

Browser camera access may depend on browser permissions and WebRTC network conditions. A deployment that works locally can still require WebRTC/ICE configuration adjustments on some networks.

## Roadmap

### Current version
- Multi-person attention estimation
- 5-second engagement threshold
- Event/campaign/location analytics
- Privacy-conscious aggregate logging

### V2 ideas
- Head-pose estimation
- More robust gaze/attention estimation
- Better handling of glasses and difficult lighting
- Accuracy benchmark dataset
- Performance monitoring
- Optional export/reporting improvements

**ADPULSE = Advertisement + Audience Pulse**
