# 🦅 HawkEye — Explainable AI Surveillance & Biometric Attendance

> **Final Year Engineering Capstone Project**  
> *"HawkEye doesn't just watch for danger — it understands the scene, explains incidents in natural language, and knows who is in the building."*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React Native](https://img.shields.io/badge/React%20Native%20%2F%20Expo-SDK%2052-61DAFB?logo=react&logoColor=black)](https://expo.dev/)
[![YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-00FFFF?logo=ultralytics)](https://github.com/ultralytics/ultralytics)
[![InsightFace](https://img.shields.io/badge/InsightFace-ArcFace%20%2B%20RetinaFace-orange)](https://github.com/deepinsight/insightface)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Executive Summary

Conventional CCTV surveillance systems suffer from three fundamental limitations:
1. **Dumb Motion Alerts**: They flag arbitrary motion (shadows, pets, ceiling fans) rather than understanding semantic context.
2. **Binary Classifications without Explanations**: A classical model outputs `anomaly: 0.88`, leaving human operators guessing what actually occurred.
3. **Disjoint Systems**: Threat monitoring and daily attendance tracking run as separate, siloed hardware setups.

**HawkEye solves this with a unified, edge-deployed AI architecture**:
- **Dual Mission from One Face Database**: Uses a single enrolled local biometric database to simultaneously perform **real-time security incident response** (identifying perpetrators/victims) and **passive, zero-touch daily attendance logging**.
- **Fast/Slow Cascade Architecture** *(inspired by LAVAD, CVPR 2024 & Holmes-VAD)*: Lightweight, real-time edge models (Stage A) run continuously at 20–30 FPS. Only when a potential anomaly is detected does an asynchronous Vision-Language Model (Stage B VLM) wake up to verify the scene, eliminate false alarms, and generate a natural-language incident report.
- **One Process, One Address**: A unified FastAPI application orchestrates the vision pipeline, runs WebSocket streaming, and serves the pre-built React Native / Expo operator dashboard—no separate frontend servers required.

---

## 🏗️ System Architecture

```
                                  ┌────────────────────────────────────────────────────────┐
                                  │            REAL-TIME PIPELINE (Stage A, ~25 FPS)       │
 ┌──────────────────────┐         │                                                        │
 │     Video Ingest     │         │  1. YOLO11n-Pose  ──▶ Skeleton Keypoints (17 joints)   │
 │  • Local Webcam (0)  │───────▶ │  2. ActionNet     ──▶ Bi-GRU (Violence / Fall / Normal)│
 │  • Wireless Phone Cam│         │  3. Weapons YOLO  ──▶ Handheld & Spatial Grasp Gating  │
 │  • Desktop Screen    │         │  4. Fire & Smoke  ──▶ Hybrid YOLO + Chromatic Flicker  │
 │  • RTSP / IP Stream  │         │                                                        │
 └──────────────────────┘         │  Annotated Frame  ──▶ StreamHub (MJPEG / WebSocket)   │
                                  └───────────────────────────┬────────────────────────────┘
                                                              │ (Trigger: Anomaly Proposal)
                                                              ▼
                                  ┌────────────────────────────────────────────────────────┐
                                  │           ASYNC WORKER POOL (Stage B & Decisions)      │
                                  │                                                        │
                                  │  1. Ring-Buffer Snapshot (3-second temporal window)    │
                                  │  2. VLM Incident Reasoner:                             │
                                  │     • Claude 3.5 Sonnet (Cloud, high fidelity)         │
                                  │     • Moondream2 (Local edge VLM, fully offline)       │
                                  │     • Mock Reasoner (Zero-dependency test harness)     │
                                  │  3. Semantic Filter: False alarms pruned silently      │
                                  │  4. Local Biometrics: InsightFace (ArcFace 512-d)      │
                                  │  5. Decision Engine: Severity/Confidence Gates + Rules │
                                  │  6. Automated Dispatch: Police / Ambulance / Fire      │
                                  │  7. Evidence Storage: SQLite + Crops + Thumbnails      │
                                  └───────────────────────────┬────────────────────────────┘
                                                              │
                                                              ▼
                                  ┌────────────────────────────────────────────────────────┐
                                  │       OPERATOR DASHBOARD (FastAPI + React Native Web)  │
                                  │                                                        │
                                  │  • Live Feed with Tactical HUD & AI Overlays           │
                                  │  • Incident Feed with Natural-Language Explanations    │
                                  │  • Daily Attendance Ledger (Passive Biometrics)        │
                                  │  • People Roster & Multi-Photo Face Enrollment         │
                                  │  • Camera Hub: Wireless Phone Broadcaster & Switcher   │
                                  └────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Capabilities

### 1. Threat Detection (Stage A)
- **Violence & Scuffles**: Custom trained Bi-GRU temporal classifier operating over normalized skeleton kinematics, coupled with combatant proximity, striking reach, and spectator rejection geometry.
- **Falls & Medical Collapses**: Keypoint aspect ratio analysis, vertical velocity drops, and ground-level post-impact stillness detection.
- **Brandished Weapons**: Fine-tuned weapon YOLO with verified wrist-to-object grasp geometry to prevent false positives from phones or belts.
- **Fire & Smoke**: Hybrid Approach C combining deep YOLO object proposals with chromatic flame flicker analysis and smoke saturation verification.

### 2. Explainable Incident Reports (Stage B VLM)
When an anomaly crosses threshold, HawkEye extracts temporal video slices and queries the Vision-Language reasoner. Instead of an obscure alert code, HawkEye generates human-readable incident logs:
> *"A physical altercation was confirmed between two individuals in the lobby. Subject A delivered repeated strikes while Subject B was on the floor. Severity: High. Recommended dispatch: Police."*

### 3. Continuous Passive Attendance
- Runs on a background cadence (~0.5s intervals) off the primary detection path.
- Automatically recognizes registered faces via InsightFace (RetinaFace + ArcFace cosine similarity $\ge 0.42$).
- Records exactly one attendance event per person per day (`first_seen` timestamp) and syncs live across WebSockets.

### 4. Turn Any Smartphone into an Edge Surveillance Eye
- **In-Browser WebRTC/WebSocket Broadcaster**: Open HawkEye on any mobile browser (Safari/Chrome), navigate to the **Camera** tab, and tap **"Broadcast to HawkEye"**.
- Hardware camera streams over WebSockets at ~18 FPS directly into the backend AI pipeline.
- Hot-swap sources on the fly between **Phone Camera**, **Webcam**, **Screen Capture**, or **RTSP IP Streams** without restarting the server.

---

## 💻 Tech Stack

| Component | Technology | Description |
|---|---|---|
| **Pose & Keypoints** | Ultralytics YOLO11n-Pose | Real-time 17-point human anatomical keypoint estimation |
| **Action Recognition** | PyTorch Bi-GRU (`action_gru_custom.pt`) | Sequence classification trained on fall, violence, and normal actions |
| **Weapon Detection** | Fine-tuned YOLO (`weapons.pt`) | Handheld knife and firearm detection with grasp validation |
| **Fire & Smoke** | Hybrid YOLO (`fire.pt`) + Chromatic Filter | Dual-spectrum fire and smoke detection |
| **Vision-Language Model** | Anthropic Claude 3.5 Sonnet / Moondream2 | Multimodal incident reasoning & natural-language reports |
| **Facial Biometrics** | InsightFace (buffalo_l: RetinaFace + ArcFace) | 512-dimensional cosine similarity face recognition |
| **Backend Framework** | FastAPI + Uvicorn + SQLite | Async REST API, WebSocket streams, and static asset serving |
| **Operator UI** | React Native Web + Expo (SDK 52) + Reanimated 3 | Responsive dark-mode dashboard with custom HUD and animations |

---

## 🛠️ Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/AdityaV33/HawkEye-V2.git
cd HawkEye-V2
```

### 2. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

**For Linux (CUDA GPU) / Windows:**
```bash
# Optional: Install PyTorch with your matching CUDA version first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-cuda.txt
```

**For macOS (Apple Silicon MPS / CPU) or standard CPU:**
```bash
pip install -r requirements.txt
```

### 4. Warm Pretrained Models (Optional)
```bash
python scripts/download_models.py
```
*(Pre-downloads YOLO11n-pose and InsightFace weights so the first run starts instantly).*

---

## 🎬 Running HawkEye

The camera pipeline, REST API, WebSocket server, and web dashboard run as **one single process on one port**:

```bash
# Default run (uses local webcam 0, mock reasoner, binds to 0.0.0.0:8000)
python -m hawkeye.run

# Use screen capture as video source:
python -m hawkeye.run --source screen

# Loop a demo video clip:
python -m hawkeye.run --source demo_incident.mp4

# Cloud AI Reasoning (Claude 3.5 Sonnet):
export ANTHROPIC_API_KEY="sk-ant-..."
python -m hawkeye.run --backend claude

# Enable HTTPS (recommended when using mobile phone camera via browser):
python -m hawkeye.run --ssl
```

Open **`http://localhost:8000`** (or **`https://localhost:8000`** with `--ssl`) in your browser.

> [!NOTE]
> **No Node.js or npm needed!** The operator dashboard is pre-built in `app/dist/` and served automatically by FastAPI. If you wish to modify the frontend UI, see [`app/README.md`](app/README.md).

---

## 📱 Using Your Phone as the Camera Feed

HawkEye allows you to prop your phone up anywhere in a room and use it as a wireless smart camera:

1. **Connect**: Ensure your phone and laptop are on the same Wi-Fi (or connect your laptop to your phone's Personal Hotspot).
2. **Start HawkEye**:
   ```bash
   python -m hawkeye.run --ssl
   ```
3. **Open on Phone**: Find your laptop's IP (e.g. `192.168.1.15`) and navigate to `https://192.168.1.15:8000` on your phone browser.
4. **Stream**: Tap the **Camera** tab at the bottom and press **"Broadcast to HawkEye"**.
5. Your laptop will immediately run real-time anomaly detection, threat explanation, and facial attendance on your phone's live camera feed!

---

## 🧪 Running Self-Checks & Tests

All detection pipelines, decision engines, and graceful shutdown handlers include unit and integration tests:

```bash
# Run all test suites
python tests/test_action_pipeline.py    # 22 tests: kinematics, models, violence/fall gates
python tests/test_decision_engine.py    # 8 tests: gating, cooldowns, emergency dispatch
python tests/test_fire_detector.py      # 9 tests: chromatic verification, YOLO proposals
python tests/test_pipeline_shutdown.py  # 4 tests: thread safety, queue drainage, stores
```

---

## 📁 Repository Structure

```text
├── app/                  # Operator Dashboard (React Native / Expo Web source code)
│   ├── app/(tabs)/       # Screens: Live, Incidents, Attendance, People, Camera
│   ├── dist/             # Production pre-built web bundle (served by FastAPI)
│   └── src/              # Design system, API hooks, WebSockets, state stores
├── configs/              # System configuration (default.yaml)
├── docs/                 # Documentation (architecture.md, setup.md, training.md, ethics.md)
├── eval/                 # Evaluation and benchmarking scripts
├── hawkeye/              # Core Python AI Engine
│   ├── attendance/       # Biometric attendance ledger & SQLite store
│   ├── decision/         # Severity gating, rule engine, emergency dispatchers
│   ├── detect/           # Stage A detectors: pose, action Bi-GRU, weapons, fire/smoke
│   ├── events/           # Schema definitions, evidence manager, SQLite stores
│   ├── identity/         # InsightFace integration, RetinaFace + ArcFace matcher
│   ├── ingest/           # Video sources: webcam, screen capture, phone push stream
│   ├── pipeline.py       # Multi-threaded orchestration pipeline
│   ├── reason/           # Stage B Vision-Language reasoners (Claude, Moondream, Mock)
│   └── run.py            # CLI entry point
├── models/               # Trained neural network weights (.pt files)
├── scripts/              # Training, model downloading, and enrollment utilities
├── server/               # FastAPI application, WebSocket hub, and REST routes
├── tests/                # Automated verification and pipeline test suites
├── requirements.txt      # Python dependencies (CPU & macOS)
└── requirements-cuda.txt # Python dependencies (NVIDIA CUDA)
```

---

## ⚖️ Ethics, Privacy & Consent

HawkEye was designed from the ground up under a strict **privacy-first biometric architecture**:
- **Consented Local DB Only**: Facial identity matching operates strictly against a locally consented enrollment roster (`data/hawkeye.db`).
- **No Remote Biometrics**: Face embeddings (512-d ArcFace vectors) remain local to the host machine.
- **Selective VLM Transmission**: The heavy vision-language model only receives a temporary 3-second visual clip when an anomaly is triggered, minimizing unnecessary data processing.
- See [`docs/ethics.md`](docs/ethics.md) for our detailed privacy and ethics framework.

---

## 👥 Authors & Acknowledgements

* **HawkEye Engineering Team** — *Final Year Project*
* Developed with guidance on modern computer vision architectures (*LAVAD, CVPR 2024*; *Holmes-VAD*).
* Built using open-source tools: Ultralytics YOLO11, InsightFace, FastAPI, and Expo.

---

<div align="center">
  <b>HawkEye — Autonomous Indoor Surveillance & Attendance System</b>
</div>
