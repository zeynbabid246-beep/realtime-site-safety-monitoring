# 🏗️ AI-Powered Construction Safety Monitoring & Hazard Detection System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0.0-009688.svg)](https://fastapi.tiangolo.com/)
[![YOLOv8/v11](https://img.shields.io/badge/YOLO-Ultralytics-00FFFF.svg)](https://docs.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent, real-time computer vision and spatial intelligence platform designed for construction site safety monitoring. The system combines multi-model deep learning (YOLO), ByteTrack object tracking, unsupervised geometric clustering (HDBSCAN), anatomical spatial association, and a trained Siamese face-verification model to automatically detect safety violations, heavy machinery proximity hazards, unauthorized danger-zone incursions, early fire/smoke incidents — and to **identify which registered worker** is involved.

> **👷 Worker Face Recognition:** registered workers are verified on camera with the trained Siamese model (`notebooks/siamesemodelv2.h5`) and linked to detected Person boxes; unrecognized faces are shown as `Unknown`. See **[docs/FACE_RECOGNITION.md](docs/FACE_RECOGNITION.md)** for worker registration, data layout, thresholds, and the REST API.

---

## 📑 Table of Contents

- [Quick Start](#-quick-start)
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Technologies & Tools](#-technologies--tools)
- [Project Architecture & Directory Structure](#-project-architecture--directory-structure)
- [Environment Setup](#-environment-setup)
- [Model & Weights Setup](#-model--weights-setup)
- [How to Run the System](#-how-to-run-the-system)
  - [1. Web Application & Live Dashboard (FastAPI)](#1-web-application--live-dashboard-fastapi)
  - [2. Offline Video Safety Analysis CLI](#2-offline-video-safety-analysis-cli)
  - [3. Real-Time Webcam / RTSP Stream CLI](#3-real-time-webcam--rtsp-stream-cli)
  - [4. Worker Face Recognition (Siamese verification)](#4-worker-face-recognition-siamese-verification)
  - [5. REST API Image & Video Endpoints](#5-rest-api-image--video-endpoints)
- [Environment Configuration](#️-environment-configuration)
- [Configuration Reference](#-configuration-reference)
- [Input & Output Specifications](#-input--output-specifications)
- [Detection Classes & Hazard Categories](#-detection-classes--hazard-categories)
- [End-to-End Pipeline Workflow](#-end-to-end-pipeline-workflow)
- [Troubleshooting & Common Issues](#-troubleshooting--common-issues)
- [Performance & Detection Tuning Tips](#-performance--detection-tuning-tips)
- [Quick Reference Paths](#-quick-reference-paths)

---

## 🚀 Quick Start

Run the entire safety platform in five copy-pasteable commands:

```bash
# 1. Clone the repository
git clone https://github.com/your-username/construction_safety_system.git
cd construction_safety_system

# 2. Create and activate a Python 3.11 virtual environment
python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Configure environment (optional - defaults work for basic usage)
cp .env.example .env
# Edit .env to enable Telegram alerts or customize thresholds

# 5. Launch the web platform and API
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and navigate to:
👉 **`http://127.0.0.1:8000`** — Live interactive dashboard with camera, alerts, history, statistics, evidence, and reports.
👉 **`http://127.0.0.1:8000/docs`** — Interactive Swagger API documentation.

---

## 🎯 Project Overview

Construction environments are among the most hazardous industrial workplaces worldwide. Common accidents include struck-by incidents with heavy equipment, falls from restricted zones, lack of personal protective equipment (PPE), and electrical/fire hazards.

### Objective
This system provides automated, continuous optical surveillance for job sites by turning standard CCTV, drone, or webcam feeds into active safety intelligence. Instead of merely drawing raw bounding boxes, the system acts as an **expert safety officer**:

1. **PPE Compliance**: Detects whether workers are properly equipped with hardhats, vests, and protective gear.
2. **Proximity Analysis**: Measures bottom-center ground distance between workers and heavy machinery (excavators, dump trucks, cranes, cement mixers) to alert when workers enter blind spots or crush zones.
3. **Dynamic Danger Zones**: Uses HDBSCAN clustering and convex hull geometry to detect boundaries demarcated by safety cones and flags workers trespassing inside.
4. **Utility Pole Proximity**: Identifies cranes or booms operating hazardously close to overhead power poles and lines.
5. **Fire & Smoke Debouncing**: Detects combustion hazards early, filtering out optical reflections and transient dust plumes.
6. **Worker Identification**: Verifies detected Person faces against registered workers (`data/face_data/input_images/<worker_id>/`) with the trained Siamese verification model; multiple simultaneous workers supported, `Unknown` for unregistered faces. REST API + CLI registration, threshold calibration script included.
7. **Overall Risk Index**: Aggregates all concurrent hazards into a standardized safety risk rating: **`SAFE`**, **`LOW`**, **`MEDIUM`**, **`HIGH`**, or **`CRITICAL`**.

---

## ✨ Key Features

| Capability | Description |
|---|---|
| **Multi-Model Fusion** | Concurrently executes unified hazard detection (`models/hazard/best.pt`) and specialized combustion detection (`models/fire_smoke/best.pt`). |
| **Resolution-Aware Filtering** | Dynamically scales bounding box size floors across 720p, 1080p, and 4K UHD resolutions to prevent small debris and distant poles from triggering false worker detections. |
| **Stationary Fixture Rejection** | Motion-variance filter (`StaticPersonFilter`) distinguishes moving human workers from immobile physical fixtures (poles, survey tripods, rebar stakes) across consecutive frames. |
| **Perspective-Aware Sizing** | Leverages camera elevation geometry: foreground objects near the bottom of the frame are required to meet realistic close-range scale requirements. |
| **Anatomical PPE Association** | Checks PPE violations against precise anatomical body sub-regions (head top 30% for hardhats; torso 25%–75% for safety vests) rather than coarse whole-body overlap. |
| **Persistent ByteTrack Tracking** | Assigns persistent tracking IDs to persons and machinery, enabling movement tracking and zone dwell-time analysis. |
| **Temporal Zone Stabilization** | `DangerZoneTracker` stabilizes cone cluster polygons across video frames with polygon IoU matching, eliminating flickering zone IDs. |
| **Ground-Point Distance** | Calculates Euclidean distance from the bottom-center contact points of bounding boxes to approximate actual ground plane proximity. |
| **Full Web Application** | FastAPI backend with WebSocket streaming (`/safety/ws/camera`) and modern responsive frontend dashboard with live camera controls and real-time alert counters. |
| **Persistent Monitoring** | SQLite storage records all safety events with cooldown-bounded deduplication. Dashboard, alerts, history, statistics, evidence gallery, and CSV reports. |
| **Alert System** | HIGH/CRITICAL events trigger in-dashboard alerts and optional Telegram push notifications with annotated snapshots. |
| **Worker Face Verification** | Siamese one-shot-model identity layer over detected Person boxes: multi-worker registry (`workers.json` + reference crops), per-track result caching, calibrated two-threshold verification, green/red identity overlay, and `/face/*` REST endpoints. Fully optional and failure-isolated. |
| **Evidence Capture** | Automatic annotated JPEG snapshots and rolling pre-buffer MP4 clips on HIGH/CRITICAL violations, with bounded storage and auto-pruning. |
| **Multi-Tab Dashboard** | Vanilla JS SPA with live camera, alerts feed, event history, statistics charts, evidence gallery, and downloadable reports. |
| **Detailed CSV Logging** | Exports comprehensive frame-by-frame risk indices, active violation counts, and hazard categories for post-job safety audits. |

---

## 🛠️ Technologies & Tools

| Category | Technology | Purpose in Project |
|---|---|---|
| **Language** | **Python 3.10 / 3.11** | Core platform programming language. |
| **Deep Learning** | **Ultralytics YOLO (v8 / v11)** | Real-time object detection and multi-object tracking (`model.track()`). |
| **Framework** | **PyTorch (torch / torchvision)** | Deep learning inference backend with automatic CPU / CUDA GPU acceleration. |
| **Web Framework** | **FastAPI & Uvicorn** | High-performance asynchronous REST API server and WebSocket streaming handler. |
| **Computer Vision** | **OpenCV (`opencv-python`)** | Video capture, frame decoding/encoding, video writing, and overlay rendering. |
| **Visualization** | **Ultralytics Annotator / CVZone** | Native bounding box rendering, label badges, and custom HUD banners. |
| **Computational Geometry**| **Shapely** | Polygon creation (convex hulls), polygon intersection, and point-in-polygon testing. |
| **Machine Learning** | **Scikit-Learn (HDBSCAN)** | Unsupervised density-based clustering of safety cones into dangerous zones. |
| **Frontend** | **HTML5 / CSS3 / Vanilla JavaScript** | Responsive web interface, WebSocket camera streaming, and dynamic metrics cards. |

---

## 📐 Project Architecture & Directory Structure

```text
construction_safety_system/
├── app/                                 # FastAPI application and backend services
│   ├── __init__.py
│   ├── main.py                          # Core server: REST endpoints, WebSocket, static mounts
│   ├── alerting.py                      # Alert engine with Dashboard feed + Telegram push
│   ├── evidence.py                      # Snapshot + clip recorder for HIGH/CRITICAL events
│   ├── monitor.py                       # SafetyMonitor: per-stream orchestrator
│   ├── reports.py                       # Time-range aggregation and CSV export
│   ├── settings.py                      # Configuration from environment variables
│   └── storage.py                       # Thread-safe SQLite persistence layer
├── data/                                # Runtime data (auto-created)
│   ├── safety.db                        # SQLite database (events, alerts, counters)
│   ├── evidence/                        # Captured snapshots and video clips
│   ├── face_data/                       # Worker identity data (face recognition)
│   │   ├── workers.json                 # Worker registry metadata (auto-managed)
│   │   ├── input_images/<worker_id>/    # Reference face crops per worker
│   │   ├── embeddings/<worker_id>.npy   # Embedding cache (auto-managed)
│   │   └── calibration.json             # Calibrated verification thresholds
│   ├── images/                          # Sample test images
│   ├── output/                          # Output directory
│   │   └── videos/                      # Generated annotated videos and CSV reports
│   └── videos/                          # Benchmark and test video recordings
├── docs/                                # Documentation
│   └── DETECTION_LIMITATIONS.md         # Code-fixable vs model-bound limitations
├── frontend/                            # Web dashboard user interface
│   ├── app.js                           # Multi-tab SPA: Dashboard, Alerts, History, Stats, Evidence, Reports
│   ├── index.html                       # Real-time monitoring dashboard layout
│   └── style.css                        # Modern dark-mode theme styling
├── models/                              # Trained YOLO model weights (.pt files)
│   ├── fire_smoke/
│   │   └── best.pt                      # Fire & Smoke detection model (classes: Fire, Smoke)
│   └── hazard/
│       └── best.pt                      # Unified hazard model (workers, PPE, cones, machines, poles)
├── scripts/                             # Standalone command-line utilities
│   ├── analyze_video.py                 # Offline video analysis with CSV export and filter telemetry
│   ├── live_webcam.py                   # Direct OpenCV webcam / camera streaming utility
│   ├── register_worker.py               # Worker identity registration CLI (files/webcam/list/remove)
│   ├── calibrate_threshold.py           # Siamese verification threshold calibration
│   └── face_verify_webcam.py            # Standalone live multi-worker face verification
├── src/                                 # Core business logic and safety algorithms
│   ├── pipeline.py                      # Shared SafetyPipeline: single per-frame code path
│   ├── fire/                            # Fire & smoke detection module
│   │   ├── __init__.py
│   │   ├── config.py                    # Fire model configuration
│   │   ├── fire_confirmation.py         # Multi-frame IoU persistence debouncer for fire/smoke
│   │   └── fire_detector.py             # Inference wrapper around fire/smoke YOLO model
│   ├── hazard/                          # Hazard and worker detection module
│   │   ├── __init__.py
│   │   ├── detection_filter.py          # Resolution-aware, perspective-aware, and aspect-ratio filter
│   │   ├── hazard_detector.py           # Inference wrapper with ByteTrack tracking integration
│   │   └── track_confirmation.py        # Track hit buffer + StaticPersonFilter motion analyzer
│   ├── machine/                         # Heavy machinery module
│   │   └── machine_detector.py          # Heavy equipment inference wrapper
│   ├── safety/                          # Central safety rules and geometry engine
│   │   ├── __init__.py
│   │   ├── distance_calculator.py       # Ground-contact proximity calculation
│   │   ├── geometry.py                  # HDBSCAN cone clustering, Shapely polygons, DangerZoneTracker
│   │   ├── overlay.py                   # High-contrast visual annotation and risk status banner
│   │   ├── rules.py                     # SafetyConfig, PPE/zone/proximity rules, risk assessment
│   │   └── safety_engine.py             # Central orchestrator combining all hazard subsystems
│   └── face_recognition/                # Worker identity / Siamese verification layer
│       ├── __init__.py
│       ├── config.py                    # FaceRecognitionConfig (env-overridable tunables)
│       ├── preprocessing.py             # Notebook-exact face crop preprocessing
│       ├── backends.py                  # SiameseFaceBackend (TF) + StubFaceBackend (tests)
│       ├── registry.py                  # WorkerRegistry: workers.json, reference images, embeddings
│       ├── face_detector.py             # Person box -> head-band face crops
│       ├── service.py                   # FaceRecognitionService: multi-worker verification
│       ├── overlay.py                   # Identity boxes/labels + FACES HUD banner
│       └── calibration.py               # Threshold calibration -> calibration.json
├── docs/
│   ├── DETECTION_LIMITATIONS.md         # Code-fixable vs model-bound limitations
│   └── FACE_RECOGNITION.md              # Worker face recognition: data layout, registration,
│                                        #   model loading, verification, thresholds, REST API
├── tests/                               # Pytest test suite (140+ model-free unit tests)
│   ├── conftest.py                      # Test fixtures and helpers
│   ├── test_face_recognition.py         # Face recognition layer tests (stub backend)
│   ├── test_pipeline_identity.py        # Pipeline identity-integration tests
│   ├── test_alerting.py                 # Alert engine tests
│   ├── test_detection_filter.py         # Detection filter tests
│   ├── test_distance.py                 # Distance calculator tests
│   ├── test_evidence.py                 # Evidence capture tests
│   ├── test_fire_confirmation.py        # Fire temporal debounce tests
│   ├── test_geometry.py                 # HDBSCAN clustering tests
│   ├── test_monitor.py                  # SafetyMonitor orchestrator tests
│   ├── test_pipeline.py                 # Shared pipeline tests
│   ├── test_reports.py                  # Reports module tests
│   ├── test_rules.py                    # Safety rules tests
│   ├── test_safety_engine.py            # Safety engine tests
│   ├── test_storage.py                  # SQLite storage tests
│   └── test_track_confirmation.py       # Track confirmation tests
├── .env.example                         # Environment configuration template
├── .gitignore                           # Git ignore rules
├── pytest.ini                           # Pytest configuration
├── requirements.txt                     # Project dependencies
└── README.md                            # Official documentation
```

---

## ⚙️ Environment Setup

### Prerequisites
- **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS
- **Python**: Version `3.10` or `3.11` (Python 3.11.8 recommended)
- **Optional GPU Acceleration**: NVIDIA GPU with CUDA 11.8+ and cuDNN installed

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/construction_safety_system.git
cd construction_safety_system
```

### 2. Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> **Note for Windows Users**: If script execution is restricted, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*(Optional) For CUDA GPU Support with PyTorch:*
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 🧠 Model & Weights Setup

The system uses two YOLO models, both pre-configured in the `models/` directory:

| Model File | Target Classes | Description |
|---|---|---|
| `models/hazard/best.pt` | `Hardhat`, `Mask`, `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`, `Person`, `Safety Cone`, `Safety Vest`, `machinery`, `utility pole`, `vehicle` | Primary multi-hazard detector with ByteTrack tracking. Covers workers, PPE compliance, cones, machinery, and poles in a single model. |
| `models/fire_smoke/best.pt` | `Fire`, `Smoke` | Specialized model for thermal and combustion hazards with temporal debouncing. |

Both models are loaded once at server startup and shared across all streams (REST endpoints, WebSocket camera, video jobs). They serialize their own forward passes behind an internal lock, so concurrent requests never race on the shared model object.

---

## 🚦 How to Run the System

### 1. Web Application & Live Dashboard (Recommended)

The web application is a modern React + TypeScript SPA (TanStack Router, Tailwind v4, shadcn/ui) with a blue/white industrial design and a dark control-room theme. It ships pre-built: FastAPI serves it directly from `frontend/.output`.

**Start the server:**
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

**Open your browser:**
- **Dashboard**: `http://127.0.0.1:8000` — live status, risk distribution, PPE/face-recognition status, recent events
- **Live Monitor**: `/monitor` — webcam through the full AI pipeline (YOLO + ByteTrack + Siamese worker verification) with per-face Verified/Unknown badges, PPE and hazard panels
- **Workers**: `/workers` — register / add faces / deactivate / purge workers (writes `data/face_data/`)
- **History**: `/history` — filterable event log + alert feed with acknowledgement
- **Reports**: `/reports` — backend-computed aggregates + CSV/JSON export
- **API Documentation**: `http://127.0.0.1:8000/docs` — Interactive Swagger UI for all REST endpoints

> To develop the frontend instead of serving the built bundle:
> ```bash
> cd frontend && bun install && bun run dev   # dev server on :5173, proxies API to :8000
> bun run build:spa                           # production build (client-only SPA shell + assets in .output/public) served by FastAPI
> bun run build                               # SSR build (separate Node server process) — NOT served by FastAPI
> ```

#### Using Live Webcam in the Dashboard

1. Navigate to `http://127.0.0.1:8000`
2. Click the **"Dashboard"** tab (default)
3. Click **"Start Live Camera"** button
4. Grant camera permission when prompted by your browser
5. The live feed processes through the full AI safety pipeline in real-time
6. Violations are automatically recorded to the database
7. HIGH/CRITICAL events trigger alerts and capture evidence (snapshots + clips)

**What happens during live streaming:**
- Every frame runs through hazard detection, fire/smoke detection, PPE checking, proximity analysis, and danger zone monitoring
- Risk levels are calculated per frame (SAFE → LOW → MEDIUM → HIGH → CRITICAL)
- Events are recorded to SQLite with cooldown-bounded deduplication
- Annotated JPEG snapshots and short MP4 clips are saved for HIGH/CRITICAL events
- Alerts appear in the Alerts tab and can be pushed to Telegram (if configured)
- Statistics update in real-time across all dashboard tabs

**Dashboard tabs:**
- **Dashboard**: Live camera feed with real-time risk indicators
- **Alerts**: Feed of HIGH/CRITICAL alerts with acknowledgment
- **History**: Searchable event log with risk/source filters
- **Statistics**: Time-range aggregates with bar charts
- **Evidence**: Gallery of captured snapshots and video clips
- **Reports**: Downloadable CSV reports with event summaries

---

### 2. Offline Video Safety Analysis CLI

Processes an existing video file on disk, applies detection filters, tracks entities, evaluates danger zones and proximity, and renders an annotated video plus CSV log:

```bash
python scripts/analyze_video.py --video data/videos/test1.mp4
```

**Customizing Parameters:**
```bash
python scripts/analyze_video.py \
    --video data/videos/test_machine2.mp4 \
    --output data/output/videos/annotated_result.mp4 \
    --csv data/output/videos/summary_log.csv \
    --hazard-conf 0.25 \
    --fire-conf 0.20 \
    --machine-distance 250.0 \
    --track-confirm-min-hits 3 \
    --show-progress-every 30
```

---

### 3. Real-Time Webcam / RTSP Stream CLI (Standalone)

Runs direct OpenCV video capture without web dependencies. This is a lightweight alternative to the web dashboard for quick testing. Press `Q` to exit:

```bash
# Default camera (index 0)
python scripts/live_webcam.py

# Specify camera index (e.g., second camera)
python scripts/live_webcam.py --camera 1

# RTSP stream from IP camera
python scripts/live_webcam.py --camera "rtsp://admin:password@192.168.1.100:554/stream"

# Run hazard model only (skip fire model for speed)
python scripts/live_webcam.py --camera 0 --skip-fire

# Adjust confidence thresholds
python scripts/live_webcam.py --camera 0 --hazard-conf 0.30 --fire-conf 0.25
```

**Note:** This standalone script does NOT record events to the database or capture evidence. For full monitoring with persistence, use the web dashboard (option 1).

---

### 4. Worker Face Recognition (Siamese verification)

Identify **which registered worker** is on camera. Full guide:
**[docs/FACE_RECOGNITION.md](docs/FACE_RECOGNITION.md)**.

```bash
# 1. Register workers (5-10 face crops each; webcam capture also available)
python scripts/register_worker.py --worker-id worker_001 \
    --name "Ahmed Ali" --role "crane operator" \
    --images crops/ahmed1.jpg crops/ahmed2.jpg "crops/ahmed_extra/*.jpg"

# 2. (Recommended) calibrate thresholds on YOUR workers
python scripts/calibrate_threshold.py

# 3. Live verification demo (identity only)
python scripts/face_verify_webcam.py

# 4. Safety pipeline + identities together
python scripts/live_webcam.py --face-recognition
python scripts/analyze_video.py --video data/videos/test1.mp4 --face-recognition

# 5. Full dashboard: identities arrive in the websocket payload and are
#    drawn on the live feed automatically when workers are registered.
```

REST: `GET/POST /face/workers`, `POST /face/workers/{id}/images`,
`DELETE /face/workers/{id}`, `POST /face/verify`, `POST /face/calibrate`.

**Where worker data lives:** `data/face_data/input_images/<worker_id>/*.jpg`
(reference crops) + `data/face_data/workers.json` (registry) - adding or
removing a worker never requires touching source code. The model
(`notebooks/siamesemodelv2.h5`) is loaded once at startup and shared;
missing TensorFlow or model file simply disables the feature.

---

### 5. REST API Image & Video Endpoints

#### Process a Single Image:
```bash
curl -X POST "http://127.0.0.1:8000/safety/detect/image" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/images/test.webp"
```

#### Process a Video File:
```bash
curl -X POST "http://127.0.0.1:8000/safety/detect/video" \
     -H "accept: video/mp4" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/videos/test1.mp4" \
     --output "annotated_video.mp4"
```

---

## ⚙️ Environment Configuration

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

**Key configuration options:**

```bash
# Storage paths
SAFETY_DB_PATH=data/safety.db
SAFETY_EVIDENCE_DIR=data/evidence

# Risk thresholds (SAFE < LOW < MEDIUM < HIGH < CRITICAL)
SAFETY_RECORD_MIN_RISK=LOW          # Minimum risk to record event
SAFETY_ALERT_MIN_RISK=HIGH          # Minimum risk to trigger alert
SAFETY_EVIDENCE_MIN_RISK=HIGH       # Minimum risk to capture snapshot/clip

# Cooldowns (seconds) - prevent spam
SAFETY_EVENT_COOLDOWN_S=10
SAFETY_ALERT_COOLDOWN_S=60
SAFETY_EVIDENCE_COOLDOWN_S=30

# Video clip recording
SAFETY_CLIP_PRE_FRAMES=30           # Frames to keep before trigger
SAFETY_CLIP_POST_FRAMES=60          # Frames to record after trigger
SAFETY_CLIP_FPS=10

# Pruning (auto-cleanup old data)
SAFETY_MAX_EVIDENCE_FILES=1000
SAFETY_MAX_EVENTS=10000

# Telegram alerts (optional)
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
SAFETY_TELEGRAM_ENABLED=true
```

**Important:** Never commit `.env` to git. It's already in `.gitignore`.

---

## 🎛️ Configuration Reference

### CLI Flags (`scripts/analyze_video.py`)

| Flag | Type | Default | Description |
|---|---|---|---|
| `--video` | `str` | *Required* | Path to the source MP4/AVI video file. |
| `--output` | `str` | `auto` | Destination path for the annotated video output. |
| `--csv` | `str` | `auto` | Destination path for the per-frame analytical CSV summary. |
| `--hazard-model` | `str` | `models/hazard/best.pt` | Path to unified hazard YOLO weights. |
| `--fire-model` | `str` | `models/fire_smoke/best.pt` | Path to fire/smoke YOLO weights. |
| `--hazard-conf` | `float` | `0.25` | Base inference confidence for hazard YOLO model. |
| `--fire-conf` | `float` | `0.20` | Base inference confidence for fire/smoke YOLO model. |
| `--machine-distance` | `float` | `250.0` | Proximity threshold (in pixels) for Worker ↔ Machine alerts. |
| `--pole-distance` | `float` | `250.0` | Proximity threshold (in pixels) for Machine ↔ Pole alerts. |
| `--ppe-overlap` | `float` | `0.05` | Minimum overlap ratio for PPE-to-body anatomical matching. |
| `--cone-min-cluster-size`| `int` | `3` | Minimum cones required to form a danger zone with HDBSCAN. |
| `--track-confirm-min-hits`| `int`| `3` | Consecutive frames a person must be detected before reporting. |
| `--disable-detection-filter`| `flag`| `False` | Disables size, perspective, and aspect ratio filters. |
| `--disable-spatial-ppe` | `flag` | `False` | Reverts to full-body overlap instead of anatomical head/torso matching. |
| `--skip-fire` | `flag` | `False` | Skips fire/smoke inference for higher throughput. |
| `--max-frames` | `int` | `None` | Process up to N frames (useful for rapid testing). |

---

## 📊 Input & Output Specifications

### Supported Inputs
- **Static Images**: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`
- **Video Files**: `.mp4`, `.avi`, `.mov`, `.mkv`
- **Live Feeds**: USB Webcams, Virtual Cameras, RTSP/HTTP surveillance streams

### Generated Outputs
1. **Annotated Video (`.mp4`)**:
   - Color-coded bounding boxes matching native YOLO model palettes.
   - Persistent tracking labels: `ID:<num> <Class> <Conf>`.
   - Translucent red polygon fills highlighting active **DANGER ZONES**.
   - Red ground-contact proximity vectors between workers and machinery.
   - High-visibility top status banner with **RISK LEVEL** and active violation counters.
2. **Analytical CSV Summary (`.csv`)**:
   Contains per-frame telemetry:
   `frame, risk_level, violation_count, violation_types, persons, machines, danger_zones, zone_ids, fire_detections`

---

## 🏷️ Detection Classes & Hazard Categories

```
                    ┌── Person (Tracked Worker)
                    ├── machinery (Excavator, Mixer, Crane)
                    ├── vehicle (Truck, Car, Van)
  Normal Entities   ├── utility pole (Power & Telephone Poles)
                    ├── Safety Cone (Zone Boundary Marker)
                    ├── Hardhat & Safety Vest (Compliant PPE)
                    └── Mask (Compliant Facial Protection)
                    
                    ┌── NO-Hardhat (Worker missing helmet)
                    ├── NO-Safety Vest (Worker missing hi-vis vest)
  Safety Hazards    ├── NO-Mask (Missing face mask)
                    ├── DANGEROUS_ZONE (Worker inside cone boundary)
                    ├── MACHINE_PROXIMITY (Worker < 250px from machinery)
                    ├── POLE_PROXIMITY (Boom/Machine < 250px from power pole)
                    └── Fire & Smoke (Active combustion detection)
```

### Risk Level Hierarchy

| Risk Level | Condition Trigger | Visual Indicator |
|---|---|---|
| 🟢 **SAFE** | Zero active violations or hazards detected. | Green HUD banner |
| 🟡 **LOW** | Minor non-critical PPE omission (e.g. missing mask). | Yellow HUD banner |
| 🟠 **MEDIUM** | Standard PPE violation (`NO_HARDHAT` or `NO_SAFETY_VEST`). | Orange HUD banner |
| 🔴 **HIGH** | Proximity breach (`MACHINE_PROXIMITY`, `POLE_PROXIMITY`) or `DANGEROUS_ZONE` incursion. | Deep Orange / Red HUD |
| 🚨 **CRITICAL** | Confirmed `Fire`, or compound hazard (worker inside danger zone + machine proximity). | Pulsing Red HUD banner |

---

## 🔄 End-to-End Pipeline Workflow

```
       Incoming Frame (Webcam / Video / Image)
                         │
                         ├─────────────────────────────────────────┐
                         ▼                                         ▼
            Hazard Model (YOLO + ByteTrack)               Fire Model (YOLO)
                         │                                         │
                         ▼                                         ▼
            Stateless Detection Filter                     Temporal Confirmation
         (Confidence + Resolution Size +                  (Area + Confidence +
          Perspective + Aspect Ratio)                      Consecutive Streak)
                         │                                         │
                         ▼                                         ▼
            TrackConfirmationTracker                        Confirmed Fire &
         (Debounce Gate + StaticPersonFilter)                     Smoke
                         │                                         │
                         └────────────────────┬────────────────────┘
                                              ▼
                                 Anatomical PPE Association
                                 (Head: 30% | Torso: 25-75%)
                                              ▼
                                   Geometric Calculations
                                  (HDBSCAN Cone Clusters +
                                   Ground Distance to Machines)
                                              ▼
                                        SafetyEngine
                               (Violation Evaluation & Rules)
                                              ▼
                                  Final Risk Assessment
                           (SAFE / LOW / MEDIUM / HIGH / CRITICAL)
                                              ▼
                                     Annotation Overlay
                                              ▼
                                   Video Output / Web Stream
```

---

## 🔧 Troubleshooting & Common Issues

#### 1. `ImportError: cannot import name 'replace' from 'dataclasses'`
- **Cause**: Outdated Python or syntax typo.
- **Solution**: Ensure Python version is $\ge 3.10$. Run `python --version` to verify.

#### 2. `SyntaxError in distance_calculator.py`
- **Cause**: Corrupted dictionary formatting.
- **Solution**: Pull latest changes; ensure line 212 of `src/safety/distance_calculator.py` properly terminates with `}`.

#### 3. Ground stake or pole detected as `Person` (`ID:504`)
- **Cause**: High-resolution video where tiny objects exceed static 30px thresholds.
- **Solution**: The detection filter now applies resolution scaling and `StaticPersonFilter`. Ensure you are not running with `--disable-detection-filter`.

#### 4. `[WinError 10013] An attempt was made to access a socket in a way forbidden`
- **Cause**: Port 8000 is occupied by a stale Python/Uvicorn process.
- **Solution (PowerShell)**:
  ```powershell
  Get-Process python | Stop-Process -Force
  uvicorn app.main:app --port 8000 --reload
  ```

#### 5. OpenCV VideoCapture fails to open camera
- **Cause**: Incorrect camera index or OS camera privacy block.
- **Solution**: Try camera index 1 (`python scripts/live_webcam.py --camera 1`) and verify Windows Camera Privacy Settings allow desktop apps.

#### 6. Face recognition unavailable / all faces show `Unknown`
- **Cause a**: TensorFlow not installed in the running environment.
- **Solution a**: `pip install -r requirements.txt` (installs `tensorflow-cpu` + `tf-keras`). The trained `.h5` is a TF-2.4-era graph and needs `tf-keras` (official Keras 2) to load on modern TensorFlow — the loader handles this automatically.
- **Cause b**: No workers registered.
- **Solution b**: `python scripts/register_worker.py --worker-id worker_001 --name "Name" --images crops/*.jpg` (or create `data/face_data/input_images/<worker_id>/` with jpgs).
- **Cause c**: Thresholds too strict for your footage.
- **Solution c**: `python scripts/calibrate_threshold.py`, or raise/lower `FACE_DETECTION_THRESHOLD` / `FACE_VERIFICATION_THRESHOLD` in `.env`.

---

## 💡 Performance & Detection Tuning Tips

1. **Resolution vs Speed**:
   - For real-time 30+ FPS performance on CPU, pass `--hazard-imgsz 480` or `--hazard-imgsz 640`.
   - For high-altitude drone footage where workers are tiny, use `--hazard-imgsz 960` or `1280`.
2. **Tuning Machine Proximity**:
   - If workers legitimately work near machines, adjust `--machine-distance 180` (default: `250.0`).
3. **Suppressing Background Objects**:
   - In static elevated cameras, define a rectangular Region of Interest (`build_rectangular_roi`) to exclude background roads or housing from triggering machinery detections.

---

## 📌 Quick Reference Paths

```text
Hazard Model Weights   : models/hazard/best.pt
Fire/Smoke Weights     : models/fire_smoke/best.pt
Siamese Face Model     : notebooks/siamesemodelv2.h5
Worker Registry        : data/face_data/workers.json
Worker Reference Faces : data/face_data/input_images/<worker_id>/*.jpg
Face Docs              : docs/FACE_RECOGNITION.md
Core FastAPI App       : app/main.py
Alert Engine           : app/alerting.py
Evidence Capture       : app/evidence.py
Safety Monitor         : app/monitor.py
Reports Module         : app/reports.py
Settings               : app/settings.py
SQLite Storage         : app/storage.py
Shared Pipeline        : src/pipeline.py
Face Recognition Layer : src/face_recognition/
Video Batch Script     : scripts/analyze_video.py
Live Webcam Script     : scripts/live_webcam.py
Worker Registration    : scripts/register_worker.py
Threshold Calibration  : scripts/calibrate_threshold.py
Face Verify Demo       : scripts/face_verify_webcam.py
Detection Filter Logic : src/hazard/detection_filter.py
Tracking & Debouncing  : src/hazard/track_confirmation.py
Web Dashboard Assets   : frontend/ (index.html, app.js, style.css)
Test Suite             : tests/ (120+ model-free unit tests)
Environment Config     : .env (copy from .env.example)
SQLite Database        : data/safety.db (auto-created)
Evidence Storage       : data/evidence/ (auto-created)
Default Output Videos  : data/output/videos/
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

