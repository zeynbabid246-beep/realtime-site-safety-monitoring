# 🏗️ AI-Powered Construction Safety Monitoring & Hazard Detection System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0.0-009688.svg)](https://fastapi.tiangolo.com/)
[![YOLOv8/v11](https://img.shields.io/badge/YOLO-Ultralytics-00FFFF.svg)](https://docs.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent, real-time computer vision and spatial intelligence platform designed for construction site safety monitoring. The system combines multi-model deep learning (YOLO), ByteTrack object tracking, unsupervised geometric clustering (HDBSCAN), and anatomical spatial association to automatically detect safety violations, heavy machinery proximity hazards, unauthorized danger-zone incursions, and early fire/smoke incidents.

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
  - [4. REST API Image & Video Endpoints](#4-rest-api-image--video-endpoints)
- [Configuration Reference](#-configuration-reference)
- [Input & Output Specifications](#-input--output-specifications)
- [Detection Classes & Hazard Categories](#-detection-classes--hazard-categories)
- [End-to-End Pipeline Workflow](#-end-to-end-pipeline-workflow)
- [Troubleshooting & Common Issues](#-troubleshooting--common-issues)
- [Performance & Detection Tuning Tips](#-performance--detection-tuning-tips)
- [Quick Reference Paths](#-quick-reference-paths)

---

## 🚀 Quick Start

Run the entire safety platform in four copy-pasteable commands:

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

# 4. Launch the web platform and API
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and navigate to:
👉 **`http://127.0.0.1:8000`** — Live interactive camera dashboard & hazard analytics.  
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
6. **Overall Risk Index**: Aggregates all concurrent hazards into a standardized safety risk rating: **`SAFE`**, **`LOW`**, **`MEDIUM`**, **`HIGH`**, or **`CRITICAL`**.

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
├── app/                                 # FastAPI application and API routes
│   ├── __init__.py
│   └── main.py                          # Core server: REST endpoints, WebSocket handler, static mount
├── data/                                # Data storage for media and outputs
│   ├── images/                          # Sample test images
│   ├── output/                          # Output directory
│   │   └── videos/                      # Generated annotated videos and CSV reports
│   └── videos/                          # Benchmark and test video recordings
├── frontend/                            # Web dashboard user interface
│   ├── app.js                           # WebSocket camera feed handler and DOM updater
│   ├── index.html                       # Real-time monitoring dashboard layout
│   └── style.css                        # Modern dark-mode theme styling
├── models/                              # Trained YOLO model weights (.pt files)
│   ├── fire_smoke/
│   │   └── best.pt                      # Fire & Smoke detection model (classes: Fire, Smoke)
│   ├── hazard/
│   │   └── best.pt                      # Unified hazard model (workers, PPE, cones, machines, poles)
│   ├── machine/
│   │   └── best.pt                      # Heavy machinery model (excavator, crane, mixer, etc.)
│   └── ppe/
│       └── best.pt                      # Standalone PPE model (helmet, vest, boots, gloves, goggles)
├── scripts/                             # Standalone command-line utilities
│   ├── analyze_video.py                 # Offline video analysis with CSV export and filter telemetry
│   └── live_webcam.py                   # Direct OpenCV webcam / camera streaming utility
├── src/                                 # Core business logic and safety algorithms
│   ├── fire/                            # Fire & smoke detection module
│   │   ├── config.py                    # Fire model configuration
│   │   ├── fire_confirmation.py         # Multi-frame IoU persistence debouncer for fire/smoke
│   │   └── fire_detector.py             # Inference wrapper around fire/smoke YOLO model
│   ├── hazard/                          # Hazard and worker detection module
│   │   ├── detection_filter.py          # Resolution-aware, perspective-aware, and aspect-ratio filter
│   │   ├── hazard_detector.py           # Inference wrapper with ByteTrack tracking integration
│   │   └── track_confirmation.py        # Track hit buffer + StaticPersonFilter motion analyzer
│   ├── machine/                         # Heavy machinery module
│   │   └── machine_detector.py          # Heavy equipment inference wrapper
│   └── safety/                          # Central safety rules and geometry engine
│       ├── distance_calculator.py       # Ground-contact proximity calculation
│       ├── geometry.py                  # HDBSCAN cone clustering, Shapely polygons, DangerZoneTracker
│       ├── overlay.py                   # High-contrast visual annotation and risk status banner
│       ├── rules.py                     # SafetyConfig, PPE/zone/proximity rules, risk assessment
│       └── safety_engine.py             # Central orchestrator combining all hazard subsystems
├── test_danger_zone_video.py            # Danger zone integration test script
├── test_detection_filter.py             # Unit test suite for detection filtering and motion checks
├── test_distance_video.py               # Worker-machine proximity integration test
├── test_fire_confirmation.py            # Unit test for fire temporal debounce gate
├── test_geometry.py                     # Unit test for HDBSCAN clustering and polygon creation
├── test_safety_engine.py                # End-to-end integration test for SafetyEngine
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

All weights are pre-configured in the `models/` directory:

| Model File | Target Classes | Description |
|---|---|---|
| `models/hazard/best.pt` | `Hardhat`, `Mask`, `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`, `Person`, `Safety Cone`, `Safety Vest`, `machinery`, `utility pole`, `vehicle` | Primary multi-hazard detector with tracking. |
| `models/fire_smoke/best.pt` | `Fire`, `Smoke` | Specialized model for thermal and combustion hazards. |
| `models/machine/best.pt` | `excavator`, `dump_truck`, `bulldozer`, `wheel_loader`, `mobile_crane`, `tower_crane`, `roller_compactor`, `cement_mixer` | Granular classification of construction machinery. |
| `models/ppe/best.pt` | `helmet`, `gloves`, `vest`, `boots`, `goggles`, `none`, `Person`, `no_helmet`, `no_goggle`, `no_gloves`, `no_boots` | Granular multi-class PPE audit model. |

---

## 🚦 How to Run the System

### 1. Web Application & Live Dashboard (FastAPI)
Launches the web server serving the frontend UI, REST API, and WebSocket streaming:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

- Open **`http://127.0.0.1:8000`** in Google Chrome or Edge.
- Click **"Start Live Camera"** to stream your webcam directly through the real-time AI safety pipeline.

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

### 3. Real-Time Webcam / RTSP Stream CLI
Runs direct OpenCV video capture without web dependencies. Press `Q` to exit:

```bash
# Default camera (index 0)
python scripts/live_webcam.py

# Specify camera index or RTSP stream URL
python scripts/live_webcam.py --camera 1

# Run hazard model only (skip fire model for speed)
python scripts/live_webcam.py --camera 0 --skip-fire
```

---

### 4. REST API Image & Video Endpoints

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
Machinery Weights      : models/machine/best.pt
PPE Model Weights      : models/ppe/best.pt
Core FastAPI App       : app/main.py
Video Batch Script     : scripts/analyze_video.py
Live Webcam Script     : scripts/live_webcam.py
Detection Filter Logic : src/hazard/detection_filter.py
Tracking & Debouncing  : src/hazard/track_confirmation.py
Web Dashboard Assets   : frontend/ (index.html, app.js, style.css)
Default Output Videos  : data/output/videos/
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

