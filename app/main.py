"""
Construction Safety AI - FastAPI application.

CHANGELOG (vs original)
------------------------
- Retired the standalone /detect/ppe/* endpoints and the legacy
  models/ppe/best.pt model. PPE is now detected as part of the unified
  hazard model and goes through the full SafetyEngine like every other
  hazard type, instead of being a separate, disconnected code path.
- New unified pipeline endpoints:
    POST /safety/detect/image   - single image, full pipeline
    POST /safety/detect/video   - full video, full pipeline, annotated output
    WS   /safety/ws/camera      - live camera, full pipeline, streamed
    POST /detect/fire/image     - kept standalone (useful for isolated
                                   fire-model debugging/testing)
- Every per-frame processing step (video loop, websocket loop) is now
  wrapped in its own try/except so ONE malformed frame logs a warning
  and gets skipped instead of killing the entire video job or dropping
  the whole websocket connection - this matters once you're running
  against real, messy construction footage instead of clean synthetic
  test videos.
- Each video job / websocket connection gets its OWN SafetyEngine +
  DangerZoneTracker instance, so danger-zone ids stay stable across
  that stream's frames without leaking state between unrelated
  uploads/streams (a shared global engine would cross-contaminate
  zone tracking between different people's videos).
- HazardDetector.track() uses ByteTrack persistently within a single
  video/stream (persist=True) so the SAME person keeps the SAME
  track_id across frames, which is required for anything that wants to
  reason about a specific worker over time (dangerous-zone dwell time,
  repeated-violation alerts, etc.).
"""

from __future__ import annotations

import base64
import logging
import shutil
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.hazard.hazard_detector import HazardDetector
from src.hazard.detection_filter import filter_detections
from src.hazard.track_confirmation import TrackConfirmationTracker
from src.fire.fire_detector import FireDetector
from src.fire.fire_confirmation import FireConfirmationTracker
from src.safety.safety_engine import SafetyEngine
from src.safety.geometry import DangerZoneTracker
from src.safety.rules import SafetyConfig
from src.safety.overlay import draw_safety_overlay


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("construction_safety")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

HAZARD_MODEL_PATH = BASE_DIR / "models" / "hazard" / "best.pt"
FIRE_MODEL_PATH = BASE_DIR / "models" / "fire_smoke" / "best.pt"

OUTPUT_VIDEO_DIR = BASE_DIR / "data" / "output" / "videos"
OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

FRONTEND_DIR = BASE_DIR / "frontend"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Construction Safety AI",
    description="AI-powered construction site safety detection system",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD MODELS (once, at startup)
# ============================================================

logger.info("Loading hazard model...")
hazard_detector = HazardDetector(HAZARD_MODEL_PATH, confidence=0.25, image_size=640)
logger.info("Hazard classes: %s", hazard_detector.class_names)

logger.info("Loading fire/smoke model...")
fire_detector = FireDetector(FIRE_MODEL_PATH, confidence=0.20, image_size=640)
logger.info("Fire/smoke classes: %s", fire_detector.class_names)

DEFAULT_SAFETY_CONFIG = SafetyConfig()

logger.info("Models loaded successfully.")


# ============================================================
# BASIC ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    return {"status": "healthy"}


# ============================================================
# SHARED PIPELINE HELPER
# ============================================================

def run_pipeline_on_frame(
    frame: np.ndarray,
    engine: SafetyEngine,
    track: bool = True,
    fire_tracker: FireConfirmationTracker = None,
    track_confirmation: TrackConfirmationTracker = None,
):
    """
    Run one frame through hazard detection + fire detection + the
    safety engine, and return (result, persons, machines, annotated_frame).

    `track=True` uses ByteTrack (persist=True) - use this for video/
    camera streams. `track=False` runs plain detection - use this for
    single standalone images where there is no "next frame" to track
    into.

    `fire_tracker`, if given, gates raw fire/smoke detections through
    FireConfirmationTracker before they reach the safety engine, so a
    single low-confidence/tiny/one-frame false positive can't push the
    risk level to CRITICAL or get logged as a real fire event. Pass
    None to skip confirmation (e.g. for a single standalone image,
    where there's no "next frame" to build persistence from anyway).

    `track_confirmation`, if given, requires a hazard detection's
    track_id to have persisted for several frames before it's used for
    anything (person extraction, PPE association, drawing, the safety
    engine) - this is what stops a one-frame flicker like "ID:15
    Person 0.37" on a thin pole from ever being treated as a real
    worker. Pass None for the same reason as fire_tracker above.

    Pipeline order matters here and matches the architecture: raw YOLO
    output is filtered (confidence/size/aspect-ratio - drops most
    single-frame junk like "ID:714 Person 0.42" on the first frame it
    ever appears) BEFORE track confirmation gets a chance to build a
    streak for it, so a detection that's obviously too small/low-
    confidence to be real never even starts accumulating hits.
    """

    detections = hazard_detector.track(frame) if track else hazard_detector.predict(frame)

    detections = filter_detections(detections, frame_shape=frame.shape[:2])

    if track_confirmation is not None:
        detections = track_confirmation.update(detections, frame_shape=frame.shape[:2])

    raw_fire_detections = fire_detector.predict(frame)

    fire_detections = (
        fire_tracker.update(raw_fire_detections) if fire_tracker is not None
        else raw_fire_detections
    )

    persons = hazard_detector.extract_persons(detections)
    machines = hazard_detector.extract_machines(detections)

    result = engine.analyze(
        detections=detections,
        persons=persons,
        machines=machines,
        fire_detections=fire_detections,
        frame_width=float(frame.shape[1]),
    )

    annotated_frame = draw_safety_overlay(
        frame, result, detections, hazard_detector.class_names
    )

    return result, persons, machines, annotated_frame


def _serialize_result(result: dict) -> dict:
    """
    Strip non-JSON-serializable objects (shapely Polygons) out of a
    SafetyEngine result before sending it over HTTP/WebSocket, while
    keeping the useful bits (zone id, area, exterior coordinates).
    """

    serialized_zones = []
    for zone in result.get("danger_zones", []):
        polygon = zone.get("polygon")
        serialized_zones.append(
            {
                "zone_id": zone.get("zone_id"),
                "age": zone.get("age"),
                "missed": zone.get("missed"),
                "area": float(polygon.area) if polygon is not None else None,
                "coordinates": (
                    [list(coord) for coord in polygon.exterior.coords]
                    if polygon is not None
                    else []
                ),
            }
        )

    return {
        "risk_level": result.get("risk_level"),
        "violations": result.get("violations", []),
        "violation_count": result.get("violation_count", 0),
        "violation_counts": result.get("violation_counts", {}),
        "danger_zones": serialized_zones,
        "people_inside_zones": result.get("people_inside_zones", []),
        "distance_results": result.get("distance_results", []),
        "pole_results": result.get("pole_results", []),
        "fire_detections": result.get("fire_detections", []),
        "statistics": result.get("statistics", {}),
    }


# ============================================================
# IMAGE - FULL SAFETY PIPELINE
# ============================================================

@app.post("/safety/detect/image")
async def detect_safety_image(file: UploadFile = File(...)):

    contents = await file.read()
    image_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return {"success": False, "error": "Invalid image"}

    # A standalone image has no "next frame" to track into, and no
    # zone/fire history worth stabilizing - fresh engine, no tracking,
    # no fire confirmation (a single image gets fire_tracker=None, so
    # any fire detection above the base confidence in rules.py counts
    # immediately - there's no "next frame" to require persistence from).
    engine = SafetyEngine(config=DEFAULT_SAFETY_CONFIG, zone_tracker=None)

    try:
        result, persons, machines, annotated = run_pipeline_on_frame(
            image, engine, track=False, fire_tracker=None
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Image pipeline failed")
        return {"success": False, "error": str(exc)}

    output_id = str(uuid.uuid4())
    output_path = OUTPUT_VIDEO_DIR / f"{output_id}_annotated.jpg"
    cv2.imwrite(str(output_path), annotated)

    return {
        "success": True,
        "filename": file.filename,
        "result": _serialize_result(result),
        "annotated_image_path": str(output_path),
    }


# ============================================================
# FIRE / SMOKE IMAGE DETECTION (kept standalone for isolated testing)
# ============================================================

@app.post("/detect/fire/image")
async def detect_fire_image(file: UploadFile = File(...)):

    contents = await file.read()
    image_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return {"success": False, "error": "Invalid image"}

    detections = fire_detector.predict(image)

    return {
        "success": True,
        "model": "Fire/Smoke",
        "filename": file.filename,
        "detections": detections,
    }


# ============================================================
# VIDEO - FULL SAFETY PIPELINE
# ============================================================

@app.post("/safety/detect/video")
async def detect_safety_video(file: UploadFile = File(...)):

    video_id = str(uuid.uuid4())
    input_path = OUTPUT_VIDEO_DIR / f"{video_id}_input.mp4"
    output_path = OUTPUT_VIDEO_DIR / f"{video_id}_safety.mp4"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        input_path.unlink(missing_ok=True)
        return {"success": False, "error": "Could not open video"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # One engine + zone tracker + fire tracker + track-confirmation
    # tracker for the WHOLE video, so track_ids, zone_ids, and
    # confirmation streaks all stay coherent frame-to-frame within
    # this job.
    zone_tracker = DangerZoneTracker()
    fire_tracker = FireConfirmationTracker()
    track_confirmation = TrackConfirmationTracker()
    engine = SafetyEngine(config=DEFAULT_SAFETY_CONFIG, zone_tracker=zone_tracker)

    frame_index = 0
    failed_frames = 0
    max_risk_seen = "SAFE"
    risk_order = ["SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    violation_log = []

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1

        try:
            result, persons, machines, annotated = run_pipeline_on_frame(
                frame, engine, track=True,
                fire_tracker=fire_tracker,
                track_confirmation=track_confirmation,
            )
        except Exception as exc:  # noqa: BLE001
            # A single bad frame should not abort the whole video.
            logger.warning("Frame %d failed, writing raw frame instead: %s", frame_index, exc)
            failed_frames += 1
            writer.write(frame)
            continue

        writer.write(annotated)

        risk_level = result.get("risk_level", "SAFE")
        if risk_order.index(risk_level) > risk_order.index(max_risk_seen):
            max_risk_seen = risk_level

        if result.get("violation_count", 0) > 0:
            violation_log.append(
                {
                    "frame": frame_index,
                    "risk_level": risk_level,
                    "violation_count": result["violation_count"],
                    "violation_counts": result.get("violation_counts", {}),
                }
            )

    cap.release()
    writer.release()
    input_path.unlink(missing_ok=True)

    elapsed = time.time() - start_time
    logger.info(
        "Video %s: %d frames (%d failed) in %.1fs, max risk = %s",
        video_id, frame_index, failed_frames, elapsed, max_risk_seen,
    )

    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename="safety_analysis.mp4",
        headers={
            "X-Frames-Processed": str(frame_index),
            "X-Frames-Failed": str(failed_frames),
            "X-Max-Risk-Level": max_risk_seen,
        },
    )


# ============================================================
# LIVE CAMERA - WEBSOCKET, FULL SAFETY PIPELINE
# ============================================================

@app.websocket("/safety/ws/camera")
async def safety_camera_websocket(websocket: WebSocket):

    await websocket.accept()
    logger.info("Safety camera WebSocket connected")

    # Fresh engine + zone tracker + fire tracker + track-confirmation
    # tracker PER CONNECTION - do not share these across different
    # clients/streams.
    zone_tracker = DangerZoneTracker()
    fire_tracker = FireConfirmationTracker()
    track_confirmation = TrackConfirmationTracker()
    engine = SafetyEngine(config=DEFAULT_SAFETY_CONFIG, zone_tracker=zone_tracker)

    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 20  # guard against a persistently broken stream

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                logger.info("Camera WebSocket disconnect requested by client")
                break

            data = message.get("bytes")

            if not data and message.get("text"):
                try:
                    text_str = message["text"]
                    if "," in text_str:
                        text_str = text_str.split(",")[1]
                    data = base64.b64decode(text_str)
                except Exception:  # noqa: BLE001
                    data = None

            if not data:
                continue

            try:
                image_array = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                if frame is None:
                    raise ValueError("Could not decode incoming frame")

                result, persons, machines, annotated = run_pipeline_on_frame(
                    frame, engine, track=True,
                    fire_tracker=fire_tracker,
                    track_confirmation=track_confirmation,
                )

                success, encoded = cv2.imencode(
                    ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75]
                )
                if not success:
                    raise ValueError("Failed to encode annotated frame")

                base64_img = base64.b64encode(encoded.tobytes()).decode("utf-8")

                await websocket.send_json(
                    {
                        "image": f"data:image/jpeg;base64,{base64_img}",
                        **_serialize_result(result),
                    }
                )

                consecutive_failures = 0

            except Exception as exc:  # noqa: BLE001
                # Skip this frame, keep the connection alive. Only bail
                # out if failures are persistent (e.g. bad stream format)
                # rather than one-off (e.g. a single corrupted JPEG).
                consecutive_failures += 1
                logger.warning(
                    "Frame processing failed (%d consecutive): %s",
                    consecutive_failures, exc,
                )

                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.error("Too many consecutive frame failures, closing connection")
                    await websocket.send_json(
                        {"error": "Too many consecutive frame failures, closing connection"}
                    )
                    break

                continue

    except WebSocketDisconnect:
        logger.info("Camera disconnected")

    except Exception as exc:  # noqa: BLE001
        logger.exception("WebSocket error: %s", exc)

    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


# ============================================================
# MOUNT FRONTEND STATIC FILES AT ROOT /
# ============================================================

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")