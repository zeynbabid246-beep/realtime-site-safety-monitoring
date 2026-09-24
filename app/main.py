"""
Construction Safety AI - FastAPI application.

ARCHITECTURE
------------
All per-frame processing goes through ONE shared pipeline
(src/pipeline.py::SafetyPipeline) - the REST image endpoint, the REST video
endpoint, and the live websocket camera all run the identical code path, so
"video" and "webcam" cannot drift apart. The heavy YOLO models
(HazardDetector, FireDetector) are loaded ONCE at startup and injected into
every per-stream pipeline; they serialise their own forward passes behind an
internal lock, so concurrent video jobs and camera clients never race on the
shared model object.

OBSERVABILITY / "FINAL" FEATURES
--------------------------------
Each stream also gets a SafetyMonitor (app/monitor.py) that turns processed
frames into durable artefacts:
  - History  : risk-bearing frames stored as events in SQLite (cooldown-bounded)
  - Evidence : annotated JPEG snapshot + short MP4 clip on HIGH/CRITICAL
  - Alerts   : HIGH/CRITICAL events recorded to SQLite (dashboard feed) and
               pushed to Telegram (optional, off the event loop)
  - Statistics / Reports : aggregates over the event/alert history
Persistence is a single thread-safe SQLite DB (app/storage.py); evidence files
are served read-only from a StaticFiles mount at /evidence.

CONCURRENCY NOTE
----------------
ByteTrack state lives inside the shared HazardDetector model object. With a
SINGLE active tracked stream (the normal case) this is correct. If you run
MULTIPLE simultaneous tracked streams against one shared model, their trackers
interleave; for strict per-stream isolation, construct a dedicated
HazardDetector per stream and inject it into that stream's SafetyPipeline
(see docs/DETECTION_LIMITATIONS.md).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from src.hazard.hazard_detector import HazardDetector
from src.fire.fire_detector import FireDetector
from src.safety.rules import SafetyConfig
from src.pipeline import (
    SafetyPipeline,
    serialize_result,
    summarize_result,
    detections_for_ui,
    identities_for_ui,
    summarize_identities,
)
from src.face_recognition.config import FaceRecognitionConfig
from src.face_recognition.registry import WorkerRegistry
from src.face_recognition.service import FaceRecognitionService
from src.face_recognition import calibration as face_calibration

from app.settings import SETTINGS, RISK_ORDER
from app.storage import init_db, get_db
from app.alerting import AlertManager
from app.monitor import SafetyMonitor
from app.evidence import prune_evidence
from app.reports import build_report, events_to_csv, RANGES


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

# Keep only the newest N generated artifacts on disk (bounded growth).
MAX_OUTPUT_FILES = 100
_OUTPUT_PATTERNS = ("*_input.mp4", "*_safety.mp4", "*_annotated.jpg")

# App-level singletons, created in the lifespan startup hook.
ALERT_MANAGER: Optional[AlertManager] = None


# ============================================================
# LIFESPAN (startup / shutdown)
# ============================================================

@asynccontextmanager
async def lifespan(_: FastAPI):
    global ALERT_MANAGER

    SETTINGS.ensure_dirs()
    init_db(SETTINGS.db_path)
    ALERT_MANAGER = AlertManager.from_settings(SETTINGS, get_db())

    db = get_db()
    db.prune_events(SETTINGS.max_events)
    prune_evidence(SETTINGS.evidence_dir, SETTINGS.max_evidence_files)

    logger.info(
        "Observability ready: db=%s evidence=%s telegram=%s",
        SETTINGS.db_path, SETTINGS.evidence_dir,
        "on" if SETTINGS.telegram_configured else "off",
    )
    yield
    try:
        get_db().close()
    except Exception:  # noqa: BLE001
        pass


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Construction Safety AI",
    description="AI-powered construction site safety detection system",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Frames-Processed", "X-Frames-Failed", "X-Max-Risk-Level"],
)


# ============================================================
# LOAD MODELS (once, at startup)
# ============================================================

logger.info("Loading hazard model...")
hazard_detector = HazardDetector(HAZARD_MODEL_PATH, confidence=0.25, image_size=640)
logger.info("Hazard classes: %s", hazard_detector.class_names)

logger.info("Loading fire/smoke model...")
fire_detector = FireDetector(FIRE_MODEL_PATH, confidence=0.35, image_size=416)
logger.info("Fire/smoke classes: %s", fire_detector.class_names)

DEFAULT_SAFETY_CONFIG = SafetyConfig()

logger.info("Models loaded successfully.")


# ============================================================
# FACE RECOGNITION / WORKER VERIFICATION (optional)
# ============================================================
# Loads the trained Siamese model ONCE and shares it across every
# stream (REST image/video, websocket camera, CLI scripts receive it
# via dependency injection). Any failure (missing TF, missing .h5,
# unreadable registry) degrades gracefully: FACE_SERVICE becomes None
# and the rest of the app runs exactly as before.

def _build_face_service() -> Optional[FaceRecognitionService]:
    """Build the shared face-recognition service, or None on failure."""

    if not SETTINGS.face_recognition_enabled:
        logger.info("Face recognition disabled (FACE_RECOGNITION_ENABLED=false).")
        return None

    try:
        face_config = FaceRecognitionConfig()

        # Calibrated thresholds (scripts/calibrate_threshold.py) win over
        # the static env defaults when present.
        face_config.apply_calibration(face_calibration.load_calibration(face_config))

        face_config.ensure_dirs()
        registry = WorkerRegistry(face_config)
        # Pick up folders the user dropped into input_images/ by hand.
        registry.auto_discover_workers()

        from src.face_recognition.backends import create_backend

        backend = create_backend(face_config)

        service = FaceRecognitionService(face_config, backend=backend, registry=registry)
        logger.info(
            "Face recognition ready: backend=%s model=%s workers=%d "
            "thresholds=(det=%.2f, ver=%.2f)",
            SETTINGS.face_backend,
            face_config.model_path,
            len(registry.worker_ids()),
            face_config.detection_threshold,
            face_config.verification_threshold,
        )
        return service
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Face recognition NOT available (%s: %s) - continuing without "
            "worker identification.", exc.__class__.__name__, exc,
        )
        return None


FACE_SERVICE: Optional[FaceRecognitionService] = _build_face_service()


# ============================================================
# HELPERS
# ============================================================

@app.get("/health")
def health():
    return {"status": "healthy", "version": app.version}


def _prune_old_outputs(directory: Path = OUTPUT_VIDEO_DIR, keep: int = MAX_OUTPUT_FILES) -> None:
    """
    Delete the oldest generated artifacts so the output directory does not
    grow without bound. Only touches files matching our own output
    patterns - never user inputs or unrelated directories.
    """

    files: List[Path] = []
    for pattern in _OUTPUT_PATTERNS:
        files.extend(p for p in directory.glob(pattern) if p.is_file())

    if len(files) <= keep:
        return

    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in files[keep:]:
        try:
            stale.unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("Could not prune %s: %s", stale, exc)


def _build_payload(
    result: Dict[str, Any],
    detections: List[Dict[str, Any]],
    identities: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """The stable JSON contract shared by the image and websocket endpoints."""

    return {
        **serialize_result(result),
        "summary": summarize_result(result),
        "detections": detections_for_ui(detections),
        "identities": identities_for_ui(identities or []),
        "identity_summary": summarize_identities(identities or []),
    }


def _new_monitor(source: str) -> Optional[SafetyMonitor]:
    """Build a per-stream monitor; None if observability isn't initialised."""
    if ALERT_MANAGER is None:
        return None
    return SafetyMonitor(source, SETTINGS, get_db(), ALERT_MANAGER)


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

    # A standalone image has no "next frame": fresh pipeline, no tracking,
    # no temporal confirmation gates.
    pipeline = SafetyPipeline(
        hazard_detector,
        fire_detector,
        config=DEFAULT_SAFETY_CONFIG,
        enable_track_confirmation=False,
        enable_fire_confirmation=False,
        face_service=FACE_SERVICE,
    )

    try:
        frame_result = await run_in_threadpool(pipeline.process_frame, image, False, True)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Image pipeline failed")
        return {"success": False, "error": str(exc)}

    _prune_old_outputs()
    output_id = str(uuid.uuid4())
    output_path = OUTPUT_VIDEO_DIR / f"{output_id}_annotated.jpg"
    cv2.imwrite(str(output_path), frame_result.annotated)

    monitor = _new_monitor("image")
    if monitor is not None:
        try:
            await run_in_threadpool(monitor.handle, frame_result.result, frame_result.annotated)
        finally:
            await run_in_threadpool(monitor.close)

    return {
        "success": True,
        "filename": file.filename,
        "result": _build_payload(frame_result.result, frame_result.detections, frame_result.identities),
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

    detections = await run_in_threadpool(fire_detector.predict, image)

    return {
        "success": True,
        "model": "Fire/Smoke",
        "filename": file.filename,
        "detections": detections,
    }


# ============================================================
# VIDEO - FULL SAFETY PIPELINE
# ============================================================

def _process_video_file(
    input_path: Path,
    output_path: Path,
    pipeline: SafetyPipeline,
    monitor: Optional[SafetyMonitor] = None,
) -> Dict[str, Any]:
    """
    Synchronous, blocking whole-video pass. Runs in a worker thread so the
    event loop stays responsive. One bad frame is skipped, never fatal.
    """

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return {"opened": False}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use H.264 (avc1) so HTML5 browsers can natively play the video without black screen
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_index = 0
    failed_frames = 0
    max_risk_seen = "SAFE"
    # Bounded aggregate (replaces the old unbounded per-frame violation_log).
    violation_totals: Dict[str, int] = {}

    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1

            try:
                frame_result = pipeline.process_frame(frame, track=True, draw=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Frame %d failed, writing raw frame instead: %s", frame_index, exc)
                failed_frames += 1
                writer.write(frame)
                continue

            writer.write(frame_result.annotated)

            result = frame_result.result
            risk_level = result.get("risk_level", "SAFE")
            if RISK_ORDER.index(risk_level) > RISK_ORDER.index(max_risk_seen):
                max_risk_seen = risk_level

            for violation_type, count in (result.get("violation_counts", {}) or {}).items():
                violation_totals[violation_type] = violation_totals.get(violation_type, 0) + count

            # Durability: history / evidence / alerts (off the event loop already).
            if monitor is not None:
                try:
                    monitor.handle(result, frame_result.annotated)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Monitor failed on frame %d: %s", frame_index, exc.__class__.__name__)
    finally:
        cap.release()
        writer.release()
        if monitor is not None:
            try:
                monitor.close()
            except Exception:  # noqa: BLE001
                pass

    return {
        "opened": True,
        "frames": frame_index,
        "failed_frames": failed_frames,
        "max_risk": max_risk_seen,
        "violation_totals": violation_totals,
        "elapsed": time.time() - start_time,
    }


@app.post("/safety/detect/video")
async def detect_safety_video(file: UploadFile = File(...)):

    video_id = str(uuid.uuid4())
    input_path = OUTPUT_VIDEO_DIR / f"{video_id}_input.mp4"
    output_path = OUTPUT_VIDEO_DIR / f"{video_id}_safety.mp4"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # One pipeline for the WHOLE video: track ids, zone ids, and
    # confirmation streaks stay coherent frame-to-frame within this job.
    pipeline = SafetyPipeline(
        hazard_detector,
        fire_detector,
        config=DEFAULT_SAFETY_CONFIG,
        face_service=FACE_SERVICE,
    )
    monitor = _new_monitor("video")

    summary = await run_in_threadpool(_process_video_file, input_path, output_path, pipeline, monitor)
    input_path.unlink(missing_ok=True)

    if not summary.get("opened"):
        output_path.unlink(missing_ok=True)
        return {"success": False, "error": "Could not open video"}

    _prune_old_outputs()
    prune_evidence(SETTINGS.evidence_dir, SETTINGS.max_evidence_files)

    logger.info(
        "Video %s: %d frames (%d failed) in %.1fs, max risk = %s",
        video_id, summary["frames"], summary["failed_frames"],
        summary["elapsed"], summary["max_risk"],
    )

    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename="safety_analysis.mp4",
        headers={
            "X-Frames-Processed": str(summary["frames"]),
            "X-Frames-Failed": str(summary["failed_frames"]),
            "X-Max-Risk-Level": summary["max_risk"],
        },
    )


# ============================================================
# LIVE CAMERA - WEBSOCKET, FULL SAFETY PIPELINE
# ============================================================

def _decode_frame(data: bytes) -> Optional[np.ndarray]:
    image_array = np.frombuffer(data, np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


@app.websocket("/safety/ws/camera")
async def safety_camera_websocket(websocket: WebSocket):

    await websocket.accept()
    logger.info("Safety camera WebSocket connected")

    # Fresh pipeline PER CONNECTION - never shared across clients/streams.
    pipeline = SafetyPipeline(
        hazard_detector,
        fire_detector,
        config=DEFAULT_SAFETY_CONFIG,
        face_service=FACE_SERVICE,
        face_frame_stride=SETTINGS.face_frame_stride,
    )
    monitor = _new_monitor("camera")

    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 20

    async def _drain_to_latest() -> Optional[bytes]:
        """Consume all queued client frames, keeping only the newest one.

        Prevents stale frames from piling up when inference is slower than
        the client's send rate - the frontend's backpressure already limits
        this, but this is a safety net for burst traffic.
        """
        latest: Optional[bytes] = None
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=0.005)
            except asyncio.TimeoutError:
                break
            if msg.get("type") == "websocket.disconnect":
                return None
            data = msg.get("bytes")
            if not data and msg.get("text"):
                try:
                    text_str = msg["text"]
                    if "," in text_str:
                        text_str = text_str.split(",")[1]
                    data = base64.b64decode(text_str)
                except Exception:  # noqa: BLE001
                    data = None
            if data:
                latest = data
        return latest

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

            # Drain any frames queued while we were processing - only the
            # freshest frame matters for a live feed.
            newer = await _drain_to_latest()
            if newer is not None:
                data = newer

            try:
                frame = await run_in_threadpool(_decode_frame, data)
                if frame is None:
                    raise ValueError("Could not decode incoming frame")

                frame_result = await run_in_threadpool(
                    pipeline.process_frame, frame, True, True
                )

                success, encoded = cv2.imencode(
                    ".jpg", frame_result.annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 60]
                )
                if not success:
                    raise ValueError("Failed to encode annotated frame")

                # Binary frame - 33% smaller than base64, no JSON parsing overhead.
                await websocket.send_bytes(encoded.tobytes())

                # JSON metadata as a separate text message. The frontend uses
                # this as the "done" signal to send the next frame.
                await websocket.send_json({
                    "frame_done": True,
                    **_build_payload(frame_result.result, frame_result.detections, frame_result.identities),
                })

                # Monitor runs in background - don't block the next frame.
                if monitor is not None:
                    asyncio.create_task(
                        run_in_threadpool(monitor.handle, frame_result.result, frame_result.annotated)
                    )

                consecutive_failures = 0

            except Exception as exc:  # noqa: BLE001
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
        if monitor is not None:
            try:
                await run_in_threadpool(monitor.close)
            except Exception:  # noqa: BLE001
                pass
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


# ============================================================
# FACE / WORKER REGISTRY API
# Register & manage worker identities and reference faces.
# All endpoints degrade to 503 when face recognition is unavailable.
# ============================================================

def _face_service_or_error():
    if FACE_SERVICE is None:
        return None, JSONResponse(
            status_code=503,
            content={"success": False, "error": "Face recognition unavailable"
                                              " (TensorFlow missing or model not found)"},
        )
    return FACE_SERVICE, None


@app.get("/face/workers")
def face_list_workers():
    """All registered workers (including deactivated ones, flagged)."""

    service, error = _face_service_or_error()
    if error:
        return error
    workers = [w.to_dict() for w in service.registry.list_workers(active_only=False)]
    return {"success": True, "workers": workers, "count": len(workers)}


@app.post("/face/workers")
async def face_register_worker(
    worker_id: str,
    name: str = "",
    role: str = "",
    files: Optional[List[UploadFile]] = File(None),
):
    """
    Register a worker. Optionally attach reference face images
    (multipart files[] - the SAME 250x250-style face crops the training
    webcam flow produced). At least one reference image is required
    before the worker can be verified.
    """

    service, error = _face_service_or_error()
    if error:
        return error

    added = 0
    for upload in files or []:
        contents = await upload.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Registration upload not a valid image: %s", upload.filename)
            continue
        if service.registry.add_reference_image(worker_id, image, filename=upload.filename):
            added += 1

    record = service.registry.get_worker(worker_id)
    if record is None:
        return {"success": False, "error": "Worker could not be registered"}

    return {
        "success": True,
        "worker": record.to_dict(),
        "images_added": added,
        "n_references": len(record.reference_images),
    }


@app.post("/face/workers/{worker_id}/images")
async def face_add_worker_images(worker_id: str, files: List[UploadFile] = File(...)):
    """Add more reference faces to an existing worker (improves accuracy)."""

    service, error = _face_service_or_error()
    if error:
        return error
    if service.registry.get_worker(worker_id) is None:
        return {"success": False, "error": f"Unknown worker: {worker_id}"}

    added = 0
    for upload in files:
        contents = await upload.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image is not None and service.registry.add_reference_image(
            worker_id, image, filename=upload.filename
        ):
            added += 1

    record = service.registry.get_worker(worker_id)
    return {"success": True, "images_added": added, "worker": record.to_dict()}


@app.delete("/face/workers/{worker_id}")
def face_remove_worker(worker_id: str, delete_images: bool = False):
    """Deactivate a worker (or purge with delete_images=true)."""

    service, error = _face_service_or_error()
    if error:
        return error
    removed = service.remove_worker(worker_id, delete_images=delete_images)
    return {"success": removed, "worker_id": worker_id, "purged": bool(delete_images)}


@app.post("/face/verify")
async def face_verify_image(file: UploadFile = File(...)):
    """
    Standalone verification endpoint: upload an image containing face(s),
    get the recognized worker(s) back. Uses the hazard model to find
    persons, then the Siamese verifier on each face crop.
    """

    service, error = _face_service_or_error()
    if error:
        return error

    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return {"success": False, "error": "Invalid image"}

    try:
        detections = await run_in_threadpool(hazard_detector.predict, image)
        persons = hazard_detector.extract_persons(detections)
        identities = await run_in_threadpool(service.identify_faces, image, persons)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Face verify endpoint failed")
        return {"success": False, "error": str(exc)}

    return {
        "success": True,
        "filename": file.filename,
        "faces": [i.to_dict() for i in identities],
        "verified_workers": [
            {"worker_id": i.worker_id, "worker_name": i.worker_name, "score": round(i.score, 4)}
            for i in identities if i.verified
        ],
    }


@app.post("/face/calibrate")
def face_calibrate():
    """
    Recalibrate verification thresholds from the current registered
    workers and persist them to calibration.json (picked up on the next
    restart, or immediately for CLI runs).
    """

    service, error = _face_service_or_error()
    if error:
        return error

    try:
        result = face_calibration.calibrate_threshold(
            service.backend,
            service.registry,
            input_size=service.config.input_size,
        )
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    face_calibration.save_calibration(service.config, result)
    service.config.apply_calibration(result)
    return {"success": True, "calibration": result}


# ============================================================
# OBSERVABILITY API (Dashboard / Alerts / History / Statistics /
#                    Evidence / Reports)
# ============================================================

@app.get("/api/events")
def api_events(
    limit: int = 100,
    risk: Optional[str] = None,
    source: Optional[str] = None,
    since: Optional[float] = None,
):
    """History feed (newest first)."""
    events = get_db().list_events(limit=min(limit, 1000), risk=risk, source=source, since=since)
    return {"events": events, "count": len(events)}


@app.get("/api/alerts")
def api_alerts(
    limit: int = 100,
    status: Optional[str] = None,
    level: Optional[str] = None,
    since: Optional[float] = None,
):
    """Alert feed + unacknowledged count (for the header badge)."""
    db = get_db()
    alerts = db.list_alerts(limit=min(limit, 1000), status=status, level=level, since=since)
    return {
        "alerts": alerts,
        "count": len(alerts),
        "unacknowledged": db.count_alerts("new"),
    }


@app.post("/api/alerts/{alert_id}/ack")
def api_ack_alert(alert_id: int):
    ok = get_db().ack_alert(alert_id)
    return {"success": ok, "id": alert_id}


@app.get("/api/statistics")
def api_statistics(range: str = "24h"):
    """Aggregate statistics for the Statistics tab."""
    return build_report(get_db(), range if range in RANGES else "24h")


@app.get("/api/evidence")
def api_evidence(limit: int = 60):
    """Evidence gallery: events that have a snapshot and/or clip."""
    events = get_db().list_events(limit=500)
    items = [
        {
            "event_id": e.get("id"),
            "ts": e.get("ts"),
            "source": e.get("source"),
            "risk_level": e.get("risk_level"),
            "violation_counts": e.get("violation_counts", {}),
            "image": e.get("evidence_image"),
            "clip": e.get("evidence_clip"),
        }
        for e in events
        if e.get("evidence_image") or e.get("evidence_clip")
    ][: min(limit, 500)]
    return {"evidence": items, "count": len(items)}


@app.get("/api/reports")
def api_report(range: str = "24h"):
    """Full period report (same aggregate as statistics, kept as its own route)."""
    return build_report(get_db(), range if range in RANGES else "24h")


@app.get("/api/reports/download")
def api_report_download(range: str = "24h", format: str = "json"):
    """Download a report as JSON (aggregate) or CSV (raw events in range)."""
    db = get_db()
    range = range if range in RANGES else "24h"
    if format.lower() == "csv":
        csv_text = events_to_csv(db, range)
        return Response(
            content=csv_text,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="safety_report_{range}.csv"'},
        )
    report = build_report(db, range)
    return JSONResponse(
        content=report,
        headers={"Content-Disposition": f'attachment; filename="safety_report_{range}.json"'},
    )


# ============================================================
# STATIC MOUNTS (must come last - "/" is a catch-all)
# ============================================================

# ensure_dirs() also runs in lifespan, but mounts are registered at import
# time (before startup), so create the evidence dir here to guarantee the
# /evidence mount exists on a fresh checkout.
SETTINGS.ensure_dirs()

if SETTINGS.evidence_dir.exists():
    app.mount("/evidence", StaticFiles(directory=str(SETTINGS.evidence_dir)), name="evidence")

OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/output", StaticFiles(directory=str(OUTPUT_VIDEO_DIR)), name="output")

# Frontend: prefer the built TanStack Start bundle (frontend/.output/public);
# fall back to legacy static files in frontend/ root for older checkouts.
# The SPA build (`bun run build:spa` in frontend/) emits a client-only shell as
# _shell.html (or index.html on some Nitro versions) — use whichever exists.
_BUILT_FRONTEND = FRONTEND_DIR / ".output" / "public"

if _BUILT_FRONTEND.exists():
    _SPA_SHELL = next(
        (
            _BUILT_FRONTEND / _name
            for _name in ("index.html", "_shell.html")
            if (_BUILT_FRONTEND / _name).is_file()
        ),
        _BUILT_FRONTEND / "_shell.html",
    )
    app.mount("/assets", StaticFiles(directory=str(_BUILT_FRONTEND / "assets")), name="frontend-assets")

    @app.get("/", include_in_schema=False)
    async def _spa_index() -> FileResponse:
        return FileResponse(_SPA_SHELL)

    @app.get("/{spa_path:path}", include_in_schema=False)
    async def _spa_fallback(spa_path: str) -> FileResponse:
        """SPA history fallback: serve known static files, otherwise the SPA
        shell so client-side routes (/monitor, /workers, ...) work on hard refresh."""

        candidate = (_BUILT_FRONTEND / spa_path).resolve()
        try:
            candidate.relative_to(_BUILT_FRONTEND.resolve())
        except ValueError:  # path traversal attempt
            candidate = _SPA_SHELL
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_SPA_SHELL)
elif FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")
