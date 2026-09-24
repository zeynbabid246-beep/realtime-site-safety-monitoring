"""
Shared safety pipeline - the SINGLE source of truth for how one frame is
turned into detections, violations, a risk level, and an annotated image.

WHY THIS MODULE EXISTS
----------------------
The exact same per-frame pipeline used to be copy-pasted in three places
(app/main.py's run_pipeline_on_frame, scripts/analyze_video.py's inline
loop, and scripts/live_webcam.py's inline loop). They had already started
to drift - the webcam script collected no fire-gate stats, the video
script had its own bespoke stats bookkeeping, and a fix applied to one
copy did not automatically reach the others. SafetyPipeline removes that
duplication: every entry point (REST image, REST video, websocket camera,
offline video CLI, live webcam CLI) now runs the SAME code path, so the
system genuinely behaves identically "for both videos and webcam".

PIPELINE ORDER (matters)
------------------------
    frame
      -> HazardDetector.track()/predict()          (one YOLO forward pass)
      -> filter_detections (conf / size / aspect / ROI)   [stateless]
      -> TrackConfirmationTracker (min-hits debounce + static-object
         suppressors)                                      [stateful]
      -> FireDetector.predict()                     (second YOLO forward pass)
      -> FireConfirmationTracker (fire & smoke debounced independently)
      -> extract_persons / extract_machines
      -> SafetyEngine.analyze (zones, distance, PPE, proximity, fire, risk)
      -> draw_safety_overlay

Raw YOLO output is filtered (confidence/size/aspect) BEFORE track
confirmation, so an obviously-too-small/low-confidence detection never
even starts accumulating a confirmation streak.

STATE / CONCURRENCY
-------------------
A SafetyPipeline instance owns the STATEFUL per-stream components
(zone tracker, track-confirmation tracker, fire-confirmation tracker, and
the SafetyEngine that holds the zone tracker). Create ONE instance per
video / camera stream / image job and never share it across unrelated
streams, or track ids, zone ids, and confirmation streaks will
cross-contaminate.

The DETECTORS (HazardDetector / FireDetector) are the opposite: they hold
the heavy model weights, are safe to share, and serialise their own
forward passes behind an internal lock. A pipeline takes them by
injection so the whole app loads each model exactly once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.hazard.hazard_detector import HazardDetector
from src.hazard.detection_filter import filter_detections_with_stats
from src.hazard.track_confirmation import TrackConfirmationTracker
from src.fire.fire_detector import FireDetector
from src.fire.fire_confirmation import FireConfirmationTracker, detection_kind
from src.safety.safety_engine import SafetyEngine
from src.safety.geometry import DangerZoneTracker
from src.safety.rules import SafetyConfig
from src.safety.overlay import draw_safety_overlay, draw_unconfirmed_fire_debug
from src.face_recognition.service import FaceRecognitionService
from src.face_recognition.overlay import (
    draw_face_recognition_banner,
    draw_identity_annotations,
)

logger = logging.getLogger(__name__)


# ============================================================
# FRAME RESULT
# ============================================================

@dataclass
class FrameResult:
    """Everything one processed frame produces, so callers never re-derive it."""

    result: Dict[str, Any]                       # SafetyEngine.analyze() output
    persons: List[Dict[str, Any]]
    machines: List[Dict[str, Any]]
    detections: List[Dict[str, Any]]             # confirmed hazard detections (drawn)
    raw_detections: List[Dict[str, Any]]         # pre-filter hazard detections
    fire_detections: List[Dict[str, Any]]        # confirmed fire/smoke
    raw_fire_detections: List[Dict[str, Any]]    # pre-confirmation fire/smoke
    identities: List[Any] = field(default_factory=list)  # FrameIdentity per verified face
    annotated: Optional[np.ndarray] = None
    filter_stats: Dict[str, int] = field(default_factory=dict)
    fire_gate_stats: Dict[str, Any] = field(default_factory=dict)


def _basic_fire_stats(
    raw_fire: List[Dict[str, Any]],
    confirmed_fire: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Fire-gate stats for when the confirmation gate is disabled."""

    def _area(det: Dict[str, Any]) -> float:
        box = det.get("bbox") or [0, 0, 0, 0]
        return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))

    raw_smoke = [d for d in raw_fire if detection_kind(d) == "smoke"]
    raw_fire_only = [d for d in raw_fire if detection_kind(d) == "fire"]

    return {
        "raw": len(raw_fire),
        "raw_fire": len(raw_fire_only),
        "raw_smoke": len(raw_smoke),
        "dropped_confidence": 0,
        "dropped_area": 0,
        "dropped_persistence": 0,
        "confirmed": len(confirmed_fire),
        "confirmed_fire": sum(1 for d in confirmed_fire if detection_kind(d) == "fire"),
        "confirmed_smoke": sum(1 for d in confirmed_fire if detection_kind(d) == "smoke"),
        "max_raw_smoke_conf": max((d.get("confidence", 0.0) for d in raw_smoke), default=0.0),
        "max_raw_smoke_area": max((_area(d) for d in raw_smoke), default=0.0),
        "max_raw_fire_conf": max((d.get("confidence", 0.0) for d in raw_fire_only), default=0.0),
        "max_raw_fire_area": max((_area(d) for d in raw_fire_only), default=0.0),
    }


# ============================================================
# PIPELINE
# ============================================================

class SafetyPipeline:
    """Runs one frame through the full, unified safety pipeline."""

    def __init__(
        self,
        hazard_detector: HazardDetector,
        fire_detector: Optional[FireDetector] = None,
        config: Optional[SafetyConfig] = None,
        enable_detection_filter: bool = True,
        enable_track_confirmation: bool = True,
        enable_fire_confirmation: bool = True,
        roi_polygon: Any = None,
        track_confirm_min_hits: int = 3,
        track_confirm_max_missed: int = 5,
        draw_raw_fire: bool = False,
        zone_tracker: Optional[DangerZoneTracker] = None,
        track_confirmation: Optional[TrackConfirmationTracker] = None,
        fire_tracker: Optional[FireConfirmationTracker] = None,
        face_service: Optional[FaceRecognitionService] = None,
        face_frame_stride: int = 1,
    ):
        self.hazard_detector = hazard_detector
        self.fire_detector = fire_detector
        self.face_service = face_service  # optional worker-identity layer
        self.face_frame_stride = max(1, int(face_frame_stride))
        self._face_frame_counter = 0
        self.config = config if config is not None else SafetyConfig()

        self.enable_detection_filter = enable_detection_filter
        self.enable_track_confirmation = enable_track_confirmation and hazard_detector is not None
        # Fire confirmation only makes sense with a fire model and across frames.
        self.enable_fire_confirmation = enable_fire_confirmation and fire_detector is not None
        self.roi_polygon = roi_polygon
        self.draw_raw_fire = draw_raw_fire

        self._track_confirm_min_hits = track_confirm_min_hits
        self._track_confirm_max_missed = track_confirm_max_missed

        # Optional injected, fully-configured trackers (used by the CLI
        # scripts, which expose per-threshold flags). When None, reset()
        # builds sensible defaults.
        self._injected_zone_tracker = zone_tracker
        self._injected_track_confirmation = track_confirmation
        self._injected_fire_tracker = fire_tracker

        self.zone_tracker: Optional[DangerZoneTracker] = None
        self.track_confirmation: Optional[TrackConfirmationTracker] = None
        self.fire_tracker: Optional[FireConfirmationTracker] = None
        self.engine: Optional[SafetyEngine] = None

        self.reset()

    # ----------------------------------------------------------
    # STATE
    # ----------------------------------------------------------

    def reset(self) -> None:
        """Clear all per-stream temporal state (call between independent streams)."""

        self.zone_tracker = self._injected_zone_tracker or DangerZoneTracker()
        self.engine = SafetyEngine(config=self.config, zone_tracker=self.zone_tracker)

        if self._injected_track_confirmation is not None:
            self.track_confirmation = self._injected_track_confirmation
        elif self.enable_track_confirmation:
            self.track_confirmation = TrackConfirmationTracker(
                min_hits=self._track_confirm_min_hits,
                max_missed_frames=self._track_confirm_max_missed,
            )
        else:
            self.track_confirmation = None

        if self._injected_fire_tracker is not None:
            self.fire_tracker = self._injected_fire_tracker
        elif self.enable_fire_confirmation:
            self.fire_tracker = FireConfirmationTracker()
        else:
            self.fire_tracker = None

    # ----------------------------------------------------------
    # MAIN ENTRY POINT
    # ----------------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray,
        track: bool = True,
        draw: bool = True,
    ) -> FrameResult:
        """
        Run one frame through the full pipeline.

        track=True uses ByteTrack (persist=True) for video/camera streams;
        track=False runs plain detection for a single standalone image
        (there is no "next frame" to track into).
        """

        frame_shape = frame.shape[:2]

        # 1. Hazard model (one forward pass) + stateless filter.
        if track:
            raw_detections = self.hazard_detector.track(frame, persist=True)
        else:
            raw_detections = self.hazard_detector.predict(frame)

        if self.enable_detection_filter:
            detections, filter_stats = filter_detections_with_stats(
                raw_detections,
                roi_polygon=self.roi_polygon,
                frame_shape=frame_shape,
            )
        else:
            detections, filter_stats = list(raw_detections), {}

        # 2. Temporal track confirmation (debounce + static-object suppressors).
        if self.track_confirmation is not None:
            detections = self.track_confirmation.update(detections, frame_shape=frame_shape)

        # 3. Fire/smoke model + independent confirmation gate.
        if self.fire_detector is not None:
            raw_fire = self.fire_detector.predict(frame)
        else:
            raw_fire = []

        if self.fire_tracker is not None:
            fire_detections = self.fire_tracker.update(raw_fire)
            fire_gate_stats = dict(self.fire_tracker.last_stats)
        else:
            fire_detections = raw_fire
            fire_gate_stats = _basic_fire_stats(raw_fire, fire_detections)

        # 4. Safety engine.
        persons = self.hazard_detector.extract_persons(detections)
        machines = self.hazard_detector.extract_machines(detections)

        result = self.engine.analyze(
            detections=detections,
            persons=persons,
            machines=machines,
            fire_detections=fire_detections,
            frame_width=float(frame.shape[1]),
        )

        # 5. Face recognition (optional): verify every detectable face
        #    against the registered workers. Strides by frame to bound cost
        #    on slow machines; cached identities keep per-track labels
        #    stable on the skipped frames.
        identities: List[Any] = []
        face_due = (
            self.face_service is not None
            and self._face_frame_counter % self.face_frame_stride == 0
        )
        self._face_frame_counter += 1
        if face_due:
            try:
                identities = self.face_service.identify_faces(frame, persons)
            except Exception as exc:  # noqa: BLE001
                # Identity must never take the safety pipeline down.
                logger.warning("Face recognition failed on frame: %s", exc)
                identities = []
        elif self.face_service is not None:
            identities = self.face_service.last_identities(frame, persons)

        # 6. Overlay.
        annotated = None
        if draw:
            annotated = draw_safety_overlay(
                frame, result, detections, self.hazard_detector.class_names
            )
            if self.draw_raw_fire:
                draw_unconfirmed_fire_debug(annotated, raw_fire, fire_detections)
            if identities:
                annotated = draw_identity_annotations(annotated, identities)
                annotated = draw_face_recognition_banner(
                    annotated,
                    identities,
                    recognition_enabled=True,
                    workers_registered=self.face_service.stats()["workers_registered"],
                )

        return FrameResult(
            result=result,
            persons=persons,
            machines=machines,
            detections=detections,
            raw_detections=raw_detections,
            fire_detections=fire_detections,
            raw_fire_detections=raw_fire,
            identities=identities,
            annotated=annotated,
            filter_stats=filter_stats,
            fire_gate_stats=fire_gate_stats,
        )


# ============================================================
# SERIALIZATION / API CONTRACT HELPERS
# ============================================================
# Shared by the REST and websocket endpoints so the frontend gets ONE
# stable contract instead of each endpoint inventing its own shape.

_PPE_VIOLATION_TYPES = ("NO_HARDHAT", "NO_SAFETY_VEST", "NO_MASK")


def serialize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strip non-JSON-serializable objects (shapely Polygons) out of a
    SafetyEngine result, keeping the useful bits (zone id, area, exterior
    coordinates).
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


def summarize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compact per-frame summary the dashboard panels consume:
    PPE / fire / smoke / person counts plus the headline risk level.
    """

    violation_counts = result.get("violation_counts", {}) or {}
    statistics = result.get("statistics", {}) or {}

    ppe_violations = sum(
        int(violation_counts.get(kind, 0)) for kind in _PPE_VIOLATION_TYPES
    )

    fire_count = 0
    smoke_count = 0
    for detection in result.get("fire_detections", []) or []:
        kind = detection_kind(detection)
        if kind == "fire":
            fire_count += 1
        elif kind == "smoke":
            smoke_count += 1

    return {
        "risk_level": result.get("risk_level", "SAFE"),
        "violation_count": result.get("violation_count", 0),
        "violation_counts": dict(violation_counts),
        "ppe_violations": ppe_violations,
        "fire_count": fire_count,
        "smoke_count": smoke_count,
        "person_count": int(statistics.get("persons", 0)),
        "machine_count": int(statistics.get("machines", 0)),
        "danger_zones": int(statistics.get("danger_zones", 0)),
    }


def detections_for_ui(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flat [{class, confidence}] list for the dashboard detection panel."""

    return [
        {
            "class": d.get("class_name", str(d.get("class_id", "unknown"))),
            "confidence": float(d.get("confidence", 0.0)),
        }
        for d in detections
        if d.get("bbox") is not None
    ]


def identities_for_ui(identities: List[Any]) -> List[Dict[str, Any]]:
    """JSON-safe identity list for the REST / websocket payloads."""

    return [identity.to_dict() for identity in identities or []]


def summarize_identities(identities: List[Any]) -> Dict[str, Any]:
    """Compact per-frame identity counts for the dashboard summary."""

    verified = [i for i in identities or [] if getattr(i, "verified", False)]
    return {
        "faces_seen": len(identities or []),
        "workers_verified": len(verified),
        "verified_workers": sorted(
            {
                (i.worker_name or i.worker_id or "worker")
                for i in verified
            }
        ),
        "unknown_faces": len(identities or []) - len(verified),
    }
