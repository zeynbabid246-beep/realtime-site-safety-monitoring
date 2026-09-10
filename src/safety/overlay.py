"""
Drawing helpers to turn a SafetyEngine result into an annotated frame.

Kept separate from app/main.py so the same overlay logic is reused
identically across the image, video, and websocket endpoints instead
of being copy-pasted three times with subtle drift between them.

CHANGELOG (vs previous version)
--------------------------------
- Box style is now IDENTICAL to what the YOLO model itself draws when
  you run it directly (e.g. `model.predict(..., save=True)`): same box
  shape/thickness, same deterministic per-class-id color, same
  "ClassName conf" label format, same filled label background. This
  uses Ultralytics' own `Annotator` + `colors()` helper (the exact
  renderer YOLO uses internally) instead of a hand-picked custom color
  scheme, so hazard classes (Hardhat, NO-Hardhat, Safety Vest,
  NO-Safety Vest, Mask, NO-Mask, Safety Cone, machinery, utility pole,
  vehicle, Person) and fire/smoke classes look exactly like native
  model output.
- Fire/smoke detections come from a SEPARATE model with its own
  class-id space (0: Fire, 1: Smoke), which would otherwise collide
  with the hazard model's own class ids 0/1 (Hardhat/Mask) and get
  colored identically. Fire/smoke class ids are offset by a constant
  before being passed to colors() so the two models' palettes never
  visually clash on the same frame.
- Falls back to manual drawing with Ultralytics' own hardcoded default
  color palette if `ultralytics.utils.plotting` can't be imported for
  some reason, so this module doesn't hard-crash the whole pipeline if
  that package is ever missing - box shape is then a plain rectangle
  instead of Ultralytics' rounded/filled-label style, but colors stay
  consistent with the real palette.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.safety.geometry import (
    get_detection_bbox,
    get_detection_class_id,
    get_detection_confidence,
    get_track_id,
    get_bottom_center,
)

try:
    from ultralytics.utils.plotting import Annotator, colors as _ultra_colors
    _HAS_ULTRALYTICS_PLOTTING = True
except ImportError:  # pragma: no cover
    Annotator = None  # type: ignore
    _HAS_ULTRALYTICS_PLOTTING = False

# Must match the hazard model's own class id for "Person" (see
# hazard_detector class_names: {0: 'Hardhat', ..., 5: 'Person', ...}).
PERSON_CLASS_ID = 5

# Fire/smoke is a SEPARATE model with its own small class-id space
# (0: Fire, 1: Smoke). Offset its ids before coloring so they never
# collide visually with the hazard model's class 0/1 (Hardhat/Mask).
FIRE_MODEL_COLOR_OFFSET = 1000

COLOR_ZONE_FILL = (0, 0, 255)
COLOR_DANGER = (0, 0, 255)
COLOR_TEXT = (255, 255, 255)

RISK_COLORS = {
    "SAFE": (80, 200, 80),
    "LOW": (0, 200, 200),
    "MEDIUM": (0, 165, 255),
    "HIGH": (0, 60, 255),
    "CRITICAL": (0, 0, 255),
}

# Ultralytics' own default 20-color palette (hex, RGB order), used only
# as a fallback if ultralytics.utils.plotting isn't importable, so
# per-class coloring still stays deterministic and consistent even in
# that edge case.
_FALLBACK_PALETTE_HEX = [
    "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17",
    "3DDB86", "1A9334", "00D4BB", "2C99A8", "00C2FF", "344593", "6473FF",
    "0018EC", "8438FF", "520085", "CB38FF", "FF95C8", "FF37C7",
]


def _fallback_color(class_id: int) -> Tuple[int, int, int]:
    """BGR color for a class id, used only if ultralytics isn't importable."""

    hex_color = _FALLBACK_PALETTE_HEX[class_id % len(_FALLBACK_PALETTE_HEX)]
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)  # OpenCV/BGR order


def _class_color(class_id: int) -> Tuple[int, int, int]:
    """
    Deterministic BGR color for a class id, matching exactly what the
    model itself would draw (same palette Ultralytics uses internally)
    when available, or the same fallback palette values otherwise.
    """

    if _HAS_ULTRALYTICS_PLOTTING:
        return _ultra_colors(class_id, True)  # BGR, bright variant

    return _fallback_color(class_id)


def _draw_zone_polygons(frame: np.ndarray, danger_zones: List[Dict[str, Any]]) -> None:
    """Draw danger-zone polygons as a translucent red fill + outline + id label."""

    if not danger_zones:
        return

    overlay = frame.copy()

    for zone in danger_zones:
        polygon = zone.get("polygon")
        if polygon is None or polygon.is_empty:
            continue

        coords = np.array(list(polygon.exterior.coords), dtype=np.int32)
        cv2.fillPoly(overlay, [coords], COLOR_ZONE_FILL)

    # Blend translucently so the fill doesn't fully obscure whatever's
    # inside the zone (workers, cones, etc.).
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, dst=frame)

    for zone in danger_zones:
        polygon = zone.get("polygon")
        if polygon is None or polygon.is_empty:
            continue

        coords = np.array(list(polygon.exterior.coords), dtype=np.int32)
        cv2.polylines(frame, [coords], isClosed=True, color=COLOR_DANGER, thickness=2)

        label_point = coords[0]
        cv2.putText(
            frame,
            f"ZONE {zone.get('zone_id')}",
            (int(label_point[0]), max(int(label_point[1]) - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_DANGER,
            2,
        )


def _label_for_detection(
    class_id: Optional[int],
    class_names: Dict[int, str],
    confidence: float,
    track_id: Optional[int],
) -> str:
    class_name = class_names.get(class_id, str(class_id)) if class_id is not None else "unknown"
    label = f"{class_name} {confidence:.2f}"
    if track_id is not None:
        label = f"ID:{track_id} {label}"
    return label


def _draw_hazard_detections_annotator(
    annotator: "Annotator",
    detections: List[Any],
    class_names: Dict[int, str],
) -> None:
    """Draw every raw hazard detection using Ultralytics' own Annotator."""

    for detection in detections:
        bbox = get_detection_bbox(detection)
        if bbox is None:
            continue

        try:
            box = [float(v) for v in bbox[:4]]
        except (TypeError, ValueError):
            continue

        class_id = get_detection_class_id(detection)
        confidence = get_detection_confidence(detection) or 0.0
        track_id = get_track_id(detection)

        label = _label_for_detection(class_id, class_names, confidence, track_id)
        color = _class_color(class_id if class_id is not None else 0)
        annotator.box_label(box, label, color=color)


def _draw_fire_detections_annotator(
    annotator: "Annotator",
    fire_detections: List[Dict[str, Any]],
) -> None:
    """Draw fire/smoke detections using Ultralytics' own Annotator."""

    for detection in fire_detections:
        bbox = detection.get("bbox")
        if bbox is None:
            continue

        try:
            box = [float(v) for v in bbox[:4]]
        except (TypeError, ValueError):
            continue

        fire_class_id = detection.get("class_id", 0) or 0
        label_name = detection.get("class_name") or detection.get("type") or "fire/smoke"
        confidence = detection.get("confidence", 0.0)

        color = _class_color(FIRE_MODEL_COLOR_OFFSET + fire_class_id)
        annotator.box_label(box, f"{label_name} {confidence:.2f}", color=color)


def _draw_hazard_detections_fallback(
    frame: np.ndarray,
    detections: List[Any],
    class_names: Dict[int, str],
) -> None:
    """Plain-rectangle fallback, used only if ultralytics isn't importable."""

    for detection in detections:
        bbox = get_detection_bbox(detection)
        if bbox is None:
            continue

        try:
            x1, y1, x2, y2 = (int(v) for v in bbox[:4])
        except (TypeError, ValueError):
            continue

        class_id = get_detection_class_id(detection)
        confidence = get_detection_confidence(detection) or 0.0
        track_id = get_track_id(detection)

        label = _label_for_detection(class_id, class_names, confidence, track_id)
        color = _class_color(class_id if class_id is not None else 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, label, (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )


def _draw_fire_detections_fallback(
    frame: np.ndarray,
    fire_detections: List[Dict[str, Any]],
) -> None:
    """Plain-rectangle fallback, used only if ultralytics isn't importable."""

    for detection in fire_detections:
        bbox = detection.get("bbox")
        if bbox is None:
            continue

        try:
            x1, y1, x2, y2 = (int(v) for v in bbox[:4])
        except (TypeError, ValueError):
            continue

        fire_class_id = detection.get("class_id", 0) or 0
        label_name = detection.get("class_name") or detection.get("type") or "fire/smoke"
        confidence = detection.get("confidence", 0.0)
        color = _class_color(FIRE_MODEL_COLOR_OFFSET + fire_class_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, f"{label_name} {confidence:.2f}", (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
        )


def _draw_proximity_lines(frame: np.ndarray, result: Dict[str, Any]) -> None:
    """Draw person↔machine ground lines and gap labels (metres when available)."""

    use_perspective = bool(result.get("use_perspective_distance"))
    threshold_px = float(result.get("proximity_threshold_px", 250.0) or 250.0)
    threshold_m = float(result.get("machine_distance_threshold_m", 2.5) or 2.5)

    for item in result.get("distance_results") or []:
        person_bbox = item.get("person_bbox")
        machine_bbox = item.get("machine_bbox")
        if person_bbox is None or machine_bbox is None:
            continue

        distance_px = item.get("distance_px")
        distance_m = item.get("distance_m_est")

        show_metres = use_perspective and distance_m is not None
        if not show_metres and distance_px is None:
            continue

        try:
            p1 = get_bottom_center(person_bbox)
            p2 = get_bottom_center(machine_bbox)
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
        except (TypeError, ValueError):
            continue

        if show_metres:
            close = distance_m <= threshold_m
            label = f"{distance_m:.1f}m"
        else:
            close = distance_px <= threshold_px
            label = f"{distance_px:.0f}px"

        color = (0, 60, 255) if close else (180, 180, 180)
        cv2.line(frame, pt1, pt2, color, 2)
        cv2.circle(frame, pt1, 5, color, -1)
        cv2.circle(frame, pt2, 5, color, -1)
        mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.putText(
            frame,
            label,
            mid,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )


def _draw_risk_banner(frame: np.ndarray, result: Dict[str, Any]) -> None:
    risk_level = result.get("risk_level", "SAFE")
    color = RISK_COLORS.get(risk_level, COLOR_TEXT)

    height, width = frame.shape[:2]
    banner_height = 42

    cv2.rectangle(frame, (0, 0), (width, banner_height), (20, 20, 20), thickness=-1)
    cv2.putText(
        frame,
        f"RISK: {risk_level}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
    )

    violation_count = result.get("violation_count", 0)
    cv2.putText(
        frame,
        f"Violations: {violation_count}",
        (260, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        COLOR_TEXT,
        1,
    )


def draw_safety_overlay(
    frame: np.ndarray,
    result: Dict[str, Any],
    detections: List[Any],
    class_names: Dict[int, str],
) -> np.ndarray:
    """
    Draw the full safety overlay onto a COPY of `frame` and return it.
    The input frame is not modified.

    - Danger zones: translucent red polygon fill/outline (not a model
      output, so this stays custom).
    - Every raw hazard detection: SAME box shape/color/label the model
      itself uses (via Ultralytics' Annotator + colors(), when
      available).
    - Fire/smoke detections: same treatment, own color band (offset so
      it never collides with the hazard model's own class 0/1 colors).
    - Risk banner: custom, since it's a SafetyEngine-level summary, not
      a per-box model output.

    Parameters
    ----------
    frame:
        The raw video frame (BGR, as read by cv2.VideoCapture).
    result:
        The dict returned by SafetyEngine.analyze() for this frame.
    detections:
        The full per-frame hazard-model output (e.g.
        HazardDetector.track(frame) or .predict(frame)) - persons, PPE,
        cones, machinery, poles, vehicles all together.
    class_names:
        The hazard model's {class_id: name} mapping (e.g.
        hazard_detector.class_names).
    """

    annotated = frame.copy()

    _draw_zone_polygons(annotated, result.get("danger_zones", []))

    fire_detections = result.get("fire_detections", [])

    if _HAS_ULTRALYTICS_PLOTTING:
        annotator = Annotator(annotated, line_width=2)
        _draw_hazard_detections_annotator(annotator, detections, class_names)
        _draw_fire_detections_annotator(annotator, fire_detections)
        annotated = annotator.result()
    else:
        _draw_hazard_detections_fallback(annotated, detections, class_names)
        _draw_fire_detections_fallback(annotated, fire_detections)

    _draw_proximity_lines(annotated, result)
    _draw_risk_banner(annotated, result)

    return annotated


def draw_unconfirmed_fire_debug(
    frame: np.ndarray,
    raw_detections: List[Dict[str, Any]],
    confirmed_detections: List[Dict[str, Any]],
) -> None:
    """
    Draw RAW fire/smoke boxes the confirmation gate dropped.

    Solid boxes already come from confirmed detections in
    draw_safety_overlay(). Dashed boxes mean: the Fire-Smoke model
    DID fire on this frame, and something downstream (confidence,
    area, or persistence) filtered it out. Empty raw list + no solid
    box means the model itself never produced a detection.
    """

    confirmed_keys = {
        (
            round(float(d["bbox"][0]), 1),
            round(float(d["bbox"][1]), 1),
            round(float(d["bbox"][2]), 1),
            round(float(d["bbox"][3]), 1),
        )
        for d in confirmed_detections
        if d.get("bbox") is not None and len(d["bbox"]) >= 4
    }

    for detection in raw_detections:
        bbox = detection.get("bbox")
        if bbox is None or len(bbox) < 4:
            continue
        key = (
            round(float(bbox[0]), 1),
            round(float(bbox[1]), 1),
            round(float(bbox[2]), 1),
            round(float(bbox[3]), 1),
        )
        if key in confirmed_keys:
            continue

        try:
            x1, y1, x2, y2 = (int(v) for v in bbox[:4])
        except (TypeError, ValueError):
            continue

        color = (0, 255, 255)  # cyan = model saw it, gate dropped it
        _draw_dashed_rect(frame, x1, y1, x2, y2, color)
        label = detection.get("class_name") or detection.get("type") or "raw"
        confidence = detection.get("confidence", 0.0)
        cv2.putText(
            frame,
            f"RAW {label} {confidence:.2f}",
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )


def _draw_dashed_rect(
    frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, color, dash: int = 8
) -> None:
    for x in range(x1, x2, dash * 2):
        cv2.line(frame, (x, y1), (min(x + dash, x2), y1), color, 1)
        cv2.line(frame, (x, y2), (min(x + dash, x2), y2), color, 1)
    for y in range(y1, y2, dash * 2):
        cv2.line(frame, (x1, y), (x1, min(y + dash, y2)), color, 1)
        cv2.line(frame, (x2, y), (x2, min(y + dash, y2)), color, 1)