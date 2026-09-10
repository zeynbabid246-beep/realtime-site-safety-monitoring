"""
Detection-level filtering for the hazard model.

Real construction video produces detections that are technically
"valid" model outputs but are not safety-relevant: a tiny 15x20px
smudge classified as Person at 0.31 (ID:15, ID:714 in your
screenshots), a background house classified as machinery at 0.68-0.82
confidence, a thin pole classified as Person.

ARCHITECTURE NOTE - why this is its own layer, not inside SafetyEngine
------------------------------------------------------------------
    YOLO -> [THIS MODULE: filtering] -> tracking (TrackConfirmationTracker)
    -> geometry -> SafetyEngine -> risk level

SafetyEngine's job is "given objects I can trust are real, is this
dangerous?" - it should never have to ask "is this detection even
real?". Keeping that question here means SafetyEngine stays about
SAFETY, not about model quality, and this module can be tuned or
replaced (e.g. once you fine-tune with hard-negative examples) without
touching rules.py at all.

Two stateless filters (no memory across frames - that's what
TrackConfirmationTracker in track_confirmation.py is for):

    1. Per-class confidence threshold. A generic 0.25 lets obvious
       junk through for some classes while needlessly rejecting good
       detections for others.
    2. Per-class minimum bounding-box size. A hardhat is naturally
       much smaller than a person - ONE minimum size either rejects
       real small PPE or lets through tiny "person"/"machinery" noise.

Plus one person-only shape filter (does not touch fire/smoke):

    3. Person aspect ratio (height / width). Tiny debris and thin
       poles mislabeled as Person usually sit outside a plausible
       human range. Crouching workers can be fairly square, so the
       lower bound is permissive (0.7); poles are typically >6.

Plus one OPTIONAL, opt-in filter for FIXED cameras only:

    4. ROI (region of interest). If a camera is fixed and you know a
       region of the frame is background (sky, houses, road) that can
       never be a real hazard, pass a Shapely Polygon as roi_polygon
       and any detection whose bottom-center falls outside it is
       dropped - regardless of confidence. This is the only practical
       mitigation here for "house classified as machinery at 0.82
       confidence" (see build_rectangular_roi() below for an easy way
       to build one). It does NOT fix the underlying model error (the
       model still thinks that pattern IS machinery) - it just stops
       that specific background region from ever being considered.
       Do NOT use ROI for a moving/drone/panning camera - the "safe"
       region moves with the camera and a static ROI will silently
       exclude real hazards.

    The real fix for semantic misclassification (house -> machinery)
    is hard-negative fine-tuning: collect frames where this happens,
    label the house/building/wall with NO machinery label, and
    retrain. Filtering can only ever be a stopgap for this specific
    failure mode.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

# ============================================================
# PER-CLASS CONFIDENCE THRESHOLDS
# ============================================================
# Keyed by class NAME (not id) so this stays readable and doesn't
# silently break if class ids get reordered in a re-trained model.
# Falls back to DEFAULT_MIN_CONFIDENCE for anything not listed.

DEFAULT_MIN_CONFIDENCE = 0.25

MIN_CONFIDENCE_BY_CLASS: Dict[str, float] = {
    "Person": 0.35,            # Lowered from 0.52 to avoid missing valid detections
    "machinery": 0.52,       # Catches house -> machinery 0.34-0.50
    "vehicle": 0.52,
    "Hardhat": 0.45,
    "NO-Hardhat": 0.30,        # Lowered from 0.40 to match violation_confidence
    "Safety Vest": 0.40,
    "NO-Safety Vest": 0.30,    # Lowered from 0.40 to match violation_confidence
    "Mask": 0.35,
    "NO-Mask": 0.30,           # Lowered from 0.40 to match violation_confidence
    "Safety Cone": 0.40,
    "utility pole": 0.50,
}

# ============================================================
# PER-CLASS MINIMUM BOUNDING-BOX SIZE (at 640x640 baseline)
# ============================================================
# When frame_shape is provided, these are dynamically scaled
# proportionally to the frame resolution so a 30px threshold doesn't
# allow microscopic 30px debris in 1080p or 4K videos.

DEFAULT_MIN_SIZE: Tuple[float, float] = (15.0, 15.0)

MIN_SIZE_BY_CLASS: Dict[str, Tuple[float, float]] = {
    "Person": (30.0, 55.0),
    "machinery": (50.0, 40.0),
    "vehicle": (50.0, 40.0),
    "Hardhat": (10.0, 10.0),
    "NO-Hardhat": (10.0, 10.0),
    "Safety Vest": (20.0, 20.0),
    "NO-Safety Vest": (20.0, 20.0),
    "Safety Cone": (10.0, 10.0),
    "utility pole": (10.0, 30.0),
}

# Edge-of-frame confidence bonus: partial/cropped peripheral objects
# must clear a higher confidence bar.
EDGE_MARGIN_PX = 8.0
EDGE_CONFIDENCE_BONUS: Dict[str, float] = {
    "Person": 0.12,
    "machinery": 0.10,
    "vehicle": 0.10,
}
DEFAULT_FALLBACK_EDGE_BONUS = 0.08

# ============================================================
# PERSON ASPECT-RATIO FILTER (hazard model only)
# ============================================================
# height / width. Independent of fire/smoke thresholds.

PERSON_MIN_ASPECT_RATIO = 0.8
PERSON_MAX_ASPECT_RATIO = 5.0
PERSON_ASPECT_RATIO_STRICT = True


def _bbox_size(bbox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox[:4]
    return (max(0.0, float(x2) - float(x1)), max(0.0, float(y2) - float(y1)))


def touches_frame_edge(
    bbox: Sequence[float],
    frame_shape: Optional[Tuple[int, int]],
    margin: float = EDGE_MARGIN_PX,
) -> bool:
    """True if bbox touches or is within `margin` pixels of image border."""
    if frame_shape is None:
        return False
    fh, fw = frame_shape[0], frame_shape[1]
    x1, y1, x2, y2 = bbox[:4]
    return (
        x1 <= margin
        or y1 <= margin
        or x2 >= fw - margin
        or y2 >= fh - margin
    )


def passes_confidence(
    class_name: str,
    confidence: float,
    bbox: Optional[Sequence[float]] = None,
    frame_shape: Optional[Tuple[int, int]] = None,
) -> bool:
    """Check confidence threshold with edge-of-frame penalty."""
    threshold = MIN_CONFIDENCE_BY_CLASS.get(class_name, DEFAULT_MIN_CONFIDENCE)
    if bbox is not None and touches_frame_edge(bbox, frame_shape):
        threshold += EDGE_CONFIDENCE_BONUS.get(class_name, DEFAULT_FALLBACK_EDGE_BONUS)
    return confidence >= threshold


def passes_size(
    class_name: str,
    bbox: Sequence[float],
    frame_shape: Optional[Tuple[int, int]] = None,
) -> bool:
    """
    True if detection meets minimum (width, height) for its class.

    Dynamically scales the size floor with frame resolution so a
    threshold tuned for 640p doesn't let microscopic debris pass
    in a 1080p or 4K frame.

    Also incorporates perspective vertical scaling for Person:
    foreground objects (bottom of frame) MUST be substantially larger
    than distant background objects near the horizon.
    """
    base_w, base_h = MIN_SIZE_BY_CLASS.get(class_name, DEFAULT_MIN_SIZE)
    width, height = _bbox_size(bbox)

    if frame_shape is not None and frame_shape[0] > 0 and frame_shape[1] > 0:
        fh, fw = frame_shape[0], frame_shape[1]
        # Reference scale relative to standard 640x640 YOLO training size
        scale_w = fw / 640.0
        scale_h = fh / 640.0

        if class_name == "Person":
            y2 = float(bbox[3])
            y_norm = min(1.0, max(0.0, y2 / fh))
            # Foreground perspective scale: objects near bottom (y_norm > 0.4)
            # must be larger because they are closer to camera.
            persp_scale = 1.0 + 1.2 * max(0.0, y_norm - 0.35)
            min_w = base_w * scale_w * persp_scale
            min_h = base_h * scale_h * persp_scale
        else:
            min_w = base_w * scale_w
            min_h = base_h * scale_h
    else:
        min_w, min_h = base_w, base_h

    return width >= min_w and height >= min_h


def person_aspect_ratio_ok(bbox: Sequence[float], confidence: float = 1.0) -> bool:
    """
    True if the box looks like a person, or if this filter is disabled.

    Rejects needle boxes (poles, wires, cables) and pancake boxes (debris).
    For lower confidence (<0.65), applies stricter human bounds (1.1 to 3.8).
    """
    if not PERSON_ASPECT_RATIO_STRICT:
        return True

    width, height = _bbox_size(bbox)
    if width <= 0 or height <= 0:
        return False

    ratio = height / width
    if confidence < 0.65:
        # Strict for uncertain detections: reject needle poles and wide pancakes
        return 1.1 <= ratio <= 3.8

    return PERSON_MIN_ASPECT_RATIO <= ratio <= PERSON_MAX_ASPECT_RATIO


def passes_person_aspect_ratio(bbox: Sequence[float]) -> bool:
    """Geometric check only (ignores PERSON_ASPECT_RATIO_STRICT)."""
    width, height = _bbox_size(bbox)
    if width <= 0 or height <= 0:
        return False
    ratio = height / width
    return PERSON_MIN_ASPECT_RATIO <= ratio <= PERSON_MAX_ASPECT_RATIO


def is_detection_valid(
    detection: Dict[str, Any],
    roi_polygon: Any = None,
    frame_shape: Optional[Tuple[int, int]] = None,
) -> bool:
    """True if this single detection passes confidence + size (+ aspect ratio for Person, + ROI)."""
    return _validate_with_reason(detection, roi_polygon, frame_shape)[0]


def _bbox_bottom_center(bbox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox[:4]
    return ((float(x1) + float(x2)) / 2.0, float(y2))


def passes_roi(bbox, roi_polygon: Any) -> bool:
    """True if detection's bottom-center falls inside roi_polygon."""
    if roi_polygon is None:
        return True

    try:
        from shapely.geometry import Point
        point = Point(*_bbox_bottom_center(bbox))
        return bool(roi_polygon.covers(point))
    except Exception:
        return True


def _validate_with_reason(
    detection: Dict[str, Any],
    roi_polygon: Any = None,
    frame_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[bool, str]:
    """Validate detection and return (is_valid, failure_reason)."""
    class_name = detection.get("class_name", "")
    confidence = detection.get("confidence", 0.0)
    bbox = detection.get("bbox")

    if bbox is None:
        return False, "missing_bbox"

    if not passes_confidence(class_name, confidence, bbox, frame_shape):
        if touches_frame_edge(bbox, frame_shape):
            return False, "edge_confidence"
        return False, "confidence"

    if not passes_size(class_name, bbox, frame_shape):
        return False, "size"

    if class_name == "Person" and not person_aspect_ratio_ok(bbox, confidence):
        return False, "aspect_ratio"

    if not passes_roi(bbox, roi_polygon):
        return False, "roi"

    return True, "kept"


def filter_detections(
    detections: List[Dict[str, Any]],
    roi_polygon: Any = None,
    frame_shape: Optional[Tuple[int, int]] = None,
) -> List[Dict[str, Any]]:
    """Drop every detection that fails is_detection_valid()."""
    return [d for d in detections if is_detection_valid(d, roi_polygon, frame_shape)]


def filter_detections_with_stats(
    detections: List[Dict[str, Any]],
    roi_polygon: Any = None,
    frame_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Same filtering as filter_detections(), but also returns breakdown stats."""
    stats = {
        "total": len(detections),
        "kept": 0,
        "rejected_confidence": 0,
        "rejected_edge_confidence": 0,
        "rejected_size": 0,
        "rejected_aspect_ratio": 0,
        "rejected_roi": 0,
        "rejected_missing_bbox": 0,
    }

    kept = []
    for detection in detections:
        ok, reason = _validate_with_reason(detection, roi_polygon, frame_shape)
        if ok:
            kept.append(detection)
            stats["kept"] += 1
        else:
            key = f"rejected_{reason}"
            stats[key] = stats.get(key, 0) + 1

    return kept, stats


def build_rectangular_roi(x1: float, y1: float, x2: float, y2: float) -> Any:
    """
    Convenience helper: build a rectangular ROI polygon from pixel
    coordinates without needing to import Shapely directly.

    Example - exclude the top 300px of a 1920x1080 fixed camera (sky/
    background houses) and only consider the work area below it:

        roi = build_rectangular_roi(0, 300, 1920, 1080)
        filtered = filter_detections(detections, roi_polygon=roi)
    """

    from shapely.geometry import box

    return box(x1, y1, x2, y2)