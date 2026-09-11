"""
Safety rules for the construction safety system.

Responsibilities:
- PPE violation rules
- Dangerous-zone rules
- Person/machinery proximity rules
- Utility-pole proximity rules
- Risk-level calculation

CHANGELOG (vs original)
------------------------
- Fixed a real bug in calculate_risk_level(): the old logic branched on
  raw violation COUNT before checking per-violation severity, so e.g.
  1 PPE violation + 1 POLE_PROXIMITY violation (2 violations total, 1
  of them HIGH-severity) resolved to MEDIUM even though a HIGH-severity
  structural hazard was present. The new version derives risk primarily
  from the "severity" field already attached to each violation dict
  (which was previously computed but never used!), with a small,
  clearly-documented set of "combo" upgrades for genuinely worse
  compound situations.
- Standardized on "distance_px" as the canonical distance field coming
  from geometry.py / distance_calculator.py, while still accepting the
  older "distance" / "distance_pixels" names so this keeps working
  during your migration.
- get_ppe_violations is now O(persons x ppe_detections) with an early
  class-id filter instead of re-scanning all detections per person
  unfiltered - same complexity class but a bit less wasted work per
  iteration, and it's now robust to a missing/malformed person bbox
  without crashing the whole batch.
"""

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ============================================================
# MODEL CLASS IDS
# ============================================================

HARDHAT = 0
MASK = 1
NO_HARDHAT = 2
NO_MASK = 3
NO_SAFETY_VEST = 4
PERSON = 5
SAFETY_CONE = 6
SAFETY_VEST = 7
MACHINERY = 8
UTILITY_POLE = 9
VEHICLE = 10


# ============================================================
# SAFETY CONFIGURATION
# ============================================================

@dataclass
class SafetyConfig:
    """Central configuration for the safety engine."""

    # Person <-> machinery distance.
    machine_distance_threshold: float = 250.0

    # Machinery/vehicle <-> utility pole distance.
    pole_distance_threshold: float = 250.0

    # Depth-aware person<->machinery proximity (calibration-free).
    # When use_perspective_distance is True and a person box is tall
    # enough to give a trustworthy scale, MACHINE_PROXIMITY is decided by
    # the estimated real-world gap in metres (machine_distance_threshold_m)
    # instead of the raw pixel gap. This stops two far-away objects that
    # happen to be near each other in the image from firing a proximity
    # alert. The metre threshold is resolution-INDEPENDENT, so unlike the
    # pixel thresholds below it is never scaled by frame width. The pixel
    # threshold is kept as a fallback for when the perspective scale is
    # unavailable (tiny/partial person box).
    use_perspective_distance: bool = True
    assumed_person_height_m: float = 1.7
    machine_distance_threshold_m: float = 2.5

    # Minimum confidence for explicit PPE violations (generic floor).
    # NO_SAFETY_VEST has an additional, stricter per-class floor applied
    # inside get_ppe_violations() - see NO_VEST_MIN_CONFIDENCE below.
    violation_confidence: float = 0.35

    # Stricter confidence floor specifically for NO_SAFETY_VEST: bright
    # high-vis vests produce split detections where the model outputs
    # both SAFETY_VEST (high conf) and NO_SAFETY_VEST (low conf) on the
    # same torso. This higher floor discards the weak false-violation leg.
    no_vest_min_confidence: float = 0.50

    # Cone clustering.
    cone_min_cluster_size: int = 3
    cone_min_samples: int = 1

    # Minimum overlap between a PPE detection's box and the PERSON'S
    # ANATOMICAL REGION it's being matched against (head region for
    # Hardhat/NO-Hardhat, torso region for Safety Vest/NO-Safety Vest -
    # see _head_region()/_torso_region() below), measured as the
    # fraction of the PPE box that falls inside that region.
    #
    # CHANGED from the old whole-person-bbox overlap scheme (which
    # used to default to 0.05 - very permissive, and prone to
    # attributing a NO_HARDHAT box to the wrong nearby person in a
    # crowded scene, or to a person's body when the box was actually
    # near their feet). 0.4 requires most of the PPE box to genuinely
    # sit in the right anatomical region.
    ppe_overlap_threshold: float = 0.40

    # Fraction of a person's bbox height, from the top, considered the
    # "head region" for Hardhat/NO-Hardhat matching.
    head_region_fraction: float = 0.35

    # Torso region for Safety Vest/NO-Safety Vest matching, as a
    # (top_fraction, bottom_fraction) of the person's bbox height.
    torso_region_fraction: tuple = (0.15, 0.85)

    # Minimum confidence for fire/smoke violations.
    fire_violation_confidence: float = 0.25

    # Pixel thresholds above are calibrated at this frame width.
    # Live webcam (1280) and 4K footage then scale automatically.
    reference_frame_width: float = 1280.0
    scale_thresholds_to_frame: bool = True

    # Expand cone hulls so the marked perimeter has a safety margin,
    # and drop sliver polygons that are not a real work area.
    zone_buffer_px: float = 20.0
    zone_min_area_px: float = 400.0

    # A danger zone that was not re-detected THIS frame (cones flickered
    # out of detection) stays "active" for this many frames - bridging
    # brief detection gaps so zones/violations do not strobe - after which
    # it is no longer used for drawing or for flagging people inside it.
    # The DangerZoneTracker keeps the id alive longer (max_missed_frames)
    # purely for identity continuity; this grace window is what gates
    # whether a zone still COUNTS as a live hazard. Keeps a worker from
    # being flagged inside a zone whose cones were genuinely removed.
    zone_presence_grace_frames: int = 2


def scaled_pixel_threshold(
    base_px: float,
    frame_width: Optional[float],
    reference_width: float = 1280.0,
) -> float:
    """Scale a pixel threshold from the calibration width to this frame."""

    if frame_width is None or frame_width <= 0 or reference_width <= 0:
        return float(base_px)
    return float(base_px) * (float(frame_width) / float(reference_width))


# ============================================================
# BASIC GEOMETRY
# ============================================================

def normalize_bbox(bbox: Sequence[float]):
    """Normalize bbox coordinates."""

    if bbox is None or len(bbox) < 4:
        raise ValueError("Bounding box must contain 4 values.")

    x1, y1, x2, y2 = map(float, bbox[:4])

    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def bbox_overlap_ratio(bbox_a: Sequence[float], bbox_b: Sequence[float]) -> float:
    """
    Intersection area divided by bbox_a area.

    In PPE matching, bbox_a is the person's bbox.
    """

    ax1, ay1, ax2, ay2 = normalize_bbox(bbox_a)
    bx1, by1, bx2, by2 = normalize_bbox(bbox_b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    person_area = max(1.0, (ax2 - ax1) * (ay2 - ay1))

    return float(intersection / person_area)


# ============================================================
# DETECTION HELPERS
# ============================================================

def detection_bbox(detection: Any):
    if isinstance(detection, dict):
        return detection.get("bbox", detection.get("box"))

    if isinstance(detection, (list, tuple)):
        return detection[:4]

    return None


def detection_class(detection: Any) -> Optional[int]:
    if isinstance(detection, dict):
        value = detection.get("class_id")
        if value is None:
            value = detection.get("class")
        if value is None:
            value = detection.get("cls")
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    if isinstance(detection, (list, tuple)):
        if len(detection) < 6:
            return None
        try:
            return int(detection[5])
        except (TypeError, ValueError):
            return None

    return None


def detection_confidence(detection: Any) -> float:
    if isinstance(detection, dict):
        value = detection.get("confidence")
        if value is None:
            value = detection.get("score")
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    if isinstance(detection, (list, tuple)):
        if len(detection) < 5:
            return 0.0
        try:
            return float(detection[4])
        except (TypeError, ValueError):
            return 0.0

    return 0.0


def get_track_id(person: Dict[str, Any]):
    """Get tracking ID from a tracked person."""

    track_id = person.get("track_id")
    if track_id is None:
        track_id = person.get("id")
    return track_id


def _extract_distance(result: Dict[str, Any]) -> Optional[float]:
    """
    Pull a distance value out of a result dict regardless of which
    field name produced it. Canonical name going forward is
    "distance_px"; "distance" and "distance_pixels" are accepted for
    backward compatibility while the rest of the codebase migrates.
    """

    for key in ("distance_px", "distance", "distance_pixels"):
        value = result.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue

    return None


# ============================================================
# PPE <-> PERSON ANATOMICAL ASSOCIATION
# ============================================================

def _head_region(person_bbox: Sequence[float], head_fraction: float) -> Tuple[float, float, float, float]:
    """Top `head_fraction` of a person's bbox height - for Hardhat/NO-Hardhat matching."""

    x1, y1, x2, y2 = normalize_bbox(person_bbox)
    height = y2 - y1
    return (x1, y1, x2, y1 + height * head_fraction)


def _torso_region(
    person_bbox: Sequence[float], fractions: Tuple[float, float]
) -> Tuple[float, float, float, float]:
    """Middle band of a person's bbox height - for Safety Vest/NO-Safety Vest matching."""

    top_fraction, bottom_fraction = fractions
    x1, y1, x2, y2 = normalize_bbox(person_bbox)
    height = y2 - y1
    return (x1, y1 + height * top_fraction, x2, y1 + height * bottom_fraction)


# ============================================================
# PPE RULES
# ============================================================

def get_ppe_violations(
    persons: List[Dict],
    detections: List,
    config: SafetyConfig,
) -> List[Dict]:
    """
    Detect explicit PPE violations.

    We only trigger NO_HARDHAT / NO_SAFETY_VEST. We intentionally do
    NOT assume that a missing positive Hardhat/Vest detection means a
    violation (too many false positives from occlusion/angle).

    CHANGED from a naive "does this PPE box overlap the person's WHOLE
    bbox" check to anatomical-region matching + single-best-match
    assignment:

      - NO_HARDHAT is only matched against a person's HEAD region (top
        `head_region_fraction` of their bbox height), not their whole
        body - a stray box near someone's feet can no longer register
        as a hardhat violation.
      - NO_SAFETY_VEST is only matched against the TORSO region.
      - Each PPE detection is assigned to at most ONE person - whoever
        has the highest region-overlap - instead of potentially firing
        a violation against every nearby person whose bbox happens to
        overlap it. This directly addresses misattribution in crowded
        scenes with many overlapping worker boxes.
    """

    violations: List[Dict[str, Any]] = []

    # Pre-filter once: only NO_HARDHAT / NO_SAFETY_VEST / NO_MASK above
    # the confidence threshold are relevant for violations.
    relevant_detections = []
    for detection in detections:
        class_id = detection_class(detection)
        if class_id not in (NO_HARDHAT, NO_SAFETY_VEST, NO_MASK):
            continue

        confidence = detection_confidence(detection)

        # NO_SAFETY_VEST uses a stricter per-class floor: bright hi-vis vests
        # often produce a low-confidence NO_SAFETY_VEST alongside a high-
        # confidence SAFETY_VEST on the same torso. The extra floor discards
        # that weak false-violation leg before it reaches matching.
        if class_id == NO_SAFETY_VEST:
            min_conf = max(config.violation_confidence, config.no_vest_min_confidence)
        else:
            min_conf = config.violation_confidence

        if confidence < min_conf:
            continue

        bbox = detection_bbox(detection)
        if bbox is None:
            continue

        relevant_detections.append((class_id, confidence, bbox))

    if not relevant_detections or not persons:
        return violations

    # Precompute each person's head/torso region once (not once per PPE box).
    person_regions = []
    for person in persons:
        person_bbox = person.get("bbox")
        if person_bbox is None:
            continue

        try:
            head_region = _head_region(person_bbox, config.head_region_fraction)
            torso_region = _torso_region(person_bbox, config.torso_region_fraction)
        except (TypeError, ValueError):
            continue

        person_regions.append(
            {
                "track_id": get_track_id(person),
                "head_region": head_region,
                "torso_region": torso_region,
            }
        )

    # ------------------------------------------------------------------
    # SAFETY_VEST cancellation: for each person build a set of track_ids
    # that already have a confirmed SAFETY_VEST on their torso. Any
    # NO_SAFETY_VEST violation for the same person will be suppressed.
    # This is the primary fix for workers in bright hi-vis vests where the
    # model simultaneously outputs SAFETY_VEST (high conf) and NO_SAFETY_VEST
    # (lower conf) on the same torso region.
    # ------------------------------------------------------------------
    protected_track_ids: set = set()
    for detection in detections:
        if detection_class(detection) != SAFETY_VEST:
            continue
        vest_conf = detection_confidence(detection)
        if vest_conf < 0.40:  # only count a reasonably confident positive vest
            continue
        vest_bbox = detection_bbox(detection)
        if vest_bbox is None:
            continue

        for pr in person_regions:
            try:
                overlap = bbox_overlap_ratio(vest_bbox, pr["torso_region"])
            except (TypeError, ValueError):
                continue
            if overlap >= config.ppe_overlap_threshold:
                protected_track_ids.add(pr["track_id"])
                break  # vest is claimed; move to next detection

    for class_id, confidence, bbox in relevant_detections:
        if class_id == NO_HARDHAT:
            region_key = "head_region"
            violation_type = "NO_HARDHAT"
            message_suffix = "is not wearing a hardhat"
            severity = "MEDIUM"
        elif class_id == NO_SAFETY_VEST:
            region_key = "torso_region"
            violation_type = "NO_SAFETY_VEST"
            message_suffix = "is not wearing a safety vest"
            severity = "MEDIUM"
        elif class_id == NO_MASK:
            region_key = "head_region"
            violation_type = "NO_MASK"
            message_suffix = "is not wearing a mask"
            severity = "LOW"
        else:
            continue

        best_person = None
        best_overlap = 0.0

        for person_region in person_regions:
            region = person_region[region_key]
            try:
                # Fraction of the PPE box that falls inside the
                # anatomical region (bbox_a=PPE box, so the ratio is
                # relative to the PPE box's own area).
                overlap = bbox_overlap_ratio(bbox, region)
            except (TypeError, ValueError):
                continue

            if overlap > best_overlap:
                best_overlap = overlap
                best_person = person_region

        if best_person is None or best_overlap < config.ppe_overlap_threshold:
            continue

        # Suppress NO_SAFETY_VEST for any person who already has a confirmed
        # SAFETY_VEST on their torso - the model is just confused by colour.
        if class_id == NO_SAFETY_VEST and best_person["track_id"] in protected_track_ids:
            continue

        violations.append(
            {
                "type": violation_type,
                "track_id": best_person["track_id"],
                "confidence": confidence,
                "overlap": best_overlap,
                "severity": severity,
                "message": f"Person {best_person['track_id']} {message_suffix}",
            }
        )

    return violations


# ============================================================
# DANGEROUS ZONE
# ============================================================

def get_zone_violations(zone_people: List[Dict]) -> List[Dict]:
    """Convert geometry zone results into violations."""

    violations: List[Dict[str, Any]] = []

    for item in zone_people:
        track_id = item.get("track_id")
        zone_id = item.get("zone_id")
        confidence = item.get("confidence", 1.0)

        violations.append(
            {
                "type": "DANGEROUS_ZONE",
                "track_id": track_id,
                "zone_id": zone_id,
                "confidence": float(confidence),
                "severity": "HIGH",
                "message": f"Person {track_id} is inside dangerous zone {zone_id}",
            }
        )

    return violations


# ============================================================
# MACHINE DISTANCE
# ============================================================

def get_machine_distance_violations(
    distance_results: List[Dict],
    config: SafetyConfig,
) -> List[Dict]:
    """
    Detect persons too close to machinery.

    Decision metric (balanced, depth-aware):
      - If perspective distance is enabled AND this result has a
        trustworthy distance_m_est, alert when the estimated real-world
        gap is within config.machine_distance_threshold_m metres. This is
        resolution-independent and does not mis-fire on distant objects
        that merely look close in the image.
      - Otherwise (tiny/partial person box, or perspective disabled) fall
        back to the pixel gap against config.machine_distance_threshold
        (already scaled to frame width by SafetyEngine).
    """

    violations: List[Dict[str, Any]] = []

    for result in distance_results:
        distance_px = _extract_distance(result)
        distance_m = result.get("distance_m_est")

        use_perspective = (
            config.use_perspective_distance and distance_m is not None
        )

        if use_perspective:
            too_close = distance_m <= config.machine_distance_threshold_m
        else:
            too_close = (
                distance_px is not None
                and distance_px <= config.machine_distance_threshold
            )

        if not too_close:
            continue

        track_id = result.get("track_id")
        machine_class = result.get("machine_class") or "machinery"
        machine_confidence = result.get(
            "machine_confidence", result.get("confidence", 0.0)
        )

        if use_perspective:
            detail = f"{distance_m:.1f} m"
        elif distance_px is not None:
            detail = f"{distance_px:.1f} px"
        else:
            detail = "unknown distance"

        violations.append(
            {
                "type": "MACHINE_PROXIMITY",
                "track_id": track_id,
                "machine_class": machine_class,
                "distance_px": distance_px,
                "distance": distance_px,  # deprecated alias
                "distance_m_est": distance_m,
                "measurement": "metres" if use_perspective else "pixels",
                "confidence": float(machine_confidence),
                "severity": "HIGH",
                "message": (
                    f"Person {track_id} is too close to {machine_class} "
                    f"({detail})"
                ),
            }
        )

    return violations


# ============================================================
# UTILITY POLE PROXIMITY
# ============================================================

def get_pole_proximity_violations(
    pole_results: List[Dict],
    config: SafetyConfig,
) -> List[Dict]:
    """Detect machinery/vehicles too close to utility poles."""

    violations: List[Dict[str, Any]] = []

    for result in pole_results:
        distance = _extract_distance(result)
        if distance is None or distance > config.pole_distance_threshold:
            continue

        object_class_id = result.get("object_class_id")
        object_type = "vehicle" if object_class_id == VEHICLE else "machinery"
        confidence = result.get("object_confidence", 1.0)

        violations.append(
            {
                "type": "POLE_PROXIMITY",
                "object_class": object_type,
                "object_class_id": object_class_id,
                "distance_px": distance,
                "distance": distance,  # deprecated alias
                "confidence": float(confidence),
                "severity": "HIGH",
                "message": (
                    f"{object_type.capitalize()} is too close to a utility pole "
                    f"({distance:.1f} px)"
                ),
            }
        )

    return violations


# ============================================================
# FIRE / SMOKE
# ============================================================

def get_fire_violations(
    fire_detections: List[Dict],
    config: SafetyConfig,
) -> List[Dict]:
    """
    Convert normalized fire/smoke detections (from FireDetector) into
    violations.

    Fire is treated as CRITICAL on its own - it doesn't need to combine
    with anything else to be the worst thing happening on site. Smoke
    alone is HIGH (could be a false alarm, dust, or an early fire that
    isn't clearly visible yet, so it's serious but not auto-CRITICAL).
    """

    violations: List[Dict[str, Any]] = []

    for detection in fire_detections:
        confidence = detection.get("confidence", 0.0)

        if confidence < config.fire_violation_confidence:
            continue

        label = (detection.get("type") or detection.get("class_name") or "").lower()

        if "fire" in label:
            violation_type = "FIRE"
            severity = "CRITICAL"
            message = f"Fire detected (confidence {confidence:.2f})"
        elif "smoke" in label:
            violation_type = "SMOKE"
            severity = "HIGH"
            message = f"Smoke detected (confidence {confidence:.2f})"
        else:
            # Unknown label from this model - skip rather than guess.
            continue

        violations.append(
            {
                "type": violation_type,
                "confidence": float(confidence),
                "bbox": detection.get("bbox"),
                "severity": severity,
                "message": message,
            }
        )

    return violations


# ============================================================
# RISK LEVEL
# ============================================================

SERIOUS_VIOLATION_TYPES = {
    "MACHINE_PROXIMITY",
    "DANGEROUS_ZONE",
    "POLE_PROXIMITY",
}

_SEVERITY_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


def calculate_risk_level(violations: List[Dict]) -> str:
    """
    Calculate global frame risk.

    Design (fixed from the original count-first logic):

    1. Start from the single most severe violation present
       (using each violation's own "severity" field, which every rule
       function already attaches - previously computed and ignored).
    2. Escalate a HIGH baseline to CRITICAL for specific compound
       situations that are worse than any single violation on its own:
         - a person is inside a dangerous zone AND close to machinery
           at the same time (highest-risk combination on a site)
         - 2 or more independently serious (HIGH-severity) hazards are
           active simultaneously, regardless of type
    3. A lone PPE violation (MEDIUM) does not get bumped up just
       because other unrelated MEDIUM violations exist in the same
       frame - severity should reflect the worst thing actually
       happening, with count only used as a tie-breaker/escalator
       among violations of otherwise-equal severity.

    Levels: SAFE < LOW < MEDIUM < HIGH < CRITICAL
    """

    if not violations:
        return "SAFE"

    violation_types = {v.get("type") for v in violations}

    serious_count = sum(
        1 for v in violations if v.get("type") in SERIOUS_VIOLATION_TYPES
    )

    # Worst compound situation on a construction site: someone is both
    # inside a marked-dangerous area and near active machinery.
    if "DANGEROUS_ZONE" in violation_types and "MACHINE_PROXIMITY" in violation_types:
        return "CRITICAL"

    # Two or more independently serious hazards at once (e.g. a pole
    # proximity issue plus a dangerous-zone breach involving different
    # people/objects) is worse than either alone, even without the
    # specific zone+machine combo above.
    if serious_count >= 2:
        return "CRITICAL"

    # Take the highest severity actually reported by any single
    # violation - this is the fix for the original bug where a HIGH
    # violation could get out-voted down to MEDIUM by unrelated
    # violation count.
    worst_rank = max(
        _SEVERITY_RANK.get(v.get("severity", "LOW"), 1) for v in violations
    )
    worst_severity = {
        rank: name for name, rank in _SEVERITY_RANK.items()
    }[worst_rank]

    # Many simultaneous MEDIUM-or-below violations (e.g. multiple
    # different PPE breaches) still deserve an escalation even with no
    # single "serious" hazard type present.
    if worst_severity in ("LOW", "MEDIUM") and len(violations) >= 3:
        return "HIGH"

    return worst_severity