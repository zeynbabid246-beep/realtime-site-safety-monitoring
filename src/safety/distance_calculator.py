"""
Person <-> machinery distance calculation.

CHANGELOG (vs original)
------------------------
- Standardized on "distance_px" as the canonical output field (kept
  "distance" as a deprecated alias so rules.py's fallback logic still
  works during migration).
- find_closest_machine / calculate_person_machine_distances now skip
  machines with a malformed bbox instead of letting an unpacking error
  propagate and kill the whole frame's analysis - important once this
  runs against real detector output where an occasional bad box is
  expected, not exceptional.
- calculate_ground_distance / calculate_center_distance now validate
  bbox length instead of blindly unpacking 4 values.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

from src.safety.geometry import bbox_separation

logger = logging.getLogger(__name__)


def _validate_bbox(bbox: List[float]) -> Tuple[float, float, float, float]:
    if bbox is None or len(bbox) < 4:
        raise ValueError(f"Invalid bbox: {bbox!r}")
    x1, y1, x2, y2 = map(float, bbox[:4])
    return x1, y1, x2, y2


def get_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """Calculate the center point of a bounding box."""

    x1, y1, x2, y2 = _validate_bbox(bbox)
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def get_bottom_center(bbox: List[float]) -> Tuple[float, float]:
    """
    Returns the bottom-center point of a bounding box.

    Useful for construction-site distance estimation because the
    bottom of the bounding box approximately corresponds to the
    object's contact point with the ground.
    """

    x1, y1, x2, y2 = _validate_bbox(bbox)
    return ((x1 + x2) / 2, y2)


def calculate_point_distance(
    point1: Tuple[float, float], point2: Tuple[float, float]
) -> float:
    """Calculate Euclidean distance between two points."""

    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_center_distance(
    person_bbox: List[float], machine_bbox: List[float]
) -> float:
    """Distance between bbox centers of a person and a machine."""

    person_center = get_bbox_center(person_bbox)
    machine_center = get_bbox_center(machine_bbox)
    return calculate_point_distance(person_center, machine_center)


def calculate_ground_distance(
    person_bbox: List[float], machine_bbox: List[float]
) -> float:
    """
    Distance using bottom-center points.

    Generally more meaningful than center-distance for construction
    video because bottom-center approximates ground position. Prefer
    calculate_proximity_distance() for safety alerts.
    """

    person_point = get_bottom_center(person_bbox)
    machine_point = get_bottom_center(machine_bbox)
    return calculate_point_distance(person_point, machine_point)


def calculate_proximity_distance(
    person_bbox: List[float], machine_bbox: List[float]
) -> float:
    """
    Conservative person↔machine distance for alerts.

    Uses the gap between boxes (0 if they overlap). A large machine
    box would otherwise look "far" from a worker standing at its
    tracks if we only measured centroid-to-centroid.
    """

    return bbox_separation(person_bbox, machine_bbox)


# ============================================================
# CALIBRATION-FREE PERSPECTIVE (DEPTH-AWARE) DISTANCE
# ============================================================
# A pure pixel gap is unreliable: two objects near the horizon can be
# 20px apart yet 30m apart in reality, while two foreground objects can
# be 200px apart yet only 1m apart. Without a calibrated camera we can
# still recover an approximate real-world scale from the ONE thing whose
# true size we know: a standing human.
#
# A person's bounding-box height in pixels corresponds to roughly
# ASSUMED_PERSON_HEIGHT_M metres at THAT person's depth, so
#     px_per_m = person_height_px / ASSUMED_PERSON_HEIGHT_M
# is a per-frame, depth-aware scale. Dividing the edge-to-edge pixel gap
# by px_per_m gives an approximate distance in metres that no longer
# collapses to nothing just because two distant objects sit close
# together in the image.
#
# This assumes the person and the machine are at a similar depth, which
# holds in exactly the case we care about (they are near each other).
# When they are far apart the estimate is rough - but we do not alert
# then anyway.

DEFAULT_ASSUMED_PERSON_HEIGHT_M = 1.7
MIN_PERSON_HEIGHT_PX = 8.0  # below this the px-per-metre scale is too noisy to trust


def calculate_perspective_distance(
    person_bbox: List[float],
    machine_bbox: List[float],
    assumed_person_height_m: float = DEFAULT_ASSUMED_PERSON_HEIGHT_M,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Depth-aware person↔machine proximity estimate in metres.

    Returns (distance_m_est, px_per_m), or (None, None) when the person
    box is too small/degenerate to give a trustworthy scale (callers
    should then fall back to the pixel-gap threshold).
    """

    try:
        _, py1, _, py2 = _validate_bbox(person_bbox)
    except (ValueError, TypeError):
        return None, None

    person_height_px = float(py2) - float(py1)
    if person_height_px < MIN_PERSON_HEIGHT_PX or assumed_person_height_m <= 0:
        return None, None

    px_per_m = person_height_px / float(assumed_person_height_m)

    try:
        gap_px = bbox_separation(person_bbox, machine_bbox)
    except (ValueError, TypeError):
        return None, None

    return gap_px / px_per_m, px_per_m


def find_closest_machine(
    person_bbox: List[float],
    machines: List[Dict],
    assumed_person_height_m: float = DEFAULT_ASSUMED_PERSON_HEIGHT_M,
) -> Optional[Dict]:
    """
    Find the closest detected machine to a person.

    Expected machines format:
        [{"class_name": "Excavator", "confidence": 0.91, "bbox": [...]}]

    Returns a copy of the closest machine dict with distance keys added
    ("distance_px"/"distance" pixel gap, "ground_distance_px", and the
    depth-aware "distance_m_est"/"px_per_m"), or None if no valid machine
    bbox is available.

    Ranking is by pixel gap. The perspective scale (px_per_m) depends only
    on the person box, so it is identical for every candidate machine and
    would not change the ordering - ranking by pixels is therefore exactly
    as correct and cheaper.
    """

    if not machines:
        return None

    closest_machine = None
    minimum_distance = float("inf")

    for machine in machines:
        bbox = machine.get("bbox")
        if bbox is None:
            continue

        try:
            distance = calculate_proximity_distance(person_bbox, bbox)
            ground_distance = calculate_ground_distance(person_bbox, bbox)
        except (ValueError, TypeError) as exc:
            logger.debug("Skipping machine with malformed bbox %s: %s", bbox, exc)
            continue

        if distance < minimum_distance:
            minimum_distance = distance
            distance_m_est, px_per_m = calculate_perspective_distance(
                person_bbox, bbox, assumed_person_height_m
            )
            closest_machine = {
                **machine,
                "distance_px": distance,
                "distance": distance,  # deprecated alias
                "ground_distance_px": ground_distance,
                "distance_m_est": distance_m_est,
                "px_per_m": px_per_m,
            }

    return closest_machine


def calculate_person_machine_distances(
    persons: List[Dict],
    machines: List[Dict],
    assumed_person_height_m: float = DEFAULT_ASSUMED_PERSON_HEIGHT_M,
) -> List[Dict]:
    """
    Calculate the closest machine for every tracked person.

    Expected persons format:
        [{"track_id": 1, "bbox": [x1, y1, x2, y2], "confidence": 0.89}]

    Returns:
        [
            {
                "track_id": 1,
                "person_bbox": [...],
                "machine_class": "Excavator",
                "machine_bbox": [...],
                "machine_confidence": 0.91,
                "distance_px": 130.5,
                "distance": 130.5,        # deprecated alias
                "distance_m_est": 1.9,    # depth-aware estimate (None if unreliable)
                "px_per_m": 68.7,         # person-height-derived scale
            }
        ]
    """

    results = []

    for person in persons:
        person_bbox = person.get("bbox")
        if person_bbox is None:
            continue

        try:
            closest_machine = find_closest_machine(
                person_bbox, machines, assumed_person_height_m
            )
        except (ValueError, TypeError) as exc:
            logger.debug(
                "Skipping person %s with malformed bbox %s: %s",
                person.get("track_id"),
                person_bbox,
                exc,
            )
            continue

        if closest_machine is None:
            results.append(
                {
                    "track_id": person.get("track_id"),
                    "person_bbox": person_bbox,
                    "machine_class": None,
                    "machine_bbox": None,
                    "distance_px": None,
                    "distance": None,
                    "distance_m_est": None,
                    "px_per_m": None,
                }
            )
            continue

        results.append(
            {
                "track_id": person.get("track_id"),
                "person_bbox": person_bbox,
                "machine_class": closest_machine.get("class_name"),
                "machine_bbox": closest_machine.get("bbox"),
                "machine_confidence": closest_machine.get("confidence"),
                "distance_px": closest_machine.get("distance_px"),
                "distance": closest_machine.get("distance"),  # deprecated alias
                "ground_distance_px": closest_machine.get("ground_distance_px"),
                "distance_m_est": closest_machine.get("distance_m_est"),
                "px_per_m": closest_machine.get("px_per_m"),
            }
        )

    return results
