"""
Geometry utilities for construction safety analysis.

Responsibilities
----------------
- Extract safety-cone positions from YOLO detections
- Cluster cones using HDBSCAN
- Build dangerous-zone polygons
- Stabilize danger zones across consecutive frames (temporal tracking)
- Check tracked persons inside dangerous zones
- Calculate machinery/vehicle proximity to utility poles

Supported YOLO detection format:
    [x1, y1, x2, y2, confidence, class_id]

Supported dictionary format:
    {
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.85,
        "class_id": 6
    }

Tracked person format:
    {
        "track_id": 12,
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.91
    }

CHANGELOG (vs original)
------------------------
- Standardized distance field name to "distance_px" everywhere a
  distance is reported (kept "distance" as a deprecated alias so
  older consumers don't break immediately).
- find_people_inside_zones now returns BOTH "zone_index" (0-based)
  and "zone_id" (1-based) since different call sites expected
  different things.
- find_machinery_near_poles now returns "machinery_index" (alias of
  "object_index") to match what callers/tests expected.
- Added logging instead of silently swallowing exceptions, which
  matters a lot once you move from synthetic data to real video
  where malformed detections *will* happen.
- Added DangerZoneTracker: matches zones frame-to-frame by polygon
  IoU and assigns persistent zone IDs, which is exactly the
  "stabilize danger-zone detection across consecutive frames" item
  on your roadmap.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from shapely.geometry import MultiPoint, Point, Polygon
except ImportError:  # pragma: no cover
    MultiPoint = None
    Point = None
    Polygon = None

try:
    from sklearn.cluster import HDBSCAN
except ImportError:  # pragma: no cover
    HDBSCAN = None


# ============================================================
# YOLO CLASS IDS
# ============================================================

PERSON_CLASS_ID = 5
SAFETY_CONE_CLASS_ID = 6
MACHINERY_CLASS_ID = 8
UTILITY_POLE_CLASS_ID = 9
VEHICLE_CLASS_ID = 10


# ============================================================
# DEFAULT PARAMETERS
# ============================================================

DEFAULT_MIN_CONES = 3
DEFAULT_MIN_SAMPLES = 1
DEFAULT_POLE_DISTANCE_THRESHOLD = 250.0
DEFAULT_ZONE_BUFFER_PX = 20.0
DEFAULT_ZONE_MIN_AREA_PX = 400.0

# Temporal zone-matching defaults.
DEFAULT_ZONE_IOU_MATCH_THRESHOLD = 0.3
DEFAULT_ZONE_MAX_MISSED_FRAMES = 5


# ============================================================
# BOUNDING BOX UTILITIES
# ============================================================

def normalize_bbox(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    """Normalize bbox coordinates to (x1, y1, x2, y2) with x1<=x2, y1<=y2."""

    if bbox is None or len(bbox) < 4:
        raise ValueError("Bounding box must contain at least 4 values.")

    x1, y1, x2, y2 = map(float, bbox[:4])

    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def get_bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    """Return the center of a bounding box."""

    x1, y1, x2, y2 = normalize_bbox(bbox)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def get_bottom_center(bbox: Sequence[float]) -> Tuple[float, float]:
    """
    Return the bottom-center of a bounding box.

    Preferred for construction-site geometry: approximates the point
    where the object touches the ground.
    """

    x1, _, x2, y2 = normalize_bbox(bbox)
    return ((x1 + x2) / 2.0, y2)


def bbox_separation(bbox_a: Sequence[float], bbox_b: Sequence[float]) -> float:
    """
    Pixel gap between two boxes (0 if they overlap or touch).

    Center-to-center distance overstates how far a worker is from a
    large excavator: the machine's centroid can sit hundreds of pixels
    inside a huge box while the person is already at the tracks.
    Edge-to-edge gap is the conservative proximity measure.
    """

    ax1, ay1, ax2, ay2 = normalize_bbox(bbox_a)
    bx1, by1, bx2, by2 = normalize_bbox(bbox_b)
    dx = max(0.0, ax1 - bx2, bx1 - ax2)
    dy = max(0.0, ay1 - by2, by1 - ay2)
    return float(np.hypot(dx, dy))


def person_ground_footprint(
    bbox: Sequence[float], height_fraction: float = 0.18
) -> Optional["Polygon"]:
    """
    Rectangle covering the bottom of a person box (feet / lower legs).

    A single bottom-center pixel misses workers who straddle a zone
    edge. Intersecting this footprint with the zone is stricter for
    safety without treating a raised arm as "inside".
    """

    if Polygon is None:
        return None

    x1, y1, x2, y2 = normalize_bbox(bbox)
    height = max(1.0, y2 - y1)
    top = y2 - (height * height_fraction)
    try:
        from shapely.geometry import box as shapely_box

        return shapely_box(x1, top, x2, y2)
    except Exception:
        return None


# ============================================================
# DETECTION HELPERS
# ============================================================

def get_detection_bbox(detection: Any) -> Optional[Sequence[float]]:
    """Extract bbox from list/tuple/ndarray or dictionary."""

    if isinstance(detection, dict):
        bbox = detection.get("bbox")
        if bbox is None:
            bbox = detection.get("box")
        return bbox

    if isinstance(detection, (list, tuple, np.ndarray)):
        if len(detection) >= 4:
            return detection[:4]

    return None


def get_detection_class_id(detection: Any) -> Optional[int]:
    """Extract class ID from a detection."""

    if isinstance(detection, dict):
        class_id = detection.get("class_id")
        if class_id is None:
            class_id = detection.get("class")
        if class_id is None:
            class_id = detection.get("cls")
        if class_id is None:
            return None
        try:
            return int(class_id)
        except (TypeError, ValueError):
            return None

    if isinstance(detection, (list, tuple, np.ndarray)):
        if len(detection) >= 6:
            try:
                return int(detection[5])
            except (TypeError, ValueError):
                return None

    return None


def get_detection_confidence(detection: Any) -> Optional[float]:
    """Extract confidence score."""

    if isinstance(detection, dict):
        confidence = detection.get("confidence")
        if confidence is None:
            confidence = detection.get("score")
        if confidence is None:
            return None
        try:
            return float(confidence)
        except (TypeError, ValueError):
            return None

    if isinstance(detection, (list, tuple, np.ndarray)):
        if len(detection) >= 5:
            try:
                return float(detection[4])
            except (TypeError, ValueError):
                return None

    return None


def get_track_id(detection: Any) -> Optional[int]:
    """Extract tracking ID when available."""

    if not isinstance(detection, dict):
        return None

    track_id = detection.get("track_id")
    if track_id is None:
        track_id = detection.get("id")
    if track_id is None:
        return None

    try:
        return int(track_id)
    except (TypeError, ValueError):
        return None


# ============================================================
# SAFETY CONES
# ============================================================

def extract_cone_points(detections: Sequence[Any]) -> List[Tuple[float, float]]:
    """Extract bottom-center points of safety cones."""

    points: List[Tuple[float, float]] = []

    for detection in detections:
        if get_detection_class_id(detection) != SAFETY_CONE_CLASS_ID:
            continue

        bbox = get_detection_bbox(detection)
        if bbox is None:
            continue

        try:
            points.append(get_bottom_center(bbox))
        except (TypeError, ValueError) as exc:
            logger.debug("Skipping malformed cone bbox %s: %s", bbox, exc)
            continue

    return points


# ============================================================
# HDBSCAN
# ============================================================

def cluster_cones(
    cone_points: Sequence[Tuple[float, float]],
    min_cluster_size: int = DEFAULT_MIN_CONES,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> List[List[Tuple[float, float]]]:
    """
    Cluster cone positions using HDBSCAN.

    At least `min_cluster_size` cones are required to create a polygon.
    min_samples=1 is intentional because real construction scenes may
    contain relatively few visible cones.
    """

    if len(cone_points) < min_cluster_size:
        return []

    if HDBSCAN is None:
        raise ImportError(
            "scikit-learn HDBSCAN is required. Install with: pip install scikit-learn"
        )

    points = np.asarray(cone_points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("cone_points must contain (x, y) pairs.")

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
        allow_single_cluster=True,
        copy=True,
    )

    labels = clusterer.fit_predict(points)

    clusters: List[List[Tuple[float, float]]] = []

    for label in sorted(set(labels)):
        if label == -1:  # noise
            continue

        cluster = [
            (float(cone_points[i][0]), float(cone_points[i][1]))
            for i, cluster_label in enumerate(labels)
            if cluster_label == label
        ]

        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)

    return clusters


# ============================================================
# POLYGON CREATION
# ============================================================

def _as_polygon(geom, buffer_px: float) -> Optional["Polygon"]:
    """Normalize hull/buffer output to a single valid Polygon."""

    if geom is None or geom.is_empty:
        return None

    if geom.geom_type == "Polygon":
        polygon = geom
    elif geom.geom_type == "MultiPolygon":
        polygon = max(geom.geoms, key=lambda part: part.area)
    elif buffer_px > 0 and geom.geom_type in ("LineString", "Point", "MultiLineString"):
        polygon = geom.buffer(max(buffer_px, 8.0))
        return _as_polygon(polygon, buffer_px=0.0)
    else:
        return None

    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    if polygon.is_empty or polygon.geom_type != "Polygon":
        return None
    return polygon


def create_polygon_from_points(
    points: Sequence[Tuple[float, float]],
    buffer_px: float = DEFAULT_ZONE_BUFFER_PX,
    min_area_px: float = DEFAULT_ZONE_MIN_AREA_PX,
) -> Optional["Polygon"]:
    """Create a dangerous-zone polygon using the convex hull of cone positions."""

    if MultiPoint is None or Polygon is None:
        raise ImportError("Shapely is required. Install with: pip install shapely")

    if len(points) < 3:
        return None

    try:
        hull = MultiPoint(points).convex_hull
        polygon = _as_polygon(hull, buffer_px=buffer_px)

        if polygon is None:
            return None

        if buffer_px > 0:
            polygon = _as_polygon(polygon.buffer(buffer_px), buffer_px=0.0)
            if polygon is None:
                return None

        if polygon.area < min_area_px:
            return None

        return polygon

    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to build polygon from points: %s", exc)
        return None


# ============================================================
# DANGER ZONES
# ============================================================

def build_danger_zones(
    detections: Sequence[Any],
    min_cluster_size: int = DEFAULT_MIN_CONES,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    buffer_px: float = DEFAULT_ZONE_BUFFER_PX,
    min_area_px: float = DEFAULT_ZONE_MIN_AREA_PX,
) -> List["Polygon"]:
    """
    Build dangerous zones from YOLO cone detections.

    Pipeline: YOLO -> cones -> bottom-center points -> HDBSCAN -> convex hull
              -> safety buffer -> min-area filter -> zones

    If HDBSCAN labels every cone as noise (common when a handful of
    cones sit in one loose group), fall back to a single hull of all
    cones rather than reporting no zone at all.
    """

    cone_points = extract_cone_points(detections)

    if len(cone_points) < min_cluster_size:
        return []

    clusters = cluster_cones(
        cone_points, min_cluster_size=min_cluster_size, min_samples=min_samples
    )

    if not clusters:
        clusters = [list(cone_points)]

    polygons: List["Polygon"] = []

    for cluster in clusters:
        polygon = create_polygon_from_points(
            cluster, buffer_px=buffer_px, min_area_px=min_area_px
        )
        if polygon is not None:
            polygons.append(polygon)

    return polygons


# ============================================================
# TEMPORAL ZONE STABILIZATION
# ============================================================

class DangerZoneTracker:
    """
    Gives danger zones a persistent identity across consecutive frames.

    Without this, `build_danger_zones()` re-runs HDBSCAN + convex-hull
    from scratch every frame, so a zone that visually looks the same to
    a human can silently change "zone_id" from one frame to the next
    (cones flicker in/out of detection, clustering is unstable at the
    boundary, etc). That breaks anything downstream that wants to say
    "worker has now been in zone 2 for 8 seconds" or wants to avoid
    spamming a new alert every frame.

    Strategy: match each new polygon to the most IoU-overlapping
    tracked zone from the previous frames. If no match is found within
    `iou_match_threshold`, a new persistent id is created. Zones that
    aren't seen for `max_missed_frames` are dropped (e.g. cones were
    picked up and moved).

    Usage:
        tracker = DangerZoneTracker()
        ...
        for frame in video:
            raw_zones = build_danger_zones(detections)
            stable_zones = tracker.update(raw_zones)
            # stable_zones: List[Dict] each with "zone_id" (persistent int),
            # "polygon", "age" (frames seen), "missed" (frames since last seen)
    """

    def __init__(
        self,
        iou_match_threshold: float = DEFAULT_ZONE_IOU_MATCH_THRESHOLD,
        max_missed_frames: int = DEFAULT_ZONE_MAX_MISSED_FRAMES,
    ):
        self.iou_match_threshold = iou_match_threshold
        self.max_missed_frames = max_missed_frames
        self._next_id = 1
        # zone_id -> {"polygon": Polygon, "age": int, "missed": int}
        self._tracked: Dict[int, Dict[str, Any]] = {}

    @staticmethod
    def _polygon_iou(poly_a: "Polygon", poly_b: "Polygon") -> float:
        try:
            if not poly_a.is_valid or not poly_b.is_valid:
                return 0.0

            intersection = poly_a.intersection(poly_b).area
            if intersection <= 0:
                return 0.0

            union = poly_a.union(poly_b).area
            if union <= 0:
                return 0.0

            return float(intersection / union)

        except Exception as exc:  # noqa: BLE001
            logger.debug("IoU computation failed: %s", exc)
            return 0.0

    def update(self, new_polygons: Sequence["Polygon"]) -> List[Dict[str, Any]]:
        """Match new polygons against tracked zones and return stabilized zones."""

        matched_tracked_ids: set = set()
        matched_new_indices: set = set()

        # Build every (tracked_id, new_index) IoU pair once, then greedily
        # assign the best matches first. This avoids the order-dependence
        # bugs you get from a naive nested-loop-with-break approach when
        # zones are close together (e.g. two adjacent cone clusters).
        candidate_pairs = []
        for tracked_id, tracked_zone in self._tracked.items():
            for new_index, new_polygon in enumerate(new_polygons):
                iou = self._polygon_iou(tracked_zone["polygon"], new_polygon)
                if iou >= self.iou_match_threshold:
                    candidate_pairs.append((iou, tracked_id, new_index))

        candidate_pairs.sort(key=lambda pair: pair[0], reverse=True)

        for iou, tracked_id, new_index in candidate_pairs:
            if tracked_id in matched_tracked_ids or new_index in matched_new_indices:
                continue

            zone = self._tracked[tracked_id]
            zone["polygon"] = new_polygons[new_index]
            zone["age"] += 1
            zone["missed"] = 0

            matched_tracked_ids.add(tracked_id)
            matched_new_indices.add(new_index)

        # Unmatched existing zones age toward removal.
        for tracked_id in list(self._tracked.keys()):
            if tracked_id in matched_tracked_ids:
                continue

            zone = self._tracked[tracked_id]
            zone["missed"] += 1
            if zone["missed"] > self.max_missed_frames:
                del self._tracked[tracked_id]

        # Unmatched new polygons become brand-new persistent zones.
        for new_index, new_polygon in enumerate(new_polygons):
            if new_index in matched_new_indices:
                continue

            self._tracked[self._next_id] = {
                "polygon": new_polygon,
                "age": 1,
                "missed": 0,
            }
            self._next_id += 1

        return [
            {
                "zone_id": tracked_id,
                "polygon": zone["polygon"],
                "age": zone["age"],
                "missed": zone["missed"],
            }
            for tracked_id, zone in self._tracked.items()
        ]


# ============================================================
# POINT / POLYGON
# ============================================================

def is_point_inside_polygon(point: Tuple[float, float], polygon: "Polygon") -> bool:
    """Check whether a point is inside or on the boundary of a polygon."""

    if Point is None:
        raise ImportError("Shapely is required. Install with: pip install shapely")

    if polygon is None:
        return False

    try:
        return bool(polygon.covers(Point(float(point[0]), float(point[1]))))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Point-in-polygon check failed: %s", exc)
        return False


# ============================================================
# PERSON / ZONE
# ============================================================

def is_person_inside_zone(detection: Any, polygon: "Polygon") -> bool:
    """
    Convenience single-item check: is this ONE person inside this ONE zone.

    NOTE: This does not filter by class_id for dict-based tracked persons
    that omit "class_id" (tracked persons are trusted to already be
    persons). Kept for ad-hoc / debugging use; `find_people_inside_zones`
    is what the engine actually uses for batch analysis.
    """

    if isinstance(detection, dict):
        class_id = detection.get("class_id")
        if class_id is not None:
            try:
                if int(class_id) != PERSON_CLASS_ID:
                    return False
            except (TypeError, ValueError):
                return False
    else:
        if get_detection_class_id(detection) != PERSON_CLASS_ID:
            return False

    bbox = get_detection_bbox(detection)
    if bbox is None:
        return False

    try:
        point = get_bottom_center(bbox)
        if is_point_inside_polygon(point, polygon):
            return True
        footprint = person_ground_footprint(bbox)
        if footprint is not None:
            return bool(polygon.intersects(footprint))
        return False
    except (TypeError, ValueError):
        return False


def find_people_inside_zones(
    persons: Sequence[Any],
    polygons: Sequence["Polygon"],
) -> List[Dict[str, Any]]:
    """
    Find tracked/detected persons inside dangerous zones.

    `persons` is trusted to already contain only person-type items
    (this is how safety_engine.py calls it - `persons` comes from your
    tracker, not raw mixed-class YOLO output).

    Returns a list of:
        {
            "person_index": 0,
            "track_id": 20,          # only present if resolvable
            "zone_index": 0,          # 0-based index into `polygons`
            "zone_id": 1,             # 1-based, human-friendly
            "point": (x, y),
            "confidence": 0.91        # only present if resolvable
        }
    """

    results: List[Dict[str, Any]] = []

    if not polygons:
        return results

    for person_index, person in enumerate(persons):
        bbox = get_detection_bbox(person)
        if bbox is None:
            continue

        try:
            point = get_bottom_center(bbox)
        except (TypeError, ValueError):
            continue

        track_id = get_track_id(person)
        confidence = get_detection_confidence(person)

        for zone_index, polygon in enumerate(polygons):
            inside = is_point_inside_polygon(point, polygon)
            if not inside:
                footprint = person_ground_footprint(bbox)
                inside = bool(
                    footprint is not None and polygon.intersects(footprint)
                )
            if not inside:
                continue

            result: Dict[str, Any] = {
                "person_index": person_index,
                "zone_index": zone_index,
                "zone_id": zone_index + 1,
                "point": point,
            }

            if track_id is not None:
                result["track_id"] = track_id

            if confidence is not None:
                result["confidence"] = confidence

            results.append(result)

    return results


# ============================================================
# UTILITY POLES
# ============================================================

def extract_utility_poles(detections: Sequence[Any]) -> List[Tuple[float, float]]:
    """Extract bottom-center points of utility poles."""

    points: List[Tuple[float, float]] = []

    for detection in detections:
        if get_detection_class_id(detection) != UTILITY_POLE_CLASS_ID:
            continue

        bbox = get_detection_bbox(detection)
        if bbox is None:
            continue

        try:
            points.append(get_bottom_center(bbox))
        except (TypeError, ValueError):
            continue

    return points


# ============================================================
# DISTANCE
# ============================================================

def calculate_pole_distance(
    point_a: Tuple[float, float], point_b: Tuple[float, float]
) -> float:
    """Euclidean distance in pixels."""

    dx = float(point_a[0]) - float(point_b[0])
    dy = float(point_a[1]) - float(point_b[1])
    return float(np.hypot(dx, dy))


# ============================================================
# MACHINERY / VEHICLES -> UTILITY POLES
# ============================================================

def find_machinery_near_poles(
    detections: Sequence[Any],
    threshold_pixels: float = DEFAULT_POLE_DISTANCE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Find machinery or vehicles that are too close to utility poles.

    Distance is measured between bottom-center points. Returns:
        {
            "object_index": int,       # index of machinery/vehicle in `detections`
            "machinery_index": int,    # alias of object_index (naming used by callers)
            "pole_index": int,
            "object_class_id": int,
            "distance_px": float,      # canonical field name
            "distance": float,         # deprecated alias, kept for compatibility
            "object_confidence": Optional[float],
        }
    """

    objects: List[Tuple[int, Any, Sequence[float]]] = []
    poles: List[Tuple[int, Sequence[float]]] = []

    for index, detection in enumerate(detections):
        class_id = get_detection_class_id(detection)
        bbox = get_detection_bbox(detection)

        if bbox is None or class_id not in (MACHINERY_CLASS_ID, VEHICLE_CLASS_ID):
            continue

        objects.append((index, detection, bbox))

    for index, detection in enumerate(detections):
        if get_detection_class_id(detection) != UTILITY_POLE_CLASS_ID:
            continue

        bbox = get_detection_bbox(detection)
        if bbox is None:
            continue

        poles.append((index, bbox))

    warnings: List[Dict[str, Any]] = []

    for object_index, object_detection, object_bbox in objects:
        object_class = get_detection_class_id(object_detection)

        for pole_index, pole_bbox in poles:
            try:
                distance = bbox_separation(object_bbox, pole_bbox)
            except (TypeError, ValueError):
                continue

            if distance > threshold_pixels:
                continue

            warnings.append(
                {
                    "object_index": object_index,
                    "machinery_index": object_index,
                    "pole_index": pole_index,
                    "object_class_id": object_class,
                    "distance_px": distance,
                    "distance": distance,  # deprecated alias
                    "object_confidence": get_detection_confidence(object_detection),
                }
            )

    return warnings


# ============================================================
# SUMMARY
# ============================================================

def geometry_summary(detections: Sequence[Any]) -> Dict[str, Any]:
    """Return a compact geometry summary."""

    counts = {
        "persons": 0,
        "cones": 0,
        "machinery": 0,
        "utility_poles": 0,
        "vehicles": 0,
    }

    for detection in detections:
        class_id = get_detection_class_id(detection)

        if class_id == PERSON_CLASS_ID:
            counts["persons"] += 1
        elif class_id == SAFETY_CONE_CLASS_ID:
            counts["cones"] += 1
        elif class_id == MACHINERY_CLASS_ID:
            counts["machinery"] += 1
        elif class_id == UTILITY_POLE_CLASS_ID:
            counts["utility_poles"] += 1
        elif class_id == VEHICLE_CLASS_ID:
            counts["vehicles"] += 1

    polygons = build_danger_zones(detections)

    return {**counts, "danger_zones": len(polygons)}
