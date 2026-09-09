"""
Fire/smoke confirmation gate.

Fire and smoke share a model but they are NOT the same problem:
    - Fire boxes are usually compact, high-contrast, and should stay
      strict so glare/tools do not go CRITICAL.
    - Smoke is diffuse, often lower-confidence, changes shape frame to
      frame, and is easy to kill with fire-oriented area/streak cuts.

Do not "fix" false Person detections by raising this gate, and do not
"fix" missed smoke by lowering Person confidence. Those knobs live in
src/hazard/detection_filter.py and src/hazard/track_confirmation.py.

A detection becomes confirmed only after:
    1. confidence >= type-specific min_confidence
    2. bounding-box area >= type-specific min_area_px
    3. the SAME region (matched by bbox IoU) persists for
       type-specific min_consecutive_frames

Usage (one tracker instance PER video/stream):

    fire_tracker = FireConfirmationTracker()
    for frame in video:
        raw = fire_detector.predict(frame)
        confirmed = fire_tracker.update(raw)
        # fire_tracker.last_stats explains drops this frame
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

# Fire (strict) — glare / tools / tiny sparks
DEFAULT_MIN_CONFIDENCE = 0.45
DEFAULT_MIN_AREA_PX = 800.0
DEFAULT_MIN_CONSECUTIVE_FRAMES = 5

# Smoke (looser) — plume can be faint, small, and morph between frames
DEFAULT_SMOKE_MIN_CONFIDENCE = 0.20
DEFAULT_SMOKE_MIN_AREA_PX = 200.0
DEFAULT_SMOKE_MIN_CONSECUTIVE_FRAMES = 2
DEFAULT_SMOKE_IOU_MATCH_THRESHOLD = 0.10

DEFAULT_IOU_MATCH_THRESHOLD = 0.2
DEFAULT_MAX_MISSED_FRAMES = 3

EMPTY_STATS: Dict[str, Any] = {
    "raw": 0,
    "raw_fire": 0,
    "raw_smoke": 0,
    "dropped_confidence": 0,
    "dropped_area": 0,
    "dropped_persistence": 0,
    "confirmed": 0,
    "confirmed_fire": 0,
    "confirmed_smoke": 0,
    "max_raw_smoke_conf": 0.0,
    "max_raw_smoke_area": 0.0,
    "max_raw_fire_conf": 0.0,
    "max_raw_fire_area": 0.0,
}


def _bbox_area(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = bbox[:4]
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def _bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    return float(intersection / union) if union > 0 else 0.0


def detection_kind(detection: Dict[str, Any]) -> str:
    label = (detection.get("type") or detection.get("class_name") or "").lower()
    if "smoke" in label:
        return "smoke"
    if "fire" in label:
        return "fire"
    return "other"


class FireConfirmationTracker:
    """Debounces fire and smoke independently before they count as real."""

    def __init__(
        self,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        min_area_px: float = DEFAULT_MIN_AREA_PX,
        min_consecutive_frames: int = DEFAULT_MIN_CONSECUTIVE_FRAMES,
        iou_match_threshold: float = DEFAULT_IOU_MATCH_THRESHOLD,
        max_missed_frames: int = DEFAULT_MAX_MISSED_FRAMES,
        smoke_min_confidence: float = DEFAULT_SMOKE_MIN_CONFIDENCE,
        smoke_min_area_px: float = DEFAULT_SMOKE_MIN_AREA_PX,
        smoke_min_consecutive_frames: int = DEFAULT_SMOKE_MIN_CONSECUTIVE_FRAMES,
        smoke_iou_match_threshold: float = DEFAULT_SMOKE_IOU_MATCH_THRESHOLD,
    ):
        self.min_confidence = min_confidence
        self.min_area_px = min_area_px
        self.min_consecutive_frames = min_consecutive_frames
        self.iou_match_threshold = iou_match_threshold
        self.max_missed_frames = max_missed_frames
        self.smoke_min_confidence = smoke_min_confidence
        self.smoke_min_area_px = smoke_min_area_px
        self.smoke_min_consecutive_frames = smoke_min_consecutive_frames
        self.smoke_iou_match_threshold = smoke_iou_match_threshold

        self._next_id = 1
        # track_id -> {"bbox", "detection", "streak", "missed", "kind"}
        self._tracked: Dict[int, Dict[str, Any]] = {}
        self.last_stats: Dict[str, Any] = dict(EMPTY_STATS)

    def _thresholds(self, kind: str) -> Tuple[float, float, int, float]:
        if kind == "smoke":
            return (
                self.smoke_min_confidence,
                self.smoke_min_area_px,
                self.smoke_min_consecutive_frames,
                self.smoke_iou_match_threshold,
            )
        return (
            self.min_confidence,
            self.min_area_px,
            self.min_consecutive_frames,
            self.iou_match_threshold,
        )

    def update(self, fire_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Feed this frame's raw fire/smoke detections in, get back only
        the ones that have passed type-specific confidence + area +
        persistence.

        Returned dicts are copies of the original detection dicts with
        two extra debug fields: "confirmed_streak" and "fire_track_id".
        Per-frame drop reasons are in self.last_stats.
        """

        stats = dict(EMPTY_STATS)
        stats["raw"] = len(fire_detections)

        candidates: List[Dict[str, Any]] = []
        for detection in fire_detections:
            kind = detection_kind(detection)
            if kind == "fire":
                stats["raw_fire"] += 1
            elif kind == "smoke":
                stats["raw_smoke"] += 1

            bbox = detection.get("bbox")
            confidence = float(detection.get("confidence", 0.0) or 0.0)
            area = _bbox_area(bbox) if bbox is not None else 0.0

            if kind == "smoke":
                stats["max_raw_smoke_conf"] = max(stats["max_raw_smoke_conf"], confidence)
                stats["max_raw_smoke_area"] = max(stats["max_raw_smoke_area"], area)
            elif kind == "fire":
                stats["max_raw_fire_conf"] = max(stats["max_raw_fire_conf"], confidence)
                stats["max_raw_fire_area"] = max(stats["max_raw_fire_area"], area)

            if bbox is None:
                stats["dropped_area"] += 1
                continue

            min_conf, min_area, _, _ = self._thresholds(kind)
            if confidence < min_conf:
                stats["dropped_confidence"] += 1
                continue
            if area < min_area:
                stats["dropped_area"] += 1
                continue

            tagged = dict(detection)
            tagged["_kind"] = kind
            candidates.append(tagged)

        matched_tracked_ids: set = set()
        matched_candidate_indices: set = set()

        candidate_pairs: List[Tuple[float, int, int]] = []
        for track_id, tracked in self._tracked.items():
            _, _, _, iou_thr = self._thresholds(tracked["kind"])
            for index, candidate in enumerate(candidates):
                if candidate["_kind"] != tracked["kind"]:
                    continue
                iou = _bbox_iou(tracked["bbox"], candidate["bbox"])
                if iou >= iou_thr:
                    candidate_pairs.append((iou, track_id, index))

        candidate_pairs.sort(key=lambda pair: pair[0], reverse=True)

        for _, track_id, index in candidate_pairs:
            if track_id in matched_tracked_ids or index in matched_candidate_indices:
                continue

            tracked = self._tracked[track_id]
            tracked["bbox"] = candidates[index]["bbox"]
            tracked["detection"] = candidates[index]
            tracked["streak"] += 1
            tracked["missed"] = 0

            matched_tracked_ids.add(track_id)
            matched_candidate_indices.add(index)

        for track_id in list(self._tracked.keys()):
            if track_id in matched_tracked_ids:
                continue
            tracked = self._tracked[track_id]
            tracked["missed"] += 1
            if tracked["missed"] > self.max_missed_frames:
                del self._tracked[track_id]

        for index, candidate in enumerate(candidates):
            if index in matched_candidate_indices:
                continue
            self._tracked[self._next_id] = {
                "bbox": candidate["bbox"],
                "detection": candidate,
                "streak": 1,
                "missed": 0,
                "kind": candidate["_kind"],
            }
            self._next_id += 1

        confirmed = []
        for track_id, tracked in self._tracked.items():
            _, _, min_frames, _ = self._thresholds(tracked["kind"])
            if tracked["streak"] >= min_frames:
                detection = dict(tracked["detection"])
                detection.pop("_kind", None)
                detection["confirmed_streak"] = tracked["streak"]
                detection["fire_track_id"] = track_id
                confirmed.append(detection)
                stats["confirmed"] += 1
                if tracked["kind"] == "smoke":
                    stats["confirmed_smoke"] += 1
                elif tracked["kind"] == "fire":
                    stats["confirmed_fire"] += 1

        stats["dropped_persistence"] = max(0, len(candidates) - stats["confirmed"])
        self.last_stats = stats
        return confirmed
