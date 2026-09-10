"""
Track confirmation gate.

ByteTrack will happily assign a persistent track_id to a single
detection that only appears for one frame - a tiny false positive
("Person 0.37" on a thin pole, ID:15 in your screenshot) gets an ID
exactly like a real worker does. This module requires a track_id to be
seen for `min_hits` frames (tolerating brief gaps up to
`max_missed_frames`) before that track is "confirmed" and allowed to
reach extract_persons()/extract_machines() and everything downstream.

Same debounce pattern as DangerZoneTracker (src/safety/geometry.py)
and FireConfirmationTracker (src/fire/fire_confirmation.py): one
instance PER video/stream, never shared across unrelated streams.

Applies generically to ANY tracked class (Person, machinery, vehicle,
etc.) since ByteTrack assigns track_ids across all classes in one
shared id space - a flickering "machinery" false positive gets the
same treatment as a flickering "Person" one.

TWO static-object suppressors (both temporal, both balanced toward
keeping real hazards - see the FP-vs-FN notes below):

  A. Static PERSON suppressor. A pole / tripod / ground stake that the
     model mislabels as "Person" produces a track whose bottom-center
     never moves. We only drop it when ALL of these hold, so a real
     (even momentarily still) worker is very unlikely to be removed:
       - it has been tracked for >= static_min_frames frames
         (default 45 ~ 1.5s at 30fps - long enough that a worker who
         is merely pausing is not penalised), AND
       - its bottom-center moved < static_motion_threshold_px across
         that whole window (normalized to a 640p baseline), AND
       - its AVERAGE confidence is below static_max_avg_confidence
         (default 0.55). Mislabelled poles/debris sit in the
         low-confidence band; a clearly visible worker - even a distant
         or partially occluded one - usually clears it.

  B. Static MACHINERY/VEHICLE suppressor (the "house detected as
     machinery" mitigation). A background building/wall the model calls
     "machinery" is detected at HIGH confidence (0.68-0.82), so neither
     a confidence gate nor a size gate in detection_filter.py can remove
     it. What DOES separate it from real plant is that a building is
     pixel-perfectly static for the entire clip. We therefore drop a
     machinery/vehicle track only when it has been essentially immobile
     (bottom-center displacement < static_machinery_motion_threshold_px,
     normalized to 640p) for >= static_machinery_min_frames frames
     (default 90 ~ 3s at 30fps).

     FP-vs-FN TRADEOFF (this is a MITIGATION, not a fix): a genuinely
     parked excavator or a truck stopped for several seconds will also
     be suppressed once it crosses the window. That is the accepted cost
     of removing building false positives by code alone. The window is
     deliberately long so briefly-paused plant is kept. The ONLY real
     fix for "house -> machinery" is hard-negative fine-tuning: collect
     frames where it happens, label the building with NO machinery box,
     and retrain (see docs/DETECTION_LIMITATIONS.md). Turn this off with
     filter_static_machinery=False if your site has long-stationary
     plant you must keep alerting on.

Detections with track_id=None (e.g. from .predict() on a single
standalone image with no tracking) pass through UNCHANGED - there's no
persistent id to confirm against, so this gate only meaningfully
applies to tracked video/stream frames.

Usage:
    track_confirmation = TrackConfirmationTracker(min_hits=3)

    for frame in video:
        detections = hazard_detector.track(frame)
        confirmed = track_confirmation.update(detections, frame_shape=frame.shape[:2])
        persons = hazard_detector.extract_persons(confirmed)
        ...
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional

# --- core debounce -------------------------------------------------
DEFAULT_MIN_HITS = 3
DEFAULT_MAX_MISSED_FRAMES = 5

# --- static PERSON suppressor (pole / tripod / stake as Person) -----
DEFAULT_STATIC_MIN_FRAMES = 45
DEFAULT_STATIC_MOTION_THRESHOLD_PX = 3.0   # normalized to 640p
DEFAULT_STATIC_MAX_AVG_CONFIDENCE = 0.55

# --- static MACHINERY/VEHICLE suppressor (building as machinery) ----
DEFAULT_STATIC_MACHINERY_MIN_FRAMES = 90   # ~3s at 30fps
DEFAULT_STATIC_MACHINERY_MOTION_THRESHOLD_PX = 2.5  # normalized to 640p

PERSON_CLASS_NAME = "Person"
MACHINERY_CLASS_NAMES = frozenset({"machinery", "vehicle"})


class TrackConfirmationTracker:
    def __init__(
        self,
        min_hits: int = DEFAULT_MIN_HITS,
        max_missed_frames: int = DEFAULT_MAX_MISSED_FRAMES,
        filter_static_persons: bool = True,
        static_min_frames: int = DEFAULT_STATIC_MIN_FRAMES,
        static_motion_threshold_px: float = DEFAULT_STATIC_MOTION_THRESHOLD_PX,
        static_max_avg_confidence: float = DEFAULT_STATIC_MAX_AVG_CONFIDENCE,
        filter_static_machinery: bool = True,
        static_machinery_min_frames: int = DEFAULT_STATIC_MACHINERY_MIN_FRAMES,
        static_machinery_motion_threshold_px: float = DEFAULT_STATIC_MACHINERY_MOTION_THRESHOLD_PX,
    ):
        self.min_hits = min_hits
        self.max_missed_frames = max_missed_frames

        self.filter_static_persons = filter_static_persons
        self.static_min_frames = static_min_frames
        self.static_motion_threshold_px = static_motion_threshold_px
        self.static_max_avg_confidence = static_max_avg_confidence

        self.filter_static_machinery = filter_static_machinery
        self.static_machinery_min_frames = static_machinery_min_frames
        self.static_machinery_motion_threshold_px = static_machinery_motion_threshold_px

        # Position history must be long enough for the largest static window.
        history_len = max(static_min_frames, static_machinery_min_frames) + 10

        # track_id -> {"hits", "missed", "positions": deque, "confs": deque, "class"}
        self._tracks: Dict[int, Dict[str, Any]] = {}
        self._history_len = history_len

    # --------------------------------------------------------------
    # STATIC-OBJECT DETECTION
    # --------------------------------------------------------------

    def _is_static(
        self,
        record: Dict[str, Any],
        min_frames: int,
        motion_threshold_px: float,
        frame_shape: Any,
        max_avg_confidence: Optional[float] = None,
    ) -> bool:
        """
        True if this track's bottom-center has barely moved across a long
        window (and, when max_avg_confidence is given, its average
        confidence is below that bar). Displacement is normalized to a
        640p baseline so the same threshold works at any resolution.
        """

        positions = record["positions"]
        if len(positions) < min_frames:
            return False

        p0 = positions[0]
        max_disp = max(math.hypot(p[0] - p0[0], p[1] - p0[1]) for p in positions)

        if frame_shape is not None and frame_shape[0] > 0 and frame_shape[1] > 0:
            scale = max(frame_shape[0], frame_shape[1]) / 640.0
            disp_norm = max_disp / scale if scale > 0 else max_disp
        else:
            disp_norm = max_disp

        if disp_norm >= motion_threshold_px:
            return False

        if max_avg_confidence is not None:
            confs = record["confs"]
            if not confs:
                return False
            avg_conf = sum(confs) / len(confs)
            if avg_conf >= max_avg_confidence:
                return False

        return True

    # --------------------------------------------------------------
    # MAIN UPDATE
    # --------------------------------------------------------------

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_shape: Any = None,
    ) -> List[Dict[str, Any]]:
        """
        Feed this frame's detections in, get back only the ones that are
        either untracked (pass through as-is) or whose track_id has
        accumulated enough hits and is not an immobile static object.
        """

        seen_track_ids = set()

        for detection in detections:
            track_id = detection.get("track_id")
            if track_id is None:
                continue

            seen_track_ids.add(track_id)
            record = self._tracks.get(track_id)
            if record is None:
                record = {
                    "hits": 0,
                    "missed": 0,
                    "positions": deque(maxlen=self._history_len),
                    "confs": deque(maxlen=self._history_len),
                    "class": detection.get("class_name", ""),
                }
                self._tracks[track_id] = record

            record["hits"] += 1
            record["missed"] = 0
            record["class"] = detection.get("class_name", record["class"])

            bbox = detection.get("bbox")
            if bbox is not None:
                cx = (float(bbox[0]) + float(bbox[2])) / 2.0
                cy = float(bbox[3])  # bottom center
                record["positions"].append((cx, cy))

            record["confs"].append(detection.get("confidence", 0.0))

        # Age out tracks not seen this frame.
        for track_id in list(self._tracks.keys()):
            if track_id in seen_track_ids:
                continue
            self._tracks[track_id]["missed"] += 1
            if self._tracks[track_id]["missed"] > self.max_missed_frames:
                del self._tracks[track_id]

        confirmed: List[Dict[str, Any]] = []
        for detection in detections:
            track_id = detection.get("track_id")

            if track_id is None:
                confirmed.append(detection)
                continue

            record = self._tracks.get(track_id)
            if record is None or record["hits"] < self.min_hits:
                continue

            class_name = record["class"]

            if (
                self.filter_static_persons
                and class_name == PERSON_CLASS_NAME
                and self._is_static(
                    record,
                    self.static_min_frames,
                    self.static_motion_threshold_px,
                    frame_shape,
                    max_avg_confidence=self.static_max_avg_confidence,
                )
            ):
                continue

            if (
                self.filter_static_machinery
                and class_name in MACHINERY_CLASS_NAMES
                and self._is_static(
                    record,
                    self.static_machinery_min_frames,
                    self.static_machinery_motion_threshold_px,
                    frame_shape,
                )
            ):
                continue

            confirmed.append(detection)

        return confirmed

    def hits_for(self, track_id: int) -> int:
        """Debug helper: how many hits a given track_id currently has."""

        record = self._tracks.get(track_id)
        return record["hits"] if record else 0
