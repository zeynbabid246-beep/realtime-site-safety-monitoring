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

Usage:
    track_confirmation = TrackConfirmationTracker(min_hits=3)

    for frame in video:
        detections = hazard_detector.track(frame)
        confirmed = track_confirmation.update(detections)
        persons = hazard_detector.extract_persons(confirmed)
        ...

Detections with track_id=None (e.g. from .predict() on a single
standalone image with no tracking) pass through UNCHANGED - there's no
persistent id to confirm against, so this gate only meaningfully
applies to tracked video/stream frames.
"""

from __future__ import annotations

from typing import Any, Dict, List

from collections import deque
import math

DEFAULT_MIN_HITS = 3
DEFAULT_MAX_MISSED_FRAMES = 5
DEFAULT_STATIC_MIN_FRAMES = 20
DEFAULT_STATIC_MOTION_THRESHOLD_PX = 4.0  # normalized to 640p


class TrackConfirmationTracker:
    def __init__(
        self,
        min_hits: int = DEFAULT_MIN_HITS,
        max_missed_frames: int = DEFAULT_MAX_MISSED_FRAMES,
        filter_static_persons: bool = True,
        static_min_frames: int = DEFAULT_STATIC_MIN_FRAMES,
        static_motion_threshold_px: float = DEFAULT_STATIC_MOTION_THRESHOLD_PX,
    ):
        self.min_hits = min_hits
        self.max_missed_frames = max_missed_frames
        self.filter_static_persons = filter_static_persons
        self.static_min_frames = static_min_frames
        self.static_motion_threshold_px = static_motion_threshold_px

        # track_id -> {"hits": int, "missed": int, "positions": deque, "confs": deque, "class": str}
        self._tracks: Dict[int, Dict[str, Any]] = {}

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_shape: Any = None,
    ) -> List[Dict[str, Any]]:
        """
        Feed this frame's detections in, get back only the ones that
        are either untracked (pass through as-is) or whose track_id
        has accumulated enough hits and is not an immobile static pole.
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
                    "positions": deque(maxlen=self.static_min_frames + 10),
                    "confs": deque(maxlen=self.static_min_frames + 10),
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

            conf = detection.get("confidence", 0.0)
            record["confs"].append(conf)

        # Age out tracks not seen this frame
        for track_id in list(self._tracks.keys()):
            if track_id in seen_track_ids:
                continue
            self._tracks[track_id]["missed"] += 1
            if self._tracks[track_id]["missed"] > self.max_missed_frames:
                del self._tracks[track_id]

        confirmed = []
        for detection in detections:
            track_id = detection.get("track_id")

            if track_id is None:
                confirmed.append(detection)
                continue

            record = self._tracks.get(track_id)
            if record is None or record["hits"] < self.min_hits:
                continue

            # Check if this is an immobile static pole misclassified as Person
            if self.filter_static_persons and record["class"] == "Person":
                positions = record["positions"]
                if len(positions) >= self.static_min_frames:
                    p0 = positions[0]
                    max_disp = max(
                        math.hypot(p[0] - p0[0], p[1] - p0[1]) for p in positions
                    )
                    # Normalize displacement to 640p baseline if frame_shape is known
                    if frame_shape is not None and frame_shape[0] > 0 and frame_shape[1] > 0:
                        scale = max(frame_shape) / 640.0
                        disp_norm = max_disp / scale
                    else:
                        disp_norm = max_disp

                    avg_conf = sum(record["confs"]) / len(record["confs"])

                    # If an object hasn't moved at all (<3.5px on 640p scale) across 20+ frames
                    # and its confidence is moderate (<0.68), it is a static pole/tripod/debris
                    if disp_norm < self.static_motion_threshold_px and avg_conf < 0.68:
                        continue

            confirmed.append(detection)

        return confirmed

    def hits_for(self, track_id: int) -> int:
        """Debug helper: how many hits a given track_id currently has."""

        record = self._tracks.get(track_id)
        return record["hits"] if record else 0