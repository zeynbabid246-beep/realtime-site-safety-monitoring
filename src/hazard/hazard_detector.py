"""
Hazard model wrapper.

Runs the unified hazard model (Hardhat, Mask, NO-Hardhat, NO-Mask,
NO-Safety Vest, Person, Safety Cone, Safety Vest, machinery,
utility pole, vehicle) with ByteTrack tracking enabled, and normalizes
raw ultralytics output into the plain dict format that
src/safety/geometry.py, rules.py, and safety_engine.py all expect:

    {
        "class_id": int,
        "class_name": str,
        "confidence": float,
        "bbox": [x1, y1, x2, y2],
        "track_id": Optional[int],   # populated by ByteTrack when available
    }

Why tracking is requested for the WHOLE frame (not just the Person
class): ultralytics assigns track IDs per-call across whatever classes
are detected. Restricting `classes=[...]` at the tracker level is fine,
but we actually want a single .track() call per frame for performance
(one model forward pass) and then filter by class afterwards in
Python - see extract_persons() / extract_machines() below.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None


class HazardDetector:
    # Must stay in sync with src/safety/rules.py and src/safety/geometry.py.
    HARDHAT = 0
    MASK = 1
    NO_HARDHAT = 2
    NO_MASK = 3
    NO_SAFETY_VEST = 4
    PERSON_CLASS = 5
    CONE_CLASS = 6
    SAFETY_VEST = 7
    MACHINERY_CLASS = 8
    UTILITY_POLE_CLASS = 9
    VEHICLE_CLASS = 10

    def __init__(
        self,
        model_path: Union[str, Path],
        confidence: float = 0.25,
        image_size: int = 640,
        tracker: str = "bytetrack.yaml",
        device: Optional[str] = None,
    ):
        if YOLO is None:
            raise ImportError("ultralytics is required. Install with: pip install ultralytics")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Hazard model not found: {model_path}")

        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.image_size = image_size
        self.tracker = tracker
        self.device = device
        self.class_names: Dict[int, str] = self.model.names

    # --------------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------------

    def track(self, frame: np.ndarray, persist: bool = True) -> List[Dict[str, Any]]:
        """
        Run tracked inference on a single frame.

        Falls back to plain (untracked) detection if the tracker
        throws - this happens occasionally with malformed frames or
        tracker re-init edge cases, and a single bad frame should not
        take down a whole video/stream.
        """

        try:
            results = self.model.track(
                source=frame,
                persist=persist,
                tracker=self.tracker,
                conf=self.confidence,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tracking failed on frame, falling back to detection: %s", exc)
            return self.predict(frame)

        return self._parse_result(results[0] if results else None)

    def predict(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run plain (untracked) detection on a single frame."""

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        return self._parse_result(results[0] if results else None)

    def _parse_result(self, result) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []

        if result is None or result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes
        has_ids = getattr(boxes, "id", None) is not None
        track_ids = boxes.id.int().cpu().tolist() if has_ids else [None] * len(boxes)

        for box, track_id in zip(boxes, track_ids):
            try:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping malformed detection box: %s", exc)
                continue

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": self.class_names.get(class_id, str(class_id)),
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "track_id": int(track_id) if track_id is not None else None,
                }
            )

        return detections

    # --------------------------------------------------------------
    # CONVENIENCE EXTRACTORS
    # --------------------------------------------------------------
    # These translate the raw normalized detections into the specific
    # shapes safety_engine.py / rules.py / distance_calculator.py expect.

    def extract_persons(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Persons for the safety engine.

        track_id may be None (e.g. calling predict() instead of track(),
        such as for a single standalone image). Downstream code tolerates
        a missing track_id, it just means violations for that person
        won't carry a stable ID.
        """

        return [
            {
                "track_id": d.get("track_id"),
                "bbox": d["bbox"],
                "confidence": d["confidence"],
            }
            for d in detections
            if d["class_id"] == self.PERSON_CLASS
        ]

    def extract_machines(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Machinery + vehicles for the safety engine's distance calculator."""

        return [
            {
                "class_id": d["class_id"],
                "class_name": d["class_name"],
                "confidence": d["confidence"],
                "bbox": d["bbox"],
            }
            for d in detections
            if d["class_id"] in (self.MACHINERY_CLASS, self.VEHICLE_CLASS)
        ]
