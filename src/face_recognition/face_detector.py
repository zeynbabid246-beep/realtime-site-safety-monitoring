"""
FaceDetector - turn YOLO Person boxes into face crops.

No extra face-detection model is needed: the hazard model already
localises every Person, and the trained Siamese model was fed 250x250
frontal webcam head-crops, so the upper head region of the person box is
exactly the distribution it expects.

For each person detection we take the TOP band of the person bbox
(head_region_fraction of its height), narrowed horizontally by
face_width_fraction, clamped to the frame. Crops smaller than
min_face_size_px are dropped - tiny/distant faces are below the
resolution the model was trained on and would only add noise.

FaceDetector is stateless; it is shared like HazardDetector.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from src.face_recognition.preprocessing import crop_with_margin

logger = logging.getLogger(__name__)


class FaceCandidate:
    """A face crop plus where it came from."""

    def __init__(
        self,
        person_index: int,
        track_id: Optional[int],
        bbox,
        crop: np.ndarray,
    ):
        self.person_index = person_index
        self.track_id = track_id
        self.bbox = bbox
        self.crop = crop

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"FaceCandidate(track_id={self.track_id}, bbox={self.bbox})"


class FaceDetector:
    """Extract face crops from person detections in a frame."""

    def __init__(
        self,
        head_region_fraction: float = 0.35,
        face_width_fraction: float = 0.80,
        min_face_size_px: int = 40,
        min_person_confidence: float = 0.45,
    ):
        self.head_region_fraction = float(head_region_fraction)
        self.face_width_fraction = float(face_width_fraction)
        self.min_face_size_px = int(min_face_size_px)
        self.min_person_confidence = float(min_person_confidence)

    def extract_face_candidates(
        self,
        frame: np.ndarray,
        persons: List[Dict[str, Any]],
    ) -> List[FaceCandidate]:
        """
        Build one FaceCandidate per qualifying person box.

        persons: the list produced by HazardDetector.extract_persons()
                 (dicts with bbox / confidence / track_id).
        """

        candidates: List[FaceCandidate] = []

        if frame is None or getattr(frame, "ndim", 0) != 3:
            return candidates

        for index, person in enumerate(persons):
            bbox = person.get("bbox")
            if bbox is None or len(bbox) < 4:
                continue

            confidence = float(person.get("confidence", 0.0) or 0.0)
            if confidence < self.min_person_confidence:
                continue

            try:
                x1, y1, x2, y2 = (float(v) for v in bbox[:4])
            except (TypeError, ValueError):
                continue

            head_height = max((y2 - y1) * self.head_region_fraction, 1.0)
            head_bbox = [x1, y1, x2, y1 + head_height]

            crop = crop_with_margin(
                frame,
                head_bbox,
                width_fraction=self.face_width_fraction,
                min_size=self.min_face_size_px,
            )
            if crop is None:
                continue

            candidates.append(
                FaceCandidate(
                    person_index=index,
                    track_id=person.get("track_id"),
                    bbox=list(head_bbox),
                    crop=crop,
                )
            )

        return candidates

    def detect(self, frame: np.ndarray, persons: List[Dict[str, Any]]) -> List[FaceCandidate]:
        """Alias kept for readability at call sites."""
        return self.extract_face_candidates(frame, persons)
