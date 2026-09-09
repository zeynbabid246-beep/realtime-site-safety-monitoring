"""
Fire / smoke model wrapper.

Normalizes ultralytics predictions from the fire_smoke model into a
small, consistent detection dict list that src/safety/rules.py can
turn into FIRE / SMOKE violations.

Detection dict format:
    {
        "class_id": int,
        "class_name": str,     # raw model label, e.g. "Fire", "Smoke"
        "type": str,           # normalized: "fire" or "smoke" (or raw lowercase)
        "confidence": float,
        "bbox": [x1, y1, x2, y2],
    }
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


class FireDetector:
    def __init__(
        self,
        model_path: Union[str, Path],
        confidence: float = 0.25,
        image_size: int = 640,
        device: Optional[str] = None,
    ):
        if YOLO is None:
            raise ImportError("ultralytics is required. Install with: pip install ultralytics")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Fire/smoke model not found: {model_path}")

        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.class_names: Dict[int, str] = self.model.names

    def predict(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run detection on a single frame. Fire/smoke doesn't need tracking."""

        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fire/smoke inference failed on frame: %s", exc)
            return []

        return self._parse_result(results[0] if results else None)

    def _parse_result(self, result) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []

        if result is None or result.boxes is None or len(result.boxes) == 0:
            return detections

        for box in result.boxes:
            try:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping malformed fire/smoke box: %s", exc)
                continue

            raw_name = self.class_names.get(class_id, str(class_id))

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": raw_name,
                    "type": raw_name.strip().lower(),
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                }
            )

        return detections
