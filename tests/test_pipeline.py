"""Tests for src.pipeline: shared pipeline + API-contract serialization helpers."""

import numpy as np

from src.pipeline import (
    SafetyPipeline,
    detections_for_ui,
    serialize_result,
    summarize_result,
)

CLASS_NAMES = {5: "Person", 6: "Safety Cone", 8: "machinery", 10: "vehicle"}


class FakeHazardDetector:
    """Stand-in for HazardDetector: returns canned detections, no model."""

    PERSON_CLASS = 5
    MACHINERY_CLASS = 8
    VEHICLE_CLASS = 10

    def __init__(self, detections):
        self._detections = detections
        self.class_names = CLASS_NAMES

    def track(self, frame, persist=True):
        return [dict(d) for d in self._detections]

    def predict(self, frame):
        return [dict(d) for d in self._detections]

    def extract_persons(self, detections):
        return [
            {"track_id": d.get("track_id"), "bbox": d["bbox"], "confidence": d["confidence"]}
            for d in detections if d["class_id"] == self.PERSON_CLASS
        ]

    def extract_machines(self, detections):
        return [
            {"class_id": d["class_id"], "class_name": d["class_name"],
             "confidence": d["confidence"], "bbox": d["bbox"]}
            for d in detections if d["class_id"] in (self.MACHINERY_CLASS, self.VEHICLE_CLASS)
        ]


class FakeFireDetector:
    def __init__(self, detections):
        self._detections = detections

    def predict(self, frame):
        return [dict(d) for d in self._detections]


# ------------------------------------------------------------------
# serialization helpers
# ------------------------------------------------------------------

def _sample_result():
    return {
        "risk_level": "MEDIUM",
        "violations": [{"type": "NO_HARDHAT", "severity": "MEDIUM"}],
        "violation_count": 1,
        "violation_counts": {"NO_HARDHAT": 1},
        "danger_zones": [],
        "people_inside_zones": [],
        "distance_results": [],
        "pole_results": [],
        "fire_detections": [{"type": "Smoke", "confidence": 0.4, "bbox": [0, 0, 10, 10]}],
        "statistics": {"persons": 2, "machines": 1, "danger_zones": 0},
    }


def test_serialize_result_is_json_safe_and_strips_polygons():
    from shapely.geometry import box

    result = _sample_result()
    result["danger_zones"] = [{"zone_id": 1, "age": 3, "missed": 0, "polygon": box(0, 0, 10, 10)}]
    serialized = serialize_result(result)

    import json
    json.dumps(serialized)  # must not raise
    zone = serialized["danger_zones"][0]
    assert zone["zone_id"] == 1
    assert zone["area"] > 0
    assert isinstance(zone["coordinates"][0], list)
    assert "polygon" not in zone


def test_summarize_result_counts_ppe_fire_smoke():
    summary = summarize_result(_sample_result())
    assert summary["risk_level"] == "MEDIUM"
    assert summary["ppe_violations"] == 1
    assert summary["smoke_count"] == 1
    assert summary["fire_count"] == 0
    assert summary["person_count"] == 2
    assert summary["machine_count"] == 1


def test_detections_for_ui_flat_shape():
    dets = [
        {"class_name": "Person", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
        {"class_name": "machinery", "confidence": 0.7, "bbox": None},  # dropped (no bbox)
    ]
    ui = detections_for_ui(dets)
    assert ui == [{"class": "Person", "confidence": 0.9}]


# ------------------------------------------------------------------
# process_frame with fake detectors (no YOLO weights needed)
# ------------------------------------------------------------------

def test_process_frame_runs_full_pipeline():
    detections = [
        {"class_id": 5, "class_name": "Person", "confidence": 0.9,
         "bbox": [100, 300, 200, 600], "track_id": 1},
        {"class_id": 8, "class_name": "machinery", "confidence": 0.9,
         "bbox": [240, 300, 400, 600], "track_id": 2},
    ]
    hazard = FakeHazardDetector(detections)
    fire = FakeFireDetector([])

    pipeline = SafetyPipeline(
        hazard_detector=hazard,
        fire_detector=fire,
        enable_track_confirmation=False,  # keep the test single-frame
    )
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    out = pipeline.process_frame(frame, track=True, draw=True)

    assert out.result["statistics"]["persons"] == 1
    assert out.result["statistics"]["machines"] == 1
    assert out.annotated is not None
    assert out.annotated.shape == frame.shape
    # The person is ~40px from the machinery; at 300px tall (176 px/m) that
    # is well under 2.5m -> a proximity violation must be raised.
    assert out.result["violation_counts"].get("MACHINE_PROXIMITY") == 1


def test_process_frame_single_image_mode_no_tracking():
    detections = [
        {"class_id": 5, "class_name": "Person", "confidence": 0.9,
         "bbox": [100, 300, 200, 600], "track_id": None},
    ]
    pipeline = SafetyPipeline(
        hazard_detector=FakeHazardDetector(detections),
        enable_track_confirmation=False,
    )
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    out = pipeline.process_frame(frame, track=False, draw=False)
    assert out.annotated is None
    assert out.result["statistics"]["persons"] == 1


def test_reset_clears_temporal_state():
    hazard = FakeHazardDetector([])
    pipeline = SafetyPipeline(hazard_detector=hazard)
    first_tracker = pipeline.zone_tracker
    pipeline.reset()
    assert pipeline.zone_tracker is not first_tracker
