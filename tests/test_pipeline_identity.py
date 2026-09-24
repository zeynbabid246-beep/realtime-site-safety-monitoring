"""
Pipeline-integration tests for the identity layer.

Verifies that:
- SafetyPipeline runs unchanged when no FaceRecognitionService is given
  (existing behavior is preserved),
- when a service IS given, FrameResult carries identities and the
  annotated frame gains the identity overlay,
- a crashing face service can never take the safety pipeline down.
"""

import numpy as np
import pytest

from src.pipeline import SafetyPipeline, summarize_identities, identities_for_ui
from src.face_recognition.service import FaceRecognitionService
from src.face_recognition.registry import WorkerRegistry
from src.face_recognition.config import FaceRecognitionConfig
from src.face_recognition.backends import StubFaceBackend


# Mirrors tests/test_pipeline.py fakes (kept local: tests/ is not a package).
CLASS_NAMES = {5: "Person", 6: "Safety Cone", 8: "machinery", 10: "vehicle"}


class FakeHazardDetector:
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
        return []


class FakeFireDetector:
    def __init__(self, detections):
        self._detections = detections

    def predict(self, frame):
        return [dict(d) for d in self._detections]


class ExplodingFaceService:
    """A service that raises - the pipeline must survive this."""

    def identify_faces(self, frame, persons):
        raise RuntimeError("model exploded")

    def last_identities(self, frame, persons):
        return []

    def stats(self):
        return {"workers_registered": 0}


def _persons():
    return [
        {"class_id": 5, "class_name": "Person", "confidence": 0.9,
         "bbox": [100, 300, 200, 600], "track_id": 1},
    ]


def test_pipeline_without_face_service_still_works():
    pipeline = SafetyPipeline(
        hazard_detector=FakeHazardDetector(_persons()),
        fire_detector=FakeFireDetector([]),
        enable_track_confirmation=False,
    )
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    out = pipeline.process_frame(frame, track=True, draw=True)

    assert out.identities == []
    assert out.result["statistics"]["persons"] == 1
    assert out.annotated is not None


def test_pipeline_attaches_verified_identities(tmp_path):
    config = FaceRecognitionConfig(data_dir=tmp_path / "face_data", backend="stub")
    config.detection_threshold = 0.5
    config.verification_threshold = 0.5
    config.cache_ttl_s = 0.0
    config.ensure_dirs()

    service = FaceRecognitionService(
        config, backend=StubFaceBackend(), registry=WorkerRegistry(config)
    )
    service.add_worker("alice", name="Alice")
    service.registry.add_reference_image("alice", np.full((64, 64, 3), 200, dtype=np.uint8))

    # Make the head crop of the person box match the registered reference
    # intensity (stub backend keys on mean brightness).
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    frame[300:405, 100:200, :] = 200  # head band of the person box

    pipeline = SafetyPipeline(
        hazard_detector=FakeHazardDetector(_persons()),
        fire_detector=FakeFireDetector([]),
        enable_track_confirmation=False,
        face_service=service,
    )
    out = pipeline.process_frame(frame, track=True, draw=True)

    assert len(out.identities) == 1
    identity = out.identities[0]
    assert identity.verified is True
    assert identity.worker_name == "Alice"
    # identity overlay must be drawn (label text present in the frame)
    assert out.annotated is not None

    ui = identities_for_ui(out.identities)
    assert ui[0]["label"] == "Alice"
    summary = summarize_identities(out.identities)
    assert summary["workers_verified"] == 1
    assert summary["verified_workers"] == ["Alice"]
    assert summary["unknown_faces"] == 0


def test_pipeline_survives_face_service_crash():
    pipeline = SafetyPipeline(
        hazard_detector=FakeHazardDetector(_persons()),
        fire_detector=FakeFireDetector([]),
        enable_track_confirmation=False,
        face_service=ExplodingFaceService(),
    )
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    out = pipeline.process_frame(frame, track=True, draw=True)
    assert out.identities == []  # degraded, not crashed
    assert out.result["statistics"]["persons"] == 1


def test_pipeline_face_stride_skips_and_serves_cache(tmp_path):
    config = FaceRecognitionConfig(data_dir=tmp_path / "face_data", backend="stub")
    config.detection_threshold = 0.5
    config.verification_threshold = 0.5
    config.cache_ttl_s = 60.0
    config.ensure_dirs()

    service = FaceRecognitionService(
        config, backend=StubFaceBackend(), registry=WorkerRegistry(config)
    )
    service.add_worker("alice", name="Alice")
    service.registry.add_reference_image("alice", np.full((64, 64, 3), 200, dtype=np.uint8))

    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    frame[300:405, 100:200, :] = 200

    pipeline = SafetyPipeline(
        hazard_detector=FakeHazardDetector(_persons()),
        fire_detector=FakeFireDetector([]),
        enable_track_confirmation=False,
        face_service=service,
        face_frame_stride=2,
    )
    first = pipeline.process_frame(frame, track=True, draw=True)
    assert len(first.identities) == 1  # stride frame: verified

    second = pipeline.process_frame(frame, track=True, draw=True)
    assert len(second.identities) == 1  # skipped frame: cached identity served
    assert second.identities[0].worker_name == "Alice"
