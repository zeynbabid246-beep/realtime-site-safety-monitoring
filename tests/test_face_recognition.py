"""
Tests for the face recognition / worker verification layer.

All tests are MODEL-FREE: they use StubFaceBackend (no TensorFlow, no
.h5 weights) so the whole suite runs in CI. The real Siamese backend is
exercised separately (see docs/FACE_RECOGNITION.md) because it needs TF.
"""

import json

import numpy as np
import pytest

from src.face_recognition.config import FaceRecognitionConfig
from src.face_recognition.registry import WorkerRegistry
from src.face_recognition.face_detector import FaceDetector
from src.face_recognition.service import FaceRecognitionService
from src.face_recognition.backends import StubFaceBackend, create_backend
from src.face_recognition import calibration as face_calibration
from src.face_recognition.preprocessing import preprocess_face, batch_preprocess


# ------------------------------------------------------------------
# fixtures / helpers
# ------------------------------------------------------------------

@pytest.fixture
def face_config(tmp_path):
    config = FaceRecognitionConfig(data_dir=tmp_path / "face_data", backend="stub")
    config.detection_threshold = 0.5
    config.verification_threshold = 0.5
    config.cache_ttl_s = 0.0
    config.ensure_dirs()
    return config


def _face_image(center_value: float, size: int = 64) -> np.ndarray:
    """A flat BGR image - the stub backend keys on mean intensity."""
    image = np.full((size, size, 3), float(center_value), dtype=np.uint8)
    return image


@pytest.fixture
def service(face_config):
    registry = WorkerRegistry(face_config)
    backend = StubFaceBackend()
    svc = FaceRecognitionService(face_config, backend=backend, registry=registry)
    return svc


def _register(svc, worker_id, values, name=None):
    svc.add_worker(worker_id, name=name or worker_id)
    for value in values:
        svc.registry.add_reference_image(worker_id, _face_image(value))


# ------------------------------------------------------------------
# registry
# ------------------------------------------------------------------

def test_registry_add_and_list_workers(face_config):
    registry = WorkerRegistry(face_config)
    registry.add_worker("worker_001", name="Ahmed Ali", role="crane operator")
    registry.add_reference_image("worker_001", _face_image(120))

    reloaded = WorkerRegistry(face_config)  # fresh instance reads workers.json
    workers = reloaded.list_workers()
    assert len(workers) == 1
    assert workers[0].name == "Ahmed Ali"
    assert workers[0].role == "crane operator"
    assert len(workers[0].reference_images) == 1
    assert (face_config.input_images_dir / "worker_001").is_dir()


def test_registry_worker_id_is_sanitized(face_config):
    registry = WorkerRegistry(face_config)
    record = registry.add_worker("../../etc/passwd")
    assert "/" not in record.worker_id
    assert "\\" not in record.worker_id
    assert ".." not in record.worker_id


def test_registry_remove_deactivates_then_purges(face_config):
    registry = WorkerRegistry(face_config)
    registry.add_worker("w1", name="One")
    registry.add_reference_image("w1", _face_image(90))

    assert registry.remove_worker("w1") is True
    assert registry.get_worker("w1").active is False
    assert registry.worker_ids() == []  # active_only default

    assert registry.remove_worker("w1", delete_images=True) is True
    assert registry.get_worker("w1") is None
    assert not (face_config.worker_image_dir("w1")).exists()


def test_registry_auto_discover_registers_folders(face_config):
    folder = face_config.worker_image_dir("new_guy")
    folder.mkdir(parents=True)
    import cv2

    cv2.imwrite(str(folder / "a.jpg"), _face_image(100))

    registry = WorkerRegistry(face_config)
    discovered = registry.auto_discover_workers()
    assert discovered == ["new_guy"]
    record = registry.get_worker("new_guy")
    assert record is not None
    assert record.reference_images == ["new_guy/a.jpg"]

    # idempotent: second run discovers nothing new
    assert registry.auto_discover_workers() == []


# ------------------------------------------------------------------
# preprocessing
# ------------------------------------------------------------------

def test_preprocess_face_matches_notebook_contract():
    crop = np.full((50, 30, 3), 200, dtype=np.uint8)
    tensor = preprocess_face(crop)
    assert tensor is not None
    assert tensor.shape == (1, 100, 100, 3)
    assert tensor.dtype == np.float32
    assert 0.0 <= float(tensor.max()) <= 1.0
    assert abs(float(tensor.mean()) - 200 / 255.0) < 0.01


def test_preprocess_face_rejects_garbage():
    assert preprocess_face(None) is None
    assert preprocess_face(np.zeros((0, 10, 3), dtype=np.uint8)) is None
    assert preprocess_face(np.zeros((10, 10), dtype=np.uint8)) is None


def test_batch_preprocess_skips_invalid():
    crops = [np.full((20, 20, 3), 50, dtype=np.uint8), None, np.full((20, 20, 3), 90, dtype=np.uint8)]
    batch, indices = batch_preprocess(crops)
    assert batch.shape == (2, 100, 100, 3)
    assert indices == [0, 2]


def test_crop_with_margin_clamps_and_narrows():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    crop = face_crop = None
    from src.face_recognition.preprocessing import crop_with_margin

    crop = crop_with_margin(frame, [-50, -50, 500, 500])
    assert crop.shape[:2] == (200, 200)  # clamped to frame

    narrowed = crop_with_margin(frame, [0, 0, 100, 200], width_fraction=0.5)
    assert narrowed.shape[1] == 50  # half the width, centred

    assert crop_with_margin(frame, [10, 10, 12, 12], min_size=40) is None
    assert crop_with_margin(frame, None) is None
    assert face_crop is None


# ------------------------------------------------------------------
# face crops from person boxes
# ------------------------------------------------------------------

def test_face_detector_crops_head_region():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    persons = [
        {"track_id": 1, "bbox": [100, 100, 200, 500], "confidence": 0.9},
        {"track_id": 2, "bbox": [300, 100, 400, 500], "confidence": 0.2},  # below conf floor
        {"track_id": 3, "bbox": [5, 5, 12, 12], "confidence": 0.9},        # too small
    ]
    detector = FaceDetector(min_face_size_px=40, min_person_confidence=0.45)
    candidates = detector.extract_face_candidates(frame, persons)
    assert len(candidates) == 1
    assert candidates[0].track_id == 1
    x1, y1, x2, y2 = candidates[0].bbox
    assert y2 - y1 == pytest.approx((500 - 100) * 0.35, rel=0.02)


# ------------------------------------------------------------------
# service: verification + multi-worker decision
# ------------------------------------------------------------------

def test_verify_matches_correct_worker(service):
    _register(service, "alice", [180.0, 180.0, 180.0], name="Alice")
    _register(service, "bob", [40.0, 40.0, 40.0], name="Bob")

    identity = service.verify_crop(_face_image(180.0), track_id=7)
    assert identity.verified is True
    assert identity.worker_id == "alice"
    assert identity.worker_name == "Alice"
    assert identity.error is None


def test_unknown_face_returns_not_verified(service):
    _register(service, "alice", [200.0])
    _register(service, "bob", [30.0])
    service.config.detection_threshold = 0.95
    service.config.verification_threshold = 0.9

    identity = service.verify_crop(_face_image(128.0))
    assert identity.verified is False
    assert identity.worker_id is None
    assert identity.label == "Unknown"
    assert len(identity.matches) == 2  # scored against ALL workers


def test_no_workers_means_no_identities(service):
    persons = [{"track_id": 1, "bbox": [10, 10, 60, 200], "confidence": 0.9}]
    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    assert service.identify_faces(frame, persons) == []


def test_multiple_faces_in_one_frame(service):
    _register(service, "alice", [200.0], name="Alice")
    _register(service, "bob", [30.0], name="Bob")

    frame = np.zeros((500, 500, 3), dtype=np.uint8)
    # Give each person region a distinct flat intensity so the stub
    # backend separates them.
    frame[0:250, :, :] = 200
    frame[250:, :, :] = 30

    persons = [
        {"track_id": 1, "bbox": [0, 10, 100, 200], "confidence": 0.9},
        {"track_id": 2, "bbox": [0, 260, 100, 450], "confidence": 0.9},
    ]
    identities = service.identify_faces(frame, persons)
    by_track = {i.track_id: i for i in identities}
    assert by_track[1].verified and by_track[1].worker_id == "alice"
    assert by_track[2].verified and by_track[2].worker_id == "bob"


def test_min_margin_blocks_ambiguous_match(service):
    # One worker whose references straddle the live value: runner-up
    # margin forces Unknown even though both "verify".
    _register(service, "alice", [190.0])
    _register(service, "bob", [170.0])
    service.config.min_margin = 0.9  # huge margin

    identity = service.verify_crop(_face_image(180.0))
    assert identity.verified is False
    assert identity.worker_id is None


def test_cache_ttl_reuses_identity(service):
    service.config.cache_ttl_s = 60.0
    _register(service, "alice", [200.0])

    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    persons = [{"track_id": 3, "bbox": [10, 10, 60, 200], "confidence": 0.9}]

    first = service.identify_faces(frame, persons)
    assert first, "expected at least one identity on the first pass"
    # Break the backend: a cached identity must still be returned.
    service.backend = None
    second = service.identify_faces(frame, persons)
    assert second and second[0].track_id == first[0].track_id
    assert second[0].verified == first[0].verified


def test_to_dict_is_json_safe(service):
    _register(service, "alice", [200.0], name="Alice")
    identity = service.verify_crop(_face_image(200.0))
    payload = json.dumps(identity.to_dict())  # must not raise
    assert "\"verified\": true" in payload


# ------------------------------------------------------------------
# calibration helpers (distribution maths, no model)
# ------------------------------------------------------------------

def test_rates_compute_tar_far():
    genuine = np.array([0.95, 0.92, 0.97])
    impostor = np.array([0.10, 0.20, 0.05])
    tar, far = face_calibration._rates(genuine, impostor, 0.85, 0.5)
    assert tar == 1.0
    assert far == 0.0


def test_apply_calibration_overrides_thresholds():
    config = FaceRecognitionConfig()
    before = (config.detection_threshold, config.verification_threshold)
    config.apply_calibration({"detection_threshold": 0.4, "verification_threshold": 0.3})
    assert config.detection_threshold == 0.4
    assert config.verification_threshold == 0.3
    config.apply_calibration(None)  # no-op
    assert (config.detection_threshold, config.verification_threshold) == (0.4, 0.3)
    assert before != (0.4, 0.3)


def test_save_and_load_calibration(face_config, tmp_path):
    result = {
        "suggested_detection_threshold": 0.83,
        "suggested_verification_threshold": 0.6,
        "calibrated_at": 123.0,
    }
    path = face_calibration.save_calibration(face_config, result)
    assert path.exists()
    loaded = face_calibration.load_calibration(face_config)
    assert loaded["suggested_detection_threshold"] == 0.83


# ------------------------------------------------------------------
# backend factory
# ------------------------------------------------------------------

def test_create_backend_stub():
    config = FaceRecognitionConfig(backend="stub")
    assert isinstance(create_backend(config), StubFaceBackend)


def test_create_backend_unknown_raises():
    with pytest.raises(ValueError):
        create_backend(FaceRecognitionConfig(backend="quantum"))


def test_stub_backend_scores_same_image_high():
    backend = StubFaceBackend()
    image = preprocess_face(_face_image(150.0))
    score = backend.verify_batch(image, image)[0]
    assert score > 0.9

    other = preprocess_face(_face_image(10.0))
    low = backend.verify_batch(image, other)[0]
    assert low < score
