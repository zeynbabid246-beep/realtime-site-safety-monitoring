"""Unit tests for src.hazard.detection_filter (stateless confidence/size/aspect/ROI)."""

from src.hazard.detection_filter import (
    build_rectangular_roi,
    filter_detections,
    filter_detections_with_stats,
    is_detection_valid,
    passes_confidence,
    passes_size,
    person_aspect_ratio_ok,
)

FRAME = (640, 640)  # (height, width)


def _det(class_name, confidence, bbox, **extra):
    return {"class_name": class_name, "confidence": confidence, "bbox": bbox, **extra}


# ------------------------------------------------------------------
# confidence
# ------------------------------------------------------------------

def test_person_confidence_threshold():
    assert passes_confidence("Person", 0.60) is True
    assert passes_confidence("Person", 0.40) is True  # Threshold lowered from 0.52 to 0.35
    assert passes_confidence("Person", 0.30) is False


def test_unknown_class_falls_back_to_default():
    # DEFAULT_MIN_CONFIDENCE == 0.25
    assert passes_confidence("SomeUnknownClass", 0.30) is True
    assert passes_confidence("SomeUnknownClass", 0.10) is False


def test_edge_of_frame_requires_higher_confidence():
    # A person touching the left edge needs base 0.35 + 0.12 edge bonus = 0.47.
    bbox_at_edge = [0, 200, 60, 600]
    assert passes_confidence("Person", 0.45, bbox_at_edge, FRAME) is False
    assert passes_confidence("Person", 0.50, bbox_at_edge, FRAME) is True


# ------------------------------------------------------------------
# size (with resolution + perspective scaling)
# ------------------------------------------------------------------

def test_tiny_person_rejected_by_size():
    assert passes_size("Person", [100, 100, 115, 130], FRAME) is False


def test_normal_person_passes_size():
    assert passes_size("Person", [100, 400, 200, 600], FRAME) is True


def test_size_floor_scales_with_resolution():
    # A 40px-wide person is fine at 640p but must be rejected at 4K,
    # where the same physical worker would be far larger in pixels.
    bbox = [100, 100, 140, 200]
    assert passes_size("Person", bbox, (640, 640)) is True
    assert passes_size("Person", bbox, (2160, 3840)) is False


# ------------------------------------------------------------------
# person aspect ratio (poles / debris)
# ------------------------------------------------------------------

def test_needle_box_rejected_as_person():
    # pole-like: very tall and thin, low confidence -> strict bounds
    assert person_aspect_ratio_ok([100, 100, 108, 500], confidence=0.4) is False


def test_plausible_person_aspect_ratio_accepted():
    assert person_aspect_ratio_ok([100, 300, 160, 480], confidence=0.9) is True


# ------------------------------------------------------------------
# end-to-end filtering + stats
# ------------------------------------------------------------------

def test_missing_bbox_is_invalid():
    assert is_detection_valid({"class_name": "Person", "confidence": 0.9}) is False


def test_roi_drops_detection_outside_region():
    roi = build_rectangular_roi(0, 300, 640, 640)  # ignore the top band
    inside = _det("Person", 0.9, [100, 400, 200, 600])
    outside = _det("Person", 0.9, [100, 50, 200, 250])
    assert is_detection_valid(inside, roi, FRAME) is True
    assert is_detection_valid(outside, roi, FRAME) is False


def test_filter_detections_with_stats_reports_reasons():
    dets = [
        _det("Person", 0.9, [100, 400, 200, 600]),      # kept
        _det("Person", 0.20, [100, 400, 200, 600]),     # rejected_confidence
        _det("Person", 0.9, [100, 100, 110, 120]),      # rejected_size
        {"class_name": "Person", "confidence": 0.9},    # rejected_missing_bbox
    ]
    kept, stats = filter_detections_with_stats(dets, frame_shape=FRAME)
    assert len(kept) == 1
    assert stats["total"] == 4
    assert stats["kept"] == 1
    assert stats["rejected_confidence"] == 1
    assert stats["rejected_missing_bbox"] == 1


def test_filter_detections_matches_stats_kept():
    dets = [
        _det("Person", 0.9, [100, 400, 200, 600]),
        _det("Person", 0.10, [100, 400, 200, 600]),
    ]
    assert len(filter_detections(dets, frame_shape=FRAME)) == 1
