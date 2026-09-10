"""Unit tests for src.safety.geometry (separation, zones, tracker, poles)."""

import pytest

shapely = pytest.importorskip("shapely")
from shapely.geometry import box

from src.safety.geometry import (
    DangerZoneTracker,
    bbox_separation,
    build_danger_zones,
    find_machinery_near_poles,
    find_people_inside_zones,
    get_bottom_center,
)

CONE, PERSON, MACHINERY, POLE = 6, 5, 8, 9


def _det(class_id, bbox, confidence=0.9):
    return {"class_id": class_id, "bbox": list(bbox), "confidence": confidence}


# ------------------------------------------------------------------
# bbox geometry
# ------------------------------------------------------------------

def test_bbox_separation_zero_when_overlapping():
    assert bbox_separation([0, 0, 100, 100], [50, 50, 150, 150]) == 0.0


def test_bbox_separation_measures_edge_gap():
    # 100px horizontal gap, no vertical gap.
    assert bbox_separation([0, 0, 100, 100], [200, 0, 300, 100]) == pytest.approx(100.0)


def test_bottom_center():
    assert get_bottom_center([10, 20, 30, 80]) == (20.0, 80.0)


# ------------------------------------------------------------------
# danger zones from cones
# ------------------------------------------------------------------

def test_build_danger_zones_needs_enough_cones():
    two = [_det(CONE, [100, 480, 120, 520]), _det(CONE, [150, 480, 170, 520])]
    assert build_danger_zones(two, min_cluster_size=3) == []


def test_build_danger_zones_from_cone_cluster():
    cones = [
        _det(CONE, [90, 480, 110, 520]),
        _det(CONE, [140, 480, 160, 520]),
        _det(CONE, [90, 530, 110, 570]),
        _det(CONE, [140, 530, 160, 570]),
    ]
    zones = build_danger_zones(cones, min_cluster_size=3, buffer_px=20, min_area_px=100)
    assert len(zones) >= 1
    assert zones[0].area > 0


# ------------------------------------------------------------------
# temporal zone stabilization
# ------------------------------------------------------------------

def test_zone_tracker_keeps_stable_id_across_frames():
    tracker = DangerZoneTracker(iou_match_threshold=0.3, max_missed_frames=5)
    poly = box(0, 0, 100, 100)
    first = tracker.update([poly])
    second = tracker.update([box(2, 2, 102, 102)])  # near-identical
    assert first[0]["zone_id"] == second[0]["zone_id"]
    assert second[0]["age"] == 2


def test_zone_tracker_assigns_new_id_for_new_region():
    tracker = DangerZoneTracker(iou_match_threshold=0.3)
    tracker.update([box(0, 0, 50, 50)])
    result = tracker.update([box(500, 500, 600, 600)])  # no overlap
    ids = {z["zone_id"] for z in result}
    assert len(ids) == 2


def test_zone_tracker_marks_missed_when_zone_disappears():
    tracker = DangerZoneTracker(max_missed_frames=5)
    tracker.update([box(0, 0, 100, 100)])
    result = tracker.update([])
    assert result[0]["missed"] == 1


# ------------------------------------------------------------------
# people in zones
# ------------------------------------------------------------------

def test_find_people_inside_zones():
    zone = box(0, 0, 100, 100)
    inside = {"track_id": 1, "bbox": [40, 50, 60, 90], "confidence": 0.9}
    outside = {"track_id": 2, "bbox": [300, 300, 320, 390], "confidence": 0.9}
    hits = find_people_inside_zones([inside, outside], [zone])
    assert len(hits) == 1
    assert hits[0]["track_id"] == 1
    assert hits[0]["zone_id"] == 1


# ------------------------------------------------------------------
# machinery near poles
# ------------------------------------------------------------------

def test_find_machinery_near_poles():
    dets = [
        _det(MACHINERY, [100, 100, 200, 200]),
        _det(POLE, [210, 100, 230, 200]),   # 10px gap
        _det(POLE, [900, 100, 920, 200]),   # far away
    ]
    warnings = find_machinery_near_poles(dets, threshold_pixels=250)
    assert len(warnings) == 1
    assert warnings[0]["distance_px"] == pytest.approx(10.0)
    assert warnings[0]["machinery_index"] == 0
