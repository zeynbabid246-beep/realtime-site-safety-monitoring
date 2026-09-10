"""Unit tests for src.safety.distance_calculator (pixel gap + perspective metres)."""

import pytest

from src.safety.distance_calculator import (
    calculate_person_machine_distances,
    calculate_perspective_distance,
    calculate_proximity_distance,
    find_closest_machine,
)


def test_proximity_distance_is_edge_gap():
    assert calculate_proximity_distance([0, 0, 50, 170], [200, 0, 250, 170]) == pytest.approx(150.0)


def test_perspective_distance_uses_person_height_as_scale():
    # Person 170px tall -> 100 px/m at 1.7m assumed height.
    # 150px gap -> 1.5m.
    dist_m, px_per_m = calculate_perspective_distance([0, 0, 50, 170], [200, 0, 250, 170])
    assert px_per_m == pytest.approx(100.0)
    assert dist_m == pytest.approx(1.5)


def test_perspective_distance_scales_with_assumed_height():
    dist_m, px_per_m = calculate_perspective_distance(
        [0, 0, 50, 170], [200, 0, 250, 170], assumed_person_height_m=3.4
    )
    assert px_per_m == pytest.approx(50.0)
    assert dist_m == pytest.approx(3.0)


def test_perspective_distance_none_for_tiny_person():
    # Person box shorter than MIN_PERSON_HEIGHT_PX -> untrustworthy scale.
    dist_m, px_per_m = calculate_perspective_distance([0, 0, 50, 5], [200, 0, 250, 170])
    assert dist_m is None and px_per_m is None


def test_perspective_distance_none_for_bad_bbox():
    assert calculate_perspective_distance(None, [0, 0, 10, 10]) == (None, None)


def test_find_closest_machine_attaches_distance_fields():
    person = [0, 0, 50, 170]
    machines = [
        {"class_name": "Excavator", "confidence": 0.9, "bbox": [400, 0, 500, 170]},
        {"class_name": "Truck", "confidence": 0.8, "bbox": [200, 0, 250, 170]},
    ]
    closest = find_closest_machine(person, machines)
    assert closest["class_name"] == "Truck"
    assert closest["distance_px"] == pytest.approx(150.0)
    assert closest["distance_m_est"] == pytest.approx(1.5)
    assert closest["px_per_m"] == pytest.approx(100.0)


def test_find_closest_machine_skips_malformed_bbox():
    person = [0, 0, 50, 170]
    machines = [{"class_name": "Bad", "bbox": None}, {"class_name": "Truck", "bbox": [200, 0, 250, 170]}]
    closest = find_closest_machine(person, machines)
    assert closest["class_name"] == "Truck"


def test_calculate_person_machine_distances_no_machine_gives_none_fields():
    persons = [{"track_id": 1, "bbox": [0, 0, 50, 170], "confidence": 0.9}]
    results = calculate_person_machine_distances(persons, [])
    assert results[0]["machine_class"] is None
    assert results[0]["distance_m_est"] is None
    assert results[0]["px_per_m"] is None


def test_calculate_person_machine_distances_includes_metres():
    persons = [{"track_id": 7, "bbox": [0, 0, 50, 170], "confidence": 0.9}]
    machines = [{"class_name": "Truck", "confidence": 0.8, "bbox": [200, 0, 250, 170]}]
    results = calculate_person_machine_distances(persons, machines)
    assert results[0]["track_id"] == 7
    assert results[0]["distance_m_est"] == pytest.approx(1.5)
