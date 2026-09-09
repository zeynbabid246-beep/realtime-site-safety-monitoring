"""
Unit and integration test for the detection filtering and PPE spatial association layer.

Tests:
1. Minimum object size filter per class (e.g. Person min 30x50px, Hardhat min 10x10px)
2. Per-class confidence thresholds
3. Person aspect-ratio filter (rejects absurdly wide or thin slivers/poles)
4. Edge-of-frame confidence bonus (e.g. ID:504 on border)
5. Track confirmation buffer (TrackConfirmationTracker - MIN_TRACK_HITS = 3)
6. Anatomical PPE spatial association (PPEPersonAssociator - head top 30%, torso 25%-75%)
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hazard.detection_filter import (
    filter_detections,
    TrackConfirmationTracker,
    PPEPersonAssociator,
    is_detection_large_enough,
    passes_class_confidence,
    passes_person_aspect_ratio,
    touches_frame_edge,
    PERSON,
    HARDHAT,
    NO_HARDHAT,
    NO_SAFETY_VEST,
    MACHINERY,
)
from src.safety.rules import SafetyConfig, get_ppe_violations


def test_min_size_filter():
    print("Testing minimum size filter...")
    # Person min size is (30, 50)
    small_person_box = [100, 100, 115, 120]  # width=15, height=20 -> too small
    normal_person_box = [100, 100, 150, 220]  # width=50, height=120 -> large enough

    assert not is_detection_large_enough(small_person_box, PERSON), "Small person box should be rejected"
    assert is_detection_large_enough(normal_person_box, PERSON), "Normal person box should pass"

    # Hardhat min size is (10, 10)
    small_hardhat = [10, 10, 15, 15]  # 5x5 -> too small
    normal_hardhat = [10, 10, 30, 30]  # 20x20 -> large enough
    assert not is_detection_large_enough(small_hardhat, HARDHAT)
    assert is_detection_large_enough(normal_hardhat, HARDHAT)
    print("  [PASS] Minimum size filter working correctly")


def test_confidence_filter():
    print("Testing per-class confidence thresholds...")
    # PERSON threshold is 0.45, MACHINERY is 0.50
    assert passes_class_confidence(PERSON, 0.46)
    assert not passes_class_confidence(PERSON, 0.40)
    assert passes_class_confidence(MACHINERY, 0.52)
    assert not passes_class_confidence(MACHINERY, 0.35)  # Catches weak "House -> machinery 0.34"
    print("  [PASS] Per-class confidence thresholds working correctly")


def test_person_aspect_ratio():
    print("Testing person aspect ratio guard...")
    # Normal upright person: width=40, height=100 -> ratio=2.5 (passes 0.7 <= ratio <= 6.0)
    normal_person = [100, 100, 140, 200]
    assert passes_person_aspect_ratio(normal_person)

    # Pole misclassified as person: width=10, height=120 -> ratio=12.0 (too thin, rejected)
    pole_person = [100, 100, 110, 220]
    assert not passes_person_aspect_ratio(pole_person)

    # Wide horizontal object: width=120, height=30 -> ratio=0.25 (too flat, rejected)
    wide_person = [100, 100, 220, 130]
    assert not passes_person_aspect_ratio(wide_person)
    print("  [PASS] Person aspect ratio guard working correctly")


def test_edge_of_frame_bonus():
    print("Testing edge-of-frame confidence bonus...")
    frame_shape = (1080, 1920)
    # Box touching the right edge: x2 = 1918 (within 6px margin)
    edge_bbox = [1880, 500, 1918, 600]
    assert touches_frame_edge(edge_bbox, frame_shape, margin=6.0)

    # Interior box: well away from borders
    inner_bbox = [500, 500, 600, 700]
    assert not touches_frame_edge(inner_bbox, frame_shape, margin=6.0)

    # In filter_detections, edge box with Person 0.50 should be dropped if threshold (0.45) + bonus (0.15) = 0.60
    edge_det = {"bbox": edge_bbox, "confidence": 0.50, "class_id": PERSON, "track_id": 504}
    inner_det = {"bbox": inner_bbox, "confidence": 0.50, "class_id": PERSON, "track_id": 101}
    kept = filter_detections([edge_det, inner_det], frame_shape=frame_shape)
    assert edge_det not in kept, "Edge detection with conf 0.50 should be dropped by edge bonus"
    assert inner_det in kept, "Inner detection with conf 0.50 should be kept"
    print("  [PASS] Edge-of-frame confidence bonus working correctly")


def test_track_confirmation_tracker():
    print("Testing track confirmation buffer (temporal stability)...")
    tracker = TrackConfirmationTracker(min_hits=3, max_missed_frames=5)

    flicker_person = {"bbox": [100, 100, 160, 240], "class_id": PERSON, "track_id": 999, "confidence": 0.8}
    stable_person = {"bbox": [200, 200, 260, 340], "class_id": PERSON, "track_id": 42, "confidence": 0.85}

    # Frame 1: neither should be confirmed yet
    res1 = tracker.update([flicker_person, stable_person])
    assert len(res1) == 0, "No tracks should be confirmed on frame 1"

    # Frame 2: stable_person appears again; flicker_person does not appear
    res2 = tracker.update([stable_person])
    assert len(res2) == 0, "Stable track needs 3 hits; only at 2"

    # Frame 3: stable_person appears third time
    res3 = tracker.update([stable_person])
    assert len(res3) == 1, "Stable track should now be confirmed (hit 3)"
    assert res3[0]["track_id"] == 42
    print("  [PASS] TrackConfirmationTracker successfully confirmed stable track and rejected flicker")


def test_ppe_spatial_association():
    print("Testing PPE spatial association (head/torso regions)...")
    # Worker: x: 100->200, y: 100->300 (width=100, height=200)
    # Head region (top 30%): y from 100 to 160
    # Torso region (25%-75%): y from 150 to 250
    worker_bbox = [100.0, 100.0, 200.0, 300.0]

    associator = PPEPersonAssociator()

    head_reg = associator.head_region(worker_bbox)
    assert head_reg == (100.0, 100.0, 200.0, 160.0)

    torso_reg = associator.torso_region(worker_bbox)
    assert torso_reg == (100.0, 150.0, 200.0, 250.0)

    # 1. Valid NO_HARDHAT box on head (y: 105->145)
    valid_no_hardhat = [120.0, 105.0, 180.0, 145.0]
    assert associator.ppe_overlaps_head(worker_bbox, valid_no_hardhat)

    # 2. Spurious NO_HARDHAT box at feet level (y: 260->295)
    feet_no_hardhat = [120.0, 260.0, 180.0, 295.0]
    assert not associator.ppe_overlaps_head(worker_bbox, feet_no_hardhat), "Hardhat at feet must be rejected"

    # 3. Valid NO_SAFETY_VEST on torso (y: 160->230)
    valid_vest = [115.0, 160.0, 185.0, 230.0]
    assert associator.ppe_overlaps_torso(worker_bbox, valid_vest)

    # 4. Spurious vest on head (y: 100->130)
    head_vest = [120.0, 100.0, 180.0, 130.0]
    assert not associator.ppe_overlaps_torso(worker_bbox, head_vest), "Vest on head must be rejected"

    # Test integration with rules.py get_ppe_violations
    persons = [{"track_id": 1, "bbox": worker_bbox, "confidence": 0.9}]
    detections = [
        [100, 100, 200, 300, 0.9, PERSON],
        [120, 260, 180, 295, 0.85, NO_HARDHAT],  # At feet -> rejected with spatial matching!
        [115, 160, 185, 230, 0.88, NO_SAFETY_VEST],  # On torso -> accepted!
    ]
    cfg = SafetyConfig(use_spatial_ppe_matching=True)
    violations = get_ppe_violations(persons, detections, cfg)
    v_types = [v["type"] for v in violations]
    assert "NO_SAFETY_VEST" in v_types, "Torso vest violation must be reported"
    assert "NO_HARDHAT" not in v_types, "Feet hardhat violation must NOT be reported when spatial matching is ON"

    # If spatial matching is disabled, feet hardhat would be falsely reported
    cfg_no_spatial = SafetyConfig(use_spatial_ppe_matching=False)
    violations_legacy = get_ppe_violations(persons, detections, cfg_no_spatial)
    v_types_legacy = [v["type"] for v in violations_legacy]
    assert "NO_HARDHAT" in v_types_legacy, "Feet hardhat was falsely reported in legacy mode"

    print("  [PASS] PPE anatomical spatial association verified")


def main():
    print("=" * 60)
    print("DETECTION FILTERING & VALIDATION LAYER TESTS")
    print("=" * 60)
    test_min_size_filter()
    test_confidence_filter()
    test_person_aspect_ratio()
    test_edge_of_frame_bonus()
    test_track_confirmation_tracker()
    test_ppe_spatial_association()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()

