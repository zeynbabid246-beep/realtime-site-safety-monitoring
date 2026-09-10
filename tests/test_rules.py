"""Unit tests for src.safety.rules (PPE anatomical matching, proximity, risk levels)."""

from src.safety.rules import (
    NO_HARDHAT,
    NO_MASK,
    NO_SAFETY_VEST,
    SafetyConfig,
    calculate_risk_level,
    get_machine_distance_violations,
    get_ppe_violations,
)

CONFIG = SafetyConfig()


def _person(track_id=1, bbox=(100, 100, 200, 400)):
    return {"track_id": track_id, "bbox": list(bbox), "confidence": 0.9}


def _ppe(class_id, bbox, confidence=0.8):
    return {"class_id": class_id, "bbox": list(bbox), "confidence": confidence}


# ------------------------------------------------------------------
# PPE anatomical association
# ------------------------------------------------------------------

def test_no_hardhat_in_head_region_is_violation():
    # head region = top 35% of person bbox (y 100..205)
    viols = get_ppe_violations([_person()], [_ppe(NO_HARDHAT, [120, 120, 180, 190])], CONFIG)
    assert len(viols) == 1
    assert viols[0]["type"] == "NO_HARDHAT"
    assert viols[0]["severity"] == "MEDIUM"


def test_no_hardhat_near_feet_is_not_a_violation():
    # Same box but down at the feet -> outside the head region.
    viols = get_ppe_violations([_person()], [_ppe(NO_HARDHAT, [120, 340, 180, 395])], CONFIG)
    assert viols == []


def test_no_safety_vest_matched_to_torso():
    # torso region = 15%..85% of height (y 145..355)
    viols = get_ppe_violations([_person()], [_ppe(NO_SAFETY_VEST, [110, 200, 190, 300])], CONFIG)
    assert len(viols) == 1
    assert viols[0]["type"] == "NO_SAFETY_VEST"


def test_no_mask_is_low_severity():
    viols = get_ppe_violations([_person()], [_ppe(NO_MASK, [120, 120, 180, 190])], CONFIG)
    assert len(viols) == 1
    assert viols[0]["type"] == "NO_MASK"
    assert viols[0]["severity"] == "LOW"


def test_low_confidence_ppe_ignored():
    viols = get_ppe_violations(
        [_person()], [_ppe(NO_HARDHAT, [120, 120, 180, 190], confidence=0.10)], CONFIG
    )
    assert viols == []


def test_ppe_assigned_to_single_best_person():
    # Two overlapping people, one NO_HARDHAT box -> exactly one violation.
    people = [_person(1, (100, 100, 200, 400)), _person(2, (150, 100, 250, 400))]
    viols = get_ppe_violations(people, [_ppe(NO_HARDHAT, [160, 120, 190, 190])], CONFIG)
    assert len(viols) == 1


# ------------------------------------------------------------------
# machine proximity: perspective (metres) vs pixel fallback
# ------------------------------------------------------------------

def _dist(distance_px=None, distance_m=None, track_id=1):
    return {
        "track_id": track_id,
        "machine_class": "Excavator",
        "machine_confidence": 0.9,
        "distance_px": distance_px,
        "distance_m_est": distance_m,
    }


def test_perspective_alerts_on_metres():
    cfg = SafetyConfig(use_perspective_distance=True, machine_distance_threshold_m=2.5)
    viols = get_machine_distance_violations([_dist(distance_px=150, distance_m=1.5)], cfg)
    assert len(viols) == 1
    assert viols[0]["measurement"] == "metres"


def test_perspective_suppects_small_pixel_gap_that_is_far_in_metres():
    # Two distant objects: tiny pixel gap but 8m apart -> no alert.
    cfg = SafetyConfig(use_perspective_distance=True, machine_distance_threshold_m=2.5,
                       machine_distance_threshold=250)
    viols = get_machine_distance_violations([_dist(distance_px=20, distance_m=8.0)], cfg)
    assert viols == []


def test_pixel_fallback_when_perspective_disabled():
    cfg = SafetyConfig(use_perspective_distance=False, machine_distance_threshold=250)
    viols = get_machine_distance_violations([_dist(distance_px=150, distance_m=None)], cfg)
    assert len(viols) == 1
    assert viols[0]["measurement"] == "pixels"


def test_pixel_fallback_when_no_metre_estimate():
    cfg = SafetyConfig(use_perspective_distance=True, machine_distance_threshold=250)
    viols = get_machine_distance_violations([_dist(distance_px=100, distance_m=None)], cfg)
    assert len(viols) == 1
    assert viols[0]["measurement"] == "pixels"


def test_no_alert_when_far_in_pixels_and_no_metres():
    cfg = SafetyConfig(use_perspective_distance=True, machine_distance_threshold=250)
    assert get_machine_distance_violations([_dist(distance_px=900, distance_m=None)], cfg) == []


# ------------------------------------------------------------------
# risk level
# ------------------------------------------------------------------

def _v(vtype, severity):
    return {"type": vtype, "severity": severity}


def test_risk_levels():
    assert calculate_risk_level([]) == "SAFE"
    assert calculate_risk_level([_v("NO_MASK", "LOW")]) == "LOW"
    assert calculate_risk_level([_v("NO_HARDHAT", "MEDIUM")]) == "MEDIUM"
    assert calculate_risk_level([_v("MACHINE_PROXIMITY", "HIGH")]) == "HIGH"
    assert calculate_risk_level([_v("FIRE", "CRITICAL")]) == "CRITICAL"


def test_risk_zone_plus_machine_is_critical():
    viols = [_v("DANGEROUS_ZONE", "HIGH"), _v("MACHINE_PROXIMITY", "HIGH")]
    assert calculate_risk_level(viols) == "CRITICAL"


def test_risk_two_serious_hazards_is_critical():
    viols = [_v("DANGEROUS_ZONE", "HIGH"), _v("POLE_PROXIMITY", "HIGH")]
    assert calculate_risk_level(viols) == "CRITICAL"


def test_risk_three_medium_escalates_to_high():
    viols = [_v("NO_HARDHAT", "MEDIUM"), _v("NO_SAFETY_VEST", "MEDIUM"), _v("NO_MASK", "MEDIUM")]
    assert calculate_risk_level(viols) == "HIGH"


def test_risk_worst_severity_wins_not_count():
    # One HIGH structural hazard + one MEDIUM PPE -> HIGH (not downgraded).
    viols = [_v("NO_HARDHAT", "MEDIUM"), _v("MACHINE_PROXIMITY", "HIGH")]
    assert calculate_risk_level(viols) == "HIGH"
