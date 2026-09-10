"""End-to-end SafetyEngine tests on synthetic detections (no models required)."""

from src.safety.rules import SafetyConfig
from src.safety.safety_engine import SafetyEngine

FW = 1280.0  # reference frame width -> pixel thresholds unscaled


def _engine(**cfg):
    return SafetyEngine(config=SafetyConfig(**cfg))


def test_no_hazards_is_safe():
    result = _engine().analyze(detections=[], persons=[], machines=[], frame_width=FW)
    assert result["risk_level"] == "SAFE"
    assert result["violation_count"] == 0
    assert result["statistics"]["persons"] == 0


def test_fire_drives_critical():
    fire = [{"type": "Fire", "confidence": 0.70, "bbox": [10, 10, 80, 80]}]
    result = _engine().analyze(detections=[], persons=[], machines=[],
                               fire_detections=fire, frame_width=FW)
    assert result["risk_level"] == "CRITICAL"
    assert result["violation_counts"].get("FIRE") == 1


def test_smoke_alone_is_high():
    smoke = [{"type": "Smoke", "confidence": 0.40, "bbox": [10, 10, 80, 80]}]
    result = _engine().analyze(detections=[], persons=[], machines=[],
                               fire_detections=smoke, frame_width=FW)
    assert result["risk_level"] == "HIGH"
    assert result["violation_counts"].get("SMOKE") == 1


def test_person_close_to_machine_uses_metres():
    persons = [{"track_id": 1, "bbox": [0, 0, 50, 170], "confidence": 0.9}]  # 100 px/m
    machines = [{"class_id": 8, "class_name": "Excavator", "confidence": 0.9,
                 "bbox": [200, 0, 250, 170]}]  # 150px gap = 1.5m
    result = _engine(use_perspective_distance=True, machine_distance_threshold_m=2.5).analyze(
        detections=[], persons=persons, machines=machines, frame_width=FW
    )
    assert result["violation_counts"].get("MACHINE_PROXIMITY") == 1
    assert result["risk_level"] == "HIGH"


def test_distant_machine_no_alert_despite_small_pixel_gap():
    # Far-away pair: small pixel gap, large metre gap -> no proximity alert.
    persons = [{"track_id": 1, "bbox": [0, 0, 20, 68], "confidence": 0.9}]  # 40 px/m
    machines = [{"class_id": 8, "class_name": "Excavator", "confidence": 0.9,
                 "bbox": [100, 0, 140, 68]}]  # 80px gap = 2.0m ... under 2.5, so alert
    result = _engine(use_perspective_distance=True, machine_distance_threshold_m=1.0).analyze(
        detections=[], persons=persons, machines=machines, frame_width=FW
    )
    assert result["violation_counts"].get("MACHINE_PROXIMITY", 0) == 0


def test_person_inside_cone_zone_is_dangerous_zone():
    cones = [
        {"class_id": 6, "bbox": [90, 480, 110, 520], "confidence": 0.9},
        {"class_id": 6, "bbox": [140, 480, 160, 520], "confidence": 0.9},
        {"class_id": 6, "bbox": [90, 530, 110, 570], "confidence": 0.9},
        {"class_id": 6, "bbox": [140, 530, 160, 570], "confidence": 0.9},
    ]
    persons = [{"track_id": 3, "bbox": [110, 480, 140, 560], "confidence": 0.9}]
    result = _engine(cone_min_cluster_size=3).analyze(
        detections=cones, persons=persons, machines=[], frame_width=FW
    )
    assert result["statistics"]["danger_zones"] >= 1
    assert result["violation_counts"].get("DANGEROUS_ZONE") == 1


def test_result_exposes_perspective_config():
    result = _engine().analyze(detections=[], persons=[], machines=[], frame_width=FW)
    assert "use_perspective_distance" in result
    assert "machine_distance_threshold_m" in result
    assert "proximity_threshold_px" in result
