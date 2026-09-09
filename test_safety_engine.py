"""
Integration test for the Construction Safety Engine (v2).

Tests:
    1. PPE violations
    2. Safety-cone extraction
    3. HDBSCAN cone clustering
    4. Dangerous-zone polygon generation
    5. Person inside dangerous zone
    6. Person-machine distance
    7. Machinery-utility-pole proximity
    8. Fire / smoke violations
    9. Danger-zone id stability across frames (DangerZoneTracker)
    10. Violation aggregation
    11. Global risk-level calculation (including the severity-based fix)

YOLO detection format:
    [x1, y1, x2, y2, confidence, class_id]

Class IDs:
    0  Hardhat
    1  Mask
    2  NO-Hardhat
    3  NO-Mask
    4  NO-Safety Vest
    5  Person
    6  Safety Cone
    7  Safety Vest
    8  machinery
    9  utility pole
    10 vehicle

NOTE: This replaces the original test_safety_engine.py, whose
assertions assumed `result["danger_zones"]` was a raw list of Shapely
Polygons. In this version, SafetyEngine.analyze() always returns
`danger_zones` as a list of dicts:

    {"zone_id": int, "polygon": Polygon, "age": int, "missed": int}

so that a persistent zone_id survives across frames when a
DangerZoneTracker is attached to the engine.
"""

from pprint import pprint

from src.safety.safety_engine import SafetyEngine
from src.safety.rules import SafetyConfig, calculate_risk_level
from src.safety.geometry import DangerZoneTracker


# ============================================================
# CONFIGURATION
# ============================================================

config = SafetyConfig(
    machine_distance_threshold=250.0,
    pole_distance_threshold=250.0,
    violation_confidence=0.30,
    cone_min_cluster_size=3,
    cone_min_samples=1,
)


# ============================================================
# SYNTHETIC YOLO DETECTIONS
# ============================================================
#
# Person 1: bbox = [300, 400, 400, 600] -> bottom-center (350, 600)
# Danger zone: x = 150 -> 650, y = 500 -> 800
# Therefore Person 1 is INSIDE the danger zone.
# ============================================================

detections = [
    [300, 400, 400, 600, 0.95, 5],   # PERSON 1
    [325, 400, 375, 460, 0.92, 2],   # NO-HARDHAT for person 1
    [315, 470, 385, 560, 0.88, 4],   # NO-SAFETY-VEST for person 1
    [800, 300, 900, 500, 0.91, 5],   # PERSON 2 (outside zone)
    [125, 450, 175, 500, 0.90, 6],   # cones forming a rectangular zone
    [625, 450, 675, 500, 0.91, 6],
    [625, 750, 675, 800, 0.89, 6],
    [125, 750, 175, 800, 0.93, 6],
    [400, 300, 500, 500, 0.94, 8],   # MACHINERY (~141px from person 1)
    [530, 300, 570, 500, 0.90, 9],   # UTILITY POLE (~100px from machinery)
]

persons = [
    {"track_id": 1, "bbox": [300, 400, 400, 600], "confidence": 0.95},
    {"track_id": 2, "bbox": [800, 300, 900, 500], "confidence": 0.91},
]

machines = [
    {
        "class_name": "machinery",
        "class_id": 8,
        "confidence": 0.94,
        "bbox": [400, 300, 500, 500],
    },
]

fire_detections_none = []

fire_detections_fire = [
    {
        "class_id": 0,
        "class_name": "Fire",
        "type": "fire",
        "confidence": 0.80,
        "bbox": [10, 10, 60, 60],
    }
]

fire_detections_smoke = [
    {
        "class_id": 1,
        "class_name": "Smoke",
        "type": "smoke",
        "confidence": 0.55,
        "bbox": [10, 10, 60, 60],
    }
]


# ============================================================
# START TEST
# ============================================================

print()
print("=" * 70)
print("SAFETY ENGINE TEST (v2 - fire integration + zone stabilization)")
print("=" * 70)

print()
print("Creating SafetyEngine with a DangerZoneTracker...")

zone_tracker = DangerZoneTracker()
engine = SafetyEngine(config=config, zone_tracker=zone_tracker)


# ============================================================
# FRAME 1: BASELINE SCENE, NO FIRE
# ============================================================

print()
print("Running SafetyEngine.analyze() - frame 1 (no fire)...")

result = engine.analyze(
    detections=detections,
    persons=persons,
    machines=machines,
    fire_detections=fire_detections_none,
)

print()
print("=" * 70)
print("RESULT - FRAME 1")
print("=" * 70)
print(f"Risk level: {result['risk_level']}")
print(f"Violation count: {result['violation_count']}")

print()
print("STATISTICS")
pprint(result["statistics"])

zones = result["danger_zones"]
print()
print(f"Number of zones: {len(zones)}")
for zone in zones:
    polygon = zone["polygon"]
    print(f"  zone_id={zone['zone_id']} age={zone['age']} missed={zone['missed']} "
          f"area={polygon.area:.2f} valid={polygon.is_valid}")

people_inside = result["people_inside_zones"]
print()
print(f"People inside zones: {len(people_inside)}")
for item in people_inside:
    print(f"  person_index={item.get('person_index')} zone_index={item.get('zone_index')} "
          f"zone_id={item.get('zone_id')} track_id={item.get('track_id')} point={item.get('point')}")

violations = result["violations"]
print()
print(f"Total violations: {len(violations)}")
for index, violation in enumerate(violations, start=1):
    print(f"  {index}. {violation.get('type')} (severity={violation.get('severity')}) "
          f"track_id={violation.get('track_id')} - {violation.get('message')}")


# ============================================================
# ASSERTIONS - FRAME 1
# ============================================================

print()
print("=" * 70)
print("ASSERTIONS - FRAME 1")
print("=" * 70)

assert len(zones) >= 1, "FAIL: No dangerous zone was generated."
print("[PASS] Dangerous zone generated")

for zone in zones:
    assert zone["polygon"].is_valid, "FAIL: Generated danger-zone polygon is invalid."
print("[PASS] Danger-zone polygon is valid")

assert len(people_inside) >= 1, "FAIL: No person detected inside dangerous zone."
print("[PASS] Person detected inside dangerous zone")

track_ids_inside = {item.get("track_id") for item in people_inside}
assert 1 in track_ids_inside, "FAIL: Person 1 should be inside the dangerous zone."
print("[PASS] Person 1 is inside dangerous zone")

assert 2 not in track_ids_inside, "FAIL: Person 2 should be outside the dangerous zone."
print("[PASS] Person 2 is outside dangerous zone")

distance_results = result["distance_results"]
assert len(distance_results) >= 1, "FAIL: No person-machine distance was calculated."
print("[PASS] Person-machine distance calculated")

person_1_distances = [d for d in distance_results if d.get("track_id") == 1]
assert len(person_1_distances) >= 1, "FAIL: No distance result for Person 1."
person_1_distance = person_1_distances[0].get("distance_px")
assert person_1_distance is not None, "FAIL: Person 1 distance_px is missing."
assert person_1_distance <= config.machine_distance_threshold, (
    "FAIL: Person 1 should be within machine proximity threshold."
)
print("[PASS] Person 1 is too close to machinery")

pole_results = result["pole_results"]
assert len(pole_results) >= 1, "FAIL: No machinery-pole proximity was calculated."
print("[PASS] Machinery-pole distance calculated")

actual_types = {v.get("type") for v in violations}
expected_types = {
    "NO_HARDHAT",
    "NO_SAFETY_VEST",
    "DANGEROUS_ZONE",
    "MACHINE_PROXIMITY",
    "POLE_PROXIMITY",
}
for violation_type in sorted(expected_types):
    assert violation_type in actual_types, f"FAIL: Missing violation type: {violation_type}"
    print(f"[PASS] {violation_type} violation detected")

assert result["risk_level"] == "CRITICAL", (
    f"FAIL: Expected risk level CRITICAL, got {result['risk_level']}"
)
print("[PASS] Risk level = CRITICAL (dangerous-zone + machine-proximity combo)")


# ============================================================
# FRAME 2: SAME SCENE - ZONE ID SHOULD BE STABLE
# ============================================================

print()
print("=" * 70)
print("FRAME 2 - zone id stability check")
print("=" * 70)

result_2 = engine.analyze(
    detections=detections,
    persons=persons,
    machines=machines,
    fire_detections=fire_detections_none,
)

zone_ids_1 = sorted(z["zone_id"] for z in result["danger_zones"])
zone_ids_2 = sorted(z["zone_id"] for z in result_2["danger_zones"])

assert zone_ids_1 == zone_ids_2, (
    f"FAIL: zone ids should be stable across frames, got {zone_ids_1} vs {zone_ids_2}"
)
print(f"[PASS] Zone ids stable across frames: {zone_ids_2}")
print(f"       zone age is now {result_2['danger_zones'][0]['age']} (should be 2)")
assert result_2["danger_zones"][0]["age"] == 2


# ============================================================
# FRAME 3: FIRE ALONE -> CRITICAL
# ============================================================

print()
print("=" * 70)
print("FRAME 3 - fire alone should be CRITICAL")
print("=" * 70)

result_fire = engine.analyze(
    detections=[], persons=[], machines=[], fire_detections=fire_detections_fire
)
print(f"Risk level: {result_fire['risk_level']}")
assert result_fire["risk_level"] == "CRITICAL", "FAIL: Fire alone should be CRITICAL."
print("[PASS] Fire alone => CRITICAL")


# ============================================================
# FRAME 4: SMOKE ALONE -> HIGH (not CRITICAL)
# ============================================================

print()
print("=" * 70)
print("FRAME 4 - smoke alone should be HIGH, not CRITICAL")
print("=" * 70)

result_smoke = engine.analyze(
    detections=[], persons=[], machines=[], fire_detections=fire_detections_smoke
)
print(f"Risk level: {result_smoke['risk_level']}")
assert result_smoke["risk_level"] == "HIGH", "FAIL: Smoke alone should be HIGH."
print("[PASS] Smoke alone => HIGH")


# ============================================================
# FRAME 5: REGRESSION TEST FOR THE ORIGINAL RISK-LEVEL BUG
# ============================================================
#
# Original bug: 1 MEDIUM violation + 1 HIGH violation (2 violations
# total) resolved to MEDIUM because the old logic checked
# `len(violations) == 2` before it checked per-violation severity.
# The fixed calculate_risk_level() derives risk from actual severity.
# ============================================================

print()
print("=" * 70)
print("FRAME 5 - risk-level severity bug regression test")
print("=" * 70)

regression_violations = [
    {"type": "NO_HARDHAT", "severity": "MEDIUM"},
    {"type": "POLE_PROXIMITY", "severity": "HIGH"},
]
regression_risk = calculate_risk_level(regression_violations)
print(f"1 MEDIUM + 1 HIGH violation => {regression_risk}")
assert regression_risk == "HIGH", (
    f"FAIL: expected HIGH, got {regression_risk} (this was the original bug)"
)
print("[PASS] Severity-based risk calculation fixed")


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("ALL SAFETY ENGINE TESTS PASSED")
print("=" * 70)
print()
print("Verified pipeline:")
print("hazard detections + fire/smoke detections")
print("    -> cones -> HDBSCAN -> danger-zone polygons -> DangerZoneTracker")
print("    -> person bottom-center -> point-in-polygon")
print("    -> person <-> machinery distance")
print("    -> PPE violations")
print("    -> machinery <-> utility pole")
print("    -> fire / smoke violations")
print("    -> violation aggregation")
print("    -> severity-based risk level")
print()
print("SAFE / LOW / MEDIUM / HIGH / CRITICAL")
print()
print("TEST COMPLETED")
