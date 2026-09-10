"""Unit tests for src.fire.fire_confirmation (independent fire & smoke debounce)."""

from src.fire.fire_confirmation import FireConfirmationTracker, detection_kind


def _det(kind, confidence, bbox):
    return {"type": kind, "confidence": confidence, "bbox": list(bbox)}


def _tracker():
    return FireConfirmationTracker(
        min_confidence=0.45, min_area_px=800.0, min_consecutive_frames=3,
        smoke_min_confidence=0.20, smoke_min_area_px=200.0, smoke_min_consecutive_frames=2,
    )


# ------------------------------------------------------------------
# detection_kind
# ------------------------------------------------------------------

def test_detection_kind_reads_type_or_class_name():
    assert detection_kind({"type": "Fire"}) == "fire"
    assert detection_kind({"class_name": "Smoke"}) == "smoke"
    assert detection_kind({"type": "Person"}) == "other"


# ------------------------------------------------------------------
# fire gate (strict, persistence-based)
# ------------------------------------------------------------------

def test_fire_needs_consecutive_frames():
    tracker = _tracker()
    fire = _det("Fire", 0.60, [100, 100, 150, 150])  # area 2500
    assert tracker.update([fire]) == []               # streak 1
    assert tracker.update([fire]) == []               # streak 2
    confirmed = tracker.update([fire])                # streak 3 -> confirmed
    assert len(confirmed) == 1
    assert confirmed[0]["confirmed_streak"] == 3
    assert tracker.last_stats["confirmed_fire"] == 1


def test_low_confidence_fire_dropped():
    tracker = _tracker()
    fire = _det("Fire", 0.30, [100, 100, 150, 150])
    for _ in range(4):
        assert tracker.update([fire]) == []
    assert tracker.last_stats["dropped_confidence"] == 1


def test_small_area_fire_dropped():
    tracker = _tracker()
    fire = _det("Fire", 0.90, [100, 100, 120, 120])  # area 400 < 800
    for _ in range(4):
        assert tracker.update([fire]) == []
    assert tracker.last_stats["dropped_area"] >= 1


# ------------------------------------------------------------------
# smoke gate (looser)
# ------------------------------------------------------------------

def test_smoke_confirms_faster_than_fire():
    tracker = _tracker()
    smoke = _det("Smoke", 0.30, [200, 200, 230, 230])  # area 900 > 200
    assert tracker.update([smoke]) == []                # streak 1
    confirmed = tracker.update([smoke])                 # streak 2 -> confirmed
    assert len(confirmed) == 1
    assert tracker.last_stats["confirmed_smoke"] == 1


def test_smoke_tolerates_lower_confidence_than_fire():
    tracker = _tracker()
    # 0.30 would fail the fire gate but passes the smoke gate.
    smoke = _det("Smoke", 0.30, [200, 200, 240, 240])
    for _ in range(2):
        out = tracker.update([smoke])
    assert len(out) == 1


# ------------------------------------------------------------------
# stats + region matching
# ------------------------------------------------------------------

def test_stats_report_raw_counts():
    tracker = _tracker()
    raw = [_det("Fire", 0.60, [100, 100, 150, 150]), _det("Smoke", 0.30, [200, 200, 230, 230])]
    tracker.update(raw)
    stats = tracker.last_stats
    assert stats["raw"] == 2
    assert stats["raw_fire"] == 1
    assert stats["raw_smoke"] == 1
    assert stats["max_raw_fire_conf"] == 0.60


def test_non_overlapping_region_starts_new_streak():
    tracker = _tracker()
    a = _det("Fire", 0.60, [100, 100, 150, 150])
    b = _det("Fire", 0.60, [400, 400, 450, 450])  # far away, no IoU
    tracker.update([a])
    tracker.update([a])
    # Switching region resets persistence -> still not confirmed.
    assert tracker.update([b]) == []
