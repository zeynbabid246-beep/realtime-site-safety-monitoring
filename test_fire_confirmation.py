"""Fire vs smoke confirmation must not share one threshold."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.fire.fire_confirmation import FireConfirmationTracker


def _box(kind, conf, x1, y1, x2, y2):
    return {
        "class_name": kind,
        "type": kind.lower(),
        "confidence": conf,
        "bbox": [x1, y1, x2, y2],
        "class_id": 1 if kind == "Smoke" else 0,
    }


def test_tiny_smoke_is_not_killed_by_fire_area_gate():
    tracker = FireConfirmationTracker()
    # ~20x15 = 300px^2: below fire min_area (800) but above smoke min_area (200)
    smoke = _box("Smoke", 0.28, 10, 10, 30, 25)

    assert tracker.update([smoke]) == []  # frame 1, persistence
    confirmed = tracker.update([smoke])
    assert len(confirmed) == 1
    assert confirmed[0]["type"] == "smoke"
    assert tracker.last_stats["confirmed_smoke"] == 1


def test_low_conf_fire_still_rejected():
    tracker = FireConfirmationTracker()
    fire = _box("Fire", 0.30, 0, 0, 80, 80)  # area 6400, but conf 0.30 < 0.45
    for _ in range(6):
        assert tracker.update([fire]) == []
    assert tracker.last_stats["dropped_confidence"] == 1


def test_person_filters_do_not_apply_here():
    # Sanity: this module never looks at Person boxes.
    tracker = FireConfirmationTracker()
    personish = {
        "class_name": "Person",
        "type": "person",
        "confidence": 0.99,
        "bbox": [0, 0, 5, 5],
        "class_id": 99,
    }
    for _ in range(6):
        tracker.update([personish])
    assert tracker.last_stats["confirmed"] == 0


def main():
    test_tiny_smoke_is_not_killed_by_fire_area_gate()
    test_low_conf_fire_still_rejected()
    test_person_filters_do_not_apply_here()
    print("test_fire_confirmation: PASS")


if __name__ == "__main__":
    main()
