"""Tests for app.reports (period aggregates + CSV export)."""

import time

from app.reports import RANGES, build_report, events_to_csv


def _insert(db, ts, source="camera", risk="HIGH", counts=None, image=None, clip=None):
    return db.insert_event({
        "ts": ts,
        "source": source,
        "risk_level": risk,
        "violation_count": sum((counts or {}).values()),
        "persons": 1,
        "machines": 0,
        "zones": 0,
        "fire": 0,
        "smoke": 0,
        "ppe": 0,
        "violation_counts": counts or {},
        "summary": {},
        "evidence_image": image,
        "evidence_clip": clip,
    })


def test_ranges_are_defined():
    assert set(RANGES) == {"24h", "7d", "30d", "all"}
    assert RANGES["all"] is None


def test_build_report_totals_and_breakdowns(db):
    now = time.time()
    _insert(db, now, source="camera", risk="HIGH", counts={"FIRE": 1}, image="a.jpg")
    _insert(db, now - 5, source="camera", risk="CRITICAL", counts={"FIRE": 2}, clip="b.mp4")
    _insert(db, now - 10, source="video", risk="LOW", counts={"NO_HARDHAT": 1})
    db.increment_counter("frames_processed", 42)

    report = build_report(db, "all")

    assert report["range"] == "all"
    totals = report["totals"]
    assert totals["events"] == 3
    assert totals["frames_processed"] == 42
    assert totals["events_with_image"] == 1
    assert totals["events_with_clip"] == 1

    assert report["by_risk"]["HIGH"] == 1
    assert report["by_risk"]["CRITICAL"] == 1
    assert report["by_risk"]["LOW"] == 1
    assert report["by_risk"]["SAFE"] == 0

    assert report["by_source"] == {"camera": 2, "video": 1}
    # Violation types aggregate across events.
    assert report["by_violation_type"]["FIRE"] == 3
    assert report["by_violation_type"]["NO_HARDHAT"] == 1


def test_build_report_24h_excludes_old_events(db):
    now = time.time()
    _insert(db, now, risk="HIGH")                       # recent
    _insert(db, now - 10 * 24 * 3600, risk="HIGH")      # 10 days old

    report = build_report(db, "24h")
    assert report["totals"]["events"] == 1


def test_top_events_sorted_by_severity(db):
    now = time.time()
    _insert(db, now, risk="LOW", counts={"NO_MASK": 1})
    _insert(db, now - 1, risk="CRITICAL", counts={"FIRE": 1})
    _insert(db, now - 2, risk="HIGH", counts={"NO_HARDHAT": 1})

    report = build_report(db, "all")
    top = report["top_events"]
    assert top[0]["risk_level"] == "CRITICAL"
    assert top[1]["risk_level"] == "HIGH"
    assert top[2]["risk_level"] == "LOW"


def test_report_counts_alerts(db):
    now = time.time()
    _insert(db, now, risk="HIGH")
    db.insert_alert({"ts": now, "level": "HIGH", "title": "T", "message": "M", "status": "new"})
    db.insert_alert({"ts": now, "level": "HIGH", "title": "T", "message": "M", "status": "acknowledged"})

    totals = build_report(db, "all")["totals"]
    assert totals["alerts"] == 2
    assert totals["alerts_acknowledged"] == 1
    assert totals["alerts_unacknowledged"] == 1


def test_events_to_csv_has_header_and_rows(db):
    now = time.time()
    _insert(db, now, source="camera", risk="HIGH", counts={"FIRE": 1})
    _insert(db, now - 1, source="video", risk="LOW")

    csv_text = events_to_csv(db, "all")
    lines = [ln for ln in csv_text.splitlines() if ln.strip()]

    # Header + 2 data rows, sorted ascending by ts (oldest first).
    assert len(lines) == 3
    assert lines[0].startswith("id,ts,datetime,source,risk_level")
    # The video event (now - 1) is older, so it comes first.
    assert "video" in lines[1]
    # The camera event (now) is newer and carries the FIRE violation.
    assert "camera" in lines[2]
    assert "FIRE:1" in lines[2]


def test_events_to_csv_empty(db):
    csv_text = events_to_csv(db, "all")
    lines = [ln for ln in csv_text.splitlines() if ln.strip()]
    # Header only.
    assert len(lines) == 1
