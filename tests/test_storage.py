"""Tests for app.storage (SQLite persistence layer)."""

import time


def test_insert_and_list_event_roundtrip(db):
    eid = db.insert_event({
        "ts": 1000.0,
        "source": "camera",
        "risk_level": "HIGH",
        "violation_count": 2,
        "persons": 3,
        "machines": 1,
        "zones": 1,
        "fire": 0,
        "smoke": 1,
        "ppe": 2,
        "violation_counts": {"NO_HARDHAT": 1, "SMOKE": 1},
        "summary": {"risk_level": "HIGH"},
        "evidence_image": "2026/camera_x_HIGH.jpg",
    })
    assert eid > 0

    events = db.list_events()
    assert len(events) == 1
    e = events[0]
    assert e["id"] == eid
    assert e["risk_level"] == "HIGH"
    assert e["persons"] == 3
    # JSON columns are parsed back into dicts.
    assert e["violation_counts"] == {"NO_HARDHAT": 1, "SMOKE": 1}
    assert e["summary"] == {"risk_level": "HIGH"}
    assert e["evidence_image"] == "2026/camera_x_HIGH.jpg"
    assert e["evidence_clip"] is None


def test_update_event_evidence_uses_coalesce(db):
    eid = db.insert_event({"ts": 1.0, "source": "video", "risk_level": "HIGH",
                           "evidence_image": "img.jpg"})
    # Passing image=None must NOT wipe the existing image; clip gets set.
    db.update_event_evidence(eid, None, "clip.mp4")
    e = db.list_events()[0]
    assert e["evidence_image"] == "img.jpg"
    assert e["evidence_clip"] == "clip.mp4"


def test_list_events_filters_and_ordering(db):
    db.insert_event({"ts": 10.0, "source": "camera", "risk_level": "LOW"})
    db.insert_event({"ts": 20.0, "source": "video", "risk_level": "HIGH"})
    db.insert_event({"ts": 30.0, "source": "camera", "risk_level": "CRITICAL"})

    # Newest first by default.
    all_events = db.list_events()
    assert [e["ts"] for e in all_events] == [30.0, 20.0, 10.0]

    assert len(db.list_events(risk="HIGH")) == 1
    assert len(db.list_events(source="camera")) == 2
    assert len(db.list_events(limit=2)) == 2
    assert len(db.list_events(since=15.0)) == 2


def test_counters_upsert_and_default(db):
    assert db.get_counter("missing") == 0
    db.increment_counter("frames_processed", 5)
    db.increment_counter("frames_processed", 3)
    assert db.get_counter("frames_processed") == 8


def test_prune_events_keeps_newest(db):
    for i in range(10):
        db.insert_event({"ts": float(i), "source": "camera", "risk_level": "LOW"})
    db.prune_events(keep=4)
    remaining = db.list_events(limit=100)
    assert len(remaining) == 4
    # The four newest timestamps survive.
    assert sorted(e["ts"] for e in remaining) == [6.0, 7.0, 8.0, 9.0]


def test_events_since_filters_by_timestamp(db):
    db.insert_event({"ts": 100.0, "source": "camera", "risk_level": "LOW"})
    db.insert_event({"ts": 200.0, "source": "camera", "risk_level": "LOW"})
    got = db.events_since(150.0)
    assert len(got) == 1
    assert got[0]["ts"] == 200.0
    # events_since returns ascending order.
    assert got == sorted(got, key=lambda e: e["ts"])


def test_ack_alert_and_counts(db):
    aid = db.insert_alert({"ts": time.time(), "level": "HIGH", "title": "T",
                           "message": "M", "status": "new"})
    assert db.count_alerts("new") == 1
    assert db.ack_alert(aid) is True
    # Second ack is a no-op (returns False because status already changed).
    assert db.ack_alert(aid) is False
    assert db.count_alerts("new") == 0
    assert db.count_alerts("acknowledged") == 1


def test_ack_all_alerts(db):
    db.insert_alert({"ts": time.time(), "level": "HIGH", "title": "T1", "message": "M1", "status": "new"})
    db.insert_alert({"ts": time.time(), "level": "CRITICAL", "title": "T2", "message": "M2", "status": "new"})
    assert db.count_alerts("new") == 2
    count = db.ack_all_alerts()
    assert count == 2
    assert db.count_alerts("new") == 0
    assert db.count_alerts("acknowledged") == 2
