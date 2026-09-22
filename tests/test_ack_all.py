"""Tests for app.storage (Database ack_all_alerts method and API)."""

from app.storage import Database


def test_ack_all_alerts(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)

    # Insert test alerts
    id1 = db.insert_alert({"level": "HIGH", "title": "Alert 1", "message": "msg 1", "status": "new"})
    id2 = db.insert_alert({"level": "CRITICAL", "title": "Alert 2", "message": "msg 2", "status": "new"})
    id3 = db.insert_alert({"level": "HIGH", "title": "Alert 3", "message": "msg 3", "status": "acknowledged"})

    assert db.count_alerts("new") == 2

    count = db.ack_all_alerts()
    assert count == 2
    assert db.count_alerts("new") == 0
    assert db.count_alerts("acknowledged") == 3
