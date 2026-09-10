"""Tests for app.monitor (SafetyMonitor orchestration)."""

from app.alerting import AlertManager
from app.monitor import SafetyMonitor
from conftest import make_result


class FakeRecorder:
    """Stands in for ClipRecorder so monitor tests never touch cv2 encoding."""

    def __init__(self):
        self.pushed = 0
        self.triggered = False
        self.closed = False
        self._finished = []

    def push(self, frame):
        self.pushed += 1

    def trigger(self):
        self.triggered = True
        return True

    def take_finished(self):
        finished, self._finished = self._finished, []
        return finished

    def queue_finished(self, rel_path):
        self._finished.append(rel_path)

    def close(self):
        self.closed = True

    @property
    def recording(self):
        return self.triggered


def make_monitor(settings, db, recorder=None):
    mgr = AlertManager(settings, db, notifiers=[])
    return SafetyMonitor("camera", settings, db, mgr, recorder=recorder or FakeRecorder())


def test_high_risk_records_event_alert_and_evidence(settings_factory, db, frame):
    settings = settings_factory()
    monitor = make_monitor(settings, db)

    out = monitor.handle(make_result("HIGH", {"FIRE": 1}, persons=2), frame)

    assert out["recorded"] is True
    assert out["risk_level"] == "HIGH"
    assert out["event_id"] is not None
    assert out["alert"] is not None
    # HIGH is in evidence_risk_levels -> a snapshot was written.
    assert out["evidence_image"] is not None
    assert (settings.evidence_dir / out["evidence_image"]).exists()

    # Persisted rows.
    assert len(db.list_events()) == 1
    assert db.count_alerts() == 1
    stored = db.list_events()[0]
    assert stored["risk_level"] == "HIGH"
    assert stored["persons"] == 2
    assert stored["evidence_image"] == out["evidence_image"]


def test_safe_frame_records_nothing(settings_factory, db, frame):
    settings = settings_factory()
    monitor = make_monitor(settings, db)

    out = monitor.handle(make_result("SAFE", {}), frame)

    assert out["recorded"] is False
    assert out["event_id"] is None
    assert out["alert"] is None
    assert db.list_events() == []
    assert db.count_alerts() == 0
    # The clip pre-buffer is still fed every frame.
    assert monitor.recorder.pushed == 1


def test_event_cooldown_suppresses_second_record(settings_factory, db, frame):
    settings = settings_factory(event_cooldown_s=1000.0, evidence_cooldown_s=1000.0)
    monitor = make_monitor(settings, db)

    first = monitor.handle(make_result("HIGH", {"FIRE": 1}), frame)
    second = monitor.handle(make_result("HIGH", {"FIRE": 1}), frame)

    assert first["recorded"] is True
    assert second["recorded"] is False
    assert len(db.list_events()) == 1


def test_medium_risk_records_event_but_no_alert_or_evidence(settings_factory, db, frame):
    settings = settings_factory()  # record>=LOW, alert>=HIGH, evidence>=HIGH
    monitor = make_monitor(settings, db)

    out = monitor.handle(make_result("MEDIUM", {"NO_HARDHAT": 1}), frame)

    assert out["recorded"] is True
    assert out["event_id"] is not None
    assert out["alert"] is None            # MEDIUM < HIGH -> no alert
    assert out["evidence_image"] is None   # MEDIUM < HIGH -> no snapshot
    assert len(db.list_events()) == 1
    assert db.count_alerts() == 0


def test_finished_clip_attaches_to_awaiting_event(settings_factory, db, frame):
    settings = settings_factory()
    recorder = FakeRecorder()
    monitor = make_monitor(settings, db, recorder=recorder)

    first = monitor.handle(make_result("HIGH", {"FIRE": 1}), frame)
    event_id = first["event_id"]
    assert recorder.triggered is True

    # The clip finishes asynchronously; on the next handle it gets attached.
    recorder.queue_finished("2026-01-01/camera_clip.mp4")
    monitor.handle(make_result("SAFE", {}), frame)

    stored = next(e for e in db.list_events() if e["id"] == event_id)
    assert stored["evidence_clip"] == "2026-01-01/camera_clip.mp4"


def test_frame_counters_flush_on_close(settings_factory, db, frame):
    settings = settings_factory()
    monitor = make_monitor(settings, db)

    # Three SAFE frames: recorded but below the flush threshold (10).
    for _ in range(3):
        monitor.handle(make_result("SAFE", {}), frame)
    assert db.get_counter("frames_processed") == 0

    monitor.close()
    assert db.get_counter("frames_processed") == 3
    assert db.get_counter("frames_camera") == 3
    assert monitor.recorder.closed is True


def test_counters_flush_at_threshold(settings_factory, db, frame):
    settings = settings_factory()
    monitor = make_monitor(settings, db)

    for _ in range(10):
        monitor.handle(make_result("SAFE", {}), frame)

    # The 10th frame hits _FRAME_FLUSH_EVERY and flushes without close().
    assert db.get_counter("frames_processed") == 10
