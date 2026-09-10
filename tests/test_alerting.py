"""Tests for app.alerting (AlertManager policy, cooldown, dispatch)."""

from app.alerting import (
    AlertManager,
    describe_violations,
    headline_type,
)


class FakeNotifier:
    def __init__(self, succeed=True):
        self.name = "fake"
        self.succeed = succeed
        self.sent = []

    def send(self, text, image_path=None):
        self.sent.append((text, image_path))
        return self.succeed


def test_should_alert_respects_levels(settings_factory, db):
    settings = settings_factory()
    mgr = AlertManager(settings, db)
    assert mgr.should_alert("HIGH") is True
    assert mgr.should_alert("CRITICAL") is True
    assert mgr.should_alert("MEDIUM") is False
    assert mgr.should_alert("LOW") is False


def test_low_risk_never_records(settings_factory, db):
    settings = settings_factory()
    mgr = AlertManager(settings, db)
    alert = mgr.evaluate_and_record("camera", "MEDIUM", {"person_count": 1},
                                    {"NO_HARDHAT": 1})
    assert alert is None
    assert db.count_alerts() == 0


def test_cooldown_dedups_repeat_alerts(settings_factory, db):
    settings = settings_factory(alert_cooldown_s=1000.0)
    mgr = AlertManager(settings, db)

    first = mgr.evaluate_and_record("camera", "HIGH", {"person_count": 2},
                                    {"FIRE": 1}, event_id=7)
    assert first is not None
    assert first["id"] > 0
    assert first["level"] == "HIGH"
    assert first["event_id"] == 7

    # Same (source, risk) within the cooldown window is suppressed.
    second = mgr.evaluate_and_record("camera", "HIGH", {"person_count": 2},
                                     {"FIRE": 1}, event_id=8)
    assert second is None
    assert db.count_alerts() == 1


def test_cooldown_is_per_source_and_risk(settings_factory, db):
    settings = settings_factory(alert_cooldown_s=1000.0)
    mgr = AlertManager(settings, db)

    assert mgr.evaluate_and_record("camera", "HIGH", {}, {"FIRE": 1}) is not None
    # Different source is a different cooldown key.
    assert mgr.evaluate_and_record("video", "HIGH", {}, {"FIRE": 1}) is not None
    # Different risk on the same source is also a different key.
    assert mgr.evaluate_and_record("camera", "CRITICAL", {}, {"FIRE": 1}) is not None
    assert db.count_alerts() == 3


def test_cooldown_expires(settings_factory, db):
    settings = settings_factory(alert_cooldown_s=0.0)
    mgr = AlertManager(settings, db)
    assert mgr.evaluate_and_record("camera", "HIGH", {}, {"FIRE": 1}) is not None
    # Zero cooldown means the next one fires immediately.
    assert mgr.evaluate_and_record("camera", "HIGH", {}, {"FIRE": 1}) is not None
    assert db.count_alerts() == 2


def test_from_settings_without_telegram_has_no_notifiers(settings_factory, db):
    settings = settings_factory(telegram_bot_token="", telegram_chat_id="")
    mgr = AlertManager.from_settings(settings, db)
    assert mgr.notifiers == []
    assert settings.telegram_configured is False


def test_from_settings_with_telegram_adds_notifier(settings_factory, db):
    settings = settings_factory(telegram_bot_token="tok", telegram_chat_id="123")
    assert settings.telegram_configured is True
    mgr = AlertManager.from_settings(settings, db)
    assert len(mgr.notifiers) == 1
    assert mgr.notifiers[0].name == "telegram"


def test_telegram_disabled_flag_overrides_credentials(settings_factory, db):
    settings = settings_factory(telegram_bot_token="tok", telegram_chat_id="123",
                                telegram_enabled=False)
    assert settings.telegram_configured is False
    mgr = AlertManager.from_settings(settings, db)
    assert mgr.notifiers == []


def test_dispatch_marks_notified_on_success(settings_factory, db):
    settings = settings_factory()
    notifier = FakeNotifier(succeed=True)
    mgr = AlertManager(settings, db, notifiers=[notifier])

    alert = mgr.evaluate_and_record("camera", "HIGH", {"person_count": 1},
                                    {"FIRE": 1})
    mgr.dispatch(alert, image_path="/tmp/does-not-matter.jpg")

    assert len(notifier.sent) == 1
    stored = db.list_alerts()[0]
    assert stored["notified"] == 1


def test_dispatch_does_not_mark_notified_on_failure(settings_factory, db):
    settings = settings_factory()
    notifier = FakeNotifier(succeed=False)
    mgr = AlertManager(settings, db, notifiers=[notifier])

    alert = mgr.evaluate_and_record("camera", "HIGH", {}, {"FIRE": 1})
    mgr.dispatch(alert)

    stored = db.list_alerts()[0]
    assert stored["notified"] == 0


def test_dispatch_with_no_notifiers_is_noop(settings_factory, db):
    settings = settings_factory()
    mgr = AlertManager(settings, db, notifiers=[])
    alert = mgr.evaluate_and_record("camera", "HIGH", {}, {"FIRE": 1})
    # Must not raise and must not mark notified.
    mgr.dispatch(alert)
    assert db.list_alerts()[0]["notified"] == 0


def test_headline_type_picks_most_severe():
    assert headline_type({"NO_HARDHAT": 3, "FIRE": 1}) == "FIRE"
    assert headline_type({"SMOKE": 1, "NO_MASK": 5}) == "SMOKE"
    assert headline_type({}) is None


def test_describe_violations_orders_by_severity():
    text = describe_violations({"NO_HARDHAT": 2, "FIRE": 1})
    # Fire (severity 5) is described before the hardhat violation (severity 2).
    assert text.index("Fire") < text.index("hardhat")
    assert describe_violations({}) == "No violation detail available."
