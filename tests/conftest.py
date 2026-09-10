"""Pytest configuration: make the project root importable as `src.*`."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def settings_factory(tmp_path):
    """
    Build a deterministic Settings pointed at a temp dir. Every tunable that
    the observability layer reads is pinned so tests never depend on the
    developer's environment / .env values.
    """
    from app.settings import Settings

    def _make(**overrides):
        base = dict(
            db_path=tmp_path / "safety.db",
            evidence_dir=tmp_path / "evidence",
            record_risk_levels=("LOW", "MEDIUM", "HIGH", "CRITICAL"),
            alert_risk_levels=("HIGH", "CRITICAL"),
            evidence_risk_levels=("HIGH", "CRITICAL"),
            event_cooldown_s=1000.0,
            alert_cooldown_s=1000.0,
            evidence_cooldown_s=1000.0,
            clip_pre_frames=2,
            clip_post_frames=2,
            clip_fps=10.0,
            max_evidence_files=100,
            max_events=1000,
            telegram_bot_token="",
            telegram_chat_id="",
        )
        base.update(overrides)
        settings = Settings(**base)
        settings.ensure_dirs()
        return settings

    return _make


@pytest.fixture
def db(tmp_path):
    """A fresh, isolated Database instance."""
    from app.storage import Database

    database = Database(tmp_path / "test.db")
    try:
        yield database
    finally:
        database.close()


@pytest.fixture
def frame():
    """A tiny BGR frame for evidence/monitor tests."""
    import numpy as np

    return np.zeros((48, 64, 3), dtype=np.uint8)


def make_result(risk="HIGH", violation_counts=None, persons=1, machines=0, zones=0):
    """Construct an engine-style result dict that summarize_result understands."""
    return {
        "risk_level": risk,
        "violation_count": sum((violation_counts or {}).values()),
        "violation_counts": violation_counts or {},
        "statistics": {"persons": persons, "machines": machines, "danger_zones": zones},
        "fire_detections": [],
    }

