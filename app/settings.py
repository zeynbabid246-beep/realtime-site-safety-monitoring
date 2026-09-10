"""
Application settings.

Everything tunable lives here and is read from environment variables (with
sane defaults), so secrets like the Telegram bot token are NEVER hardcoded.
A `.env` file in the project root is loaded automatically if python-dotenv is
installed (it is - see requirements.txt). See `.env.example` for the keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at runtime
    pass

BASE_DIR = Path(__file__).resolve().parent.parent

RISK_ORDER: Tuple[str, ...] = ("SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL")


def _risk_at_or_above(level: str) -> Tuple[str, ...]:
    """Return the risk levels at or above `level` (inclusive)."""
    try:
        idx = RISK_ORDER.index(level.upper())
    except ValueError:
        idx = RISK_ORDER.index("HIGH")
    return RISK_ORDER[idx:]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass
class Settings:
    # --- storage ---------------------------------------------------
    db_path: Path = field(default_factory=lambda: Path(os.environ.get("SAFETY_DB_PATH", BASE_DIR / "data" / "safety.db")))
    evidence_dir: Path = field(default_factory=lambda: Path(os.environ.get("SAFETY_EVIDENCE_DIR", BASE_DIR / "data" / "evidence")))

    # --- what gets recorded / alerted / captured -------------------
    # Risk levels that create a stored History event.
    record_risk_levels: Tuple[str, ...] = field(
        default_factory=lambda: _risk_at_or_above(os.environ.get("SAFETY_RECORD_MIN_RISK", "LOW"))
    )
    # Risk levels that fire an Alert (dashboard + Telegram).
    alert_risk_levels: Tuple[str, ...] = field(
        default_factory=lambda: _risk_at_or_above(os.environ.get("SAFETY_ALERT_MIN_RISK", "HIGH"))
    )
    # Risk levels that capture screenshot / clip evidence.
    evidence_risk_levels: Tuple[str, ...] = field(
        default_factory=lambda: _risk_at_or_above(os.environ.get("SAFETY_EVIDENCE_MIN_RISK", "HIGH"))
    )

    # --- cooldowns (seconds) - bound growth + notification spam ----
    event_cooldown_s: float = field(default_factory=lambda: _env_float("SAFETY_EVENT_COOLDOWN_S", 3.0))
    alert_cooldown_s: float = field(default_factory=lambda: _env_float("SAFETY_ALERT_COOLDOWN_S", 30.0))
    evidence_cooldown_s: float = field(default_factory=lambda: _env_float("SAFETY_EVIDENCE_COOLDOWN_S", 10.0))

    # --- evidence clip ---------------------------------------------
    clip_pre_frames: int = field(default_factory=lambda: _env_int("SAFETY_CLIP_PRE_FRAMES", 15))
    clip_post_frames: int = field(default_factory=lambda: _env_int("SAFETY_CLIP_POST_FRAMES", 30))
    clip_fps: float = field(default_factory=lambda: _env_float("SAFETY_CLIP_FPS", 10.0))

    # --- pruning ---------------------------------------------------
    max_evidence_files: int = field(default_factory=lambda: _env_int("SAFETY_MAX_EVIDENCE_FILES", 500))
    max_events: int = field(default_factory=lambda: _env_int("SAFETY_MAX_EVENTS", 20000))

    # --- Telegram (optional; disabled when either value is missing) --
    telegram_bot_token: str = field(default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN", "").strip())
    telegram_chat_id: str = field(default_factory=lambda: os.environ.get("TELEGRAM_CHAT_ID", "").strip())
    telegram_enabled: bool = field(default_factory=lambda: os.environ.get("SAFETY_TELEGRAM_ENABLED", "true").lower() != "false")

    @property
    def telegram_configured(self) -> bool:
        return bool(self.telegram_enabled and self.telegram_bot_token and self.telegram_chat_id)

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()
