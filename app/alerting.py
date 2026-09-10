"""
Alerting engine.

An AlertManager decides WHEN an alert fires (risk level + cooldown/dedup),
records it to SQLite (the dashboard feed reads from there), and dispatches it
to pluggable notifiers. Ships with a Telegram notifier; the dashboard "notifier"
is implicit (the alert row itself is the feed).

Secrets (bot token, chat id) come only from settings/env and are never logged.
Telegram dispatch is best-effort and runs off the event loop - a network
failure can never break the frame pipeline.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.settings import Settings

logger = logging.getLogger("construction_safety.alerting")

# Severity ordering used to pick the "headline" violation for an alert.
_TYPE_SEVERITY = {
    "FIRE": 5,
    "DANGEROUS_ZONE": 4,
    "MACHINE_PROXIMITY": 4,
    "POLE_PROXIMITY": 3,
    "SMOKE": 3,
    "NO_HARDHAT": 2,
    "NO_SAFETY_VEST": 2,
    "NO_MASK": 1,
}

_TYPE_LABELS = {
    "FIRE": "Fire detected",
    "SMOKE": "Smoke detected",
    "DANGEROUS_ZONE": "Worker in danger zone",
    "MACHINE_PROXIMITY": "Worker too close to machinery",
    "POLE_PROXIMITY": "Machinery too close to utility pole",
    "NO_HARDHAT": "Missing hardhat",
    "NO_SAFETY_VEST": "Missing safety vest",
    "NO_MASK": "Missing mask",
}


def describe_violations(violation_counts: Dict[str, int]) -> str:
    """Human-readable summary of the violation counts, worst-first."""
    if not violation_counts:
        return "No violation detail available."
    ordered = sorted(
        violation_counts.items(),
        key=lambda kv: _TYPE_SEVERITY.get(kv[0], 0),
        reverse=True,
    )
    parts = [f"{count}x {_TYPE_LABELS.get(vtype, vtype)}" for vtype, count in ordered]
    return "; ".join(parts)


def headline_type(violation_counts: Dict[str, int]) -> Optional[str]:
    """The most severe violation type present (used for alert title + dedup)."""
    if not violation_counts:
        return None
    return max(violation_counts.items(), key=lambda kv: _TYPE_SEVERITY.get(kv[0], 0))[0]


# --------------------------------------------------------------
# Notifiers
# --------------------------------------------------------------

class TelegramNotifier:
    """Sends alert text (and an evidence photo when available) via the Bot API."""

    def __init__(self, token: str, chat_id: str, timeout: float = 10.0):
        self._token = token
        self._chat_id = chat_id
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "telegram"

    def send(self, text: str, image_path: Optional[str] = None) -> bool:
        try:
            import requests
        except Exception:  # noqa: BLE001
            logger.warning("requests not available - Telegram alert skipped")
            return False

        # NOTE: the token is part of the URL - never log the URL itself.
        if image_path and Path(image_path).exists():
            url = f"https://api.telegram.org/bot{self._token}/sendPhoto"
            try:
                with open(image_path, "rb") as photo:
                    resp = requests.post(
                        url,
                        data={"chat_id": self._chat_id, "caption": text[:1000]},
                        files={"photo": photo},
                        timeout=self._timeout,
                    )
                if resp.status_code == 200:
                    return True
                logger.warning("Telegram sendPhoto failed: HTTP %s", resp.status_code)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Telegram sendPhoto error: %s", exc.__class__.__name__)
            # fall through to a text-only alert if the photo failed

        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        try:
            resp = requests.post(
                url,
                json={"chat_id": self._chat_id, "text": text},
                timeout=self._timeout,
            )
            if resp.status_code == 200:
                return True
            logger.warning("Telegram sendMessage failed: HTTP %s", resp.status_code)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Telegram sendMessage error: %s", exc.__class__.__name__)
        return False


# --------------------------------------------------------------
# Alert manager
# --------------------------------------------------------------

class AlertManager:
    """
    Decides when to alert, records alerts to the DB, and dispatches to
    notifiers. Cooldown state is per (source, risk_level) so an ongoing
    incident produces at most one alert per `alert_cooldown_s`.
    """

    def __init__(self, settings: Settings, db, notifiers: Optional[List[Any]] = None):
        self.settings = settings
        self.db = db
        self.notifiers: List[Any] = notifiers or []
        self._cooldowns: Dict[tuple, float] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings, db) -> "AlertManager":
        notifiers: List[Any] = []
        if settings.telegram_configured:
            notifiers.append(
                TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id)
            )
            logger.info("Telegram alerting enabled")
        else:
            logger.info("Telegram alerting disabled (not configured)")
        return cls(settings, db, notifiers)

    def should_alert(self, risk_level: str) -> bool:
        return risk_level in self.settings.alert_risk_levels

    def _on_cooldown(self, key: tuple, now: float) -> bool:
        with self._lock:
            last = self._cooldowns.get(key)
            if last is not None and (now - last) < self.settings.alert_cooldown_s:
                return True
            self._cooldowns[key] = now
            return False

    def evaluate_and_record(
        self,
        source: str,
        risk_level: str,
        summary: Dict[str, Any],
        violation_counts: Dict[str, int],
        event_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Synchronous: apply the alert policy + cooldown and, if it fires,
        insert the alert row. Returns the stored alert dict (with id) or None.
        Does NOT perform network I/O - call dispatch() for that.
        """
        if not self.should_alert(risk_level):
            return None

        now = time.time()
        if self._on_cooldown((source, risk_level), now):
            return None

        head = headline_type(violation_counts)
        title = _TYPE_LABELS.get(head, f"{risk_level} risk") if head else f"{risk_level} risk"
        message = (
            f"[{risk_level}] {title} on {source}. "
            f"{describe_violations(violation_counts)}. "
            f"persons={summary.get('person_count', 0)} "
            f"machines={summary.get('machine_count', 0)} "
            f"zones={summary.get('danger_zones', 0)}"
        )

        alert_id = self.db.insert_alert(
            {
                "ts": now,
                "event_id": event_id,
                "level": risk_level,
                "title": title,
                "message": message,
                "channel": "dashboard",
                "status": "new",
                "notified": 0,
            }
        )
        return {
            "id": alert_id,
            "level": risk_level,
            "title": title,
            "message": message,
            "event_id": event_id,
        }

    def dispatch(self, alert: Dict[str, Any], image_path: Optional[str] = None) -> None:
        """
        Synchronous, best-effort fan-out to external notifiers (network I/O).
        Callers should run this off the event loop (run_in_threadpool) and
        never let its failure affect the pipeline.
        """
        if not self.notifiers:
            return

        text = f"🚨 {alert['message']}"
        sent_any = False
        for notifier in self.notifiers:
            try:
                if notifier.send(text, image_path=image_path):
                    sent_any = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("Notifier %s failed: %s", getattr(notifier, "name", "?"), exc.__class__.__name__)

        if sent_any and alert.get("id") is not None:
            try:
                self.db.mark_alert_notified(alert["id"], True)
            except Exception:  # noqa: BLE001
                pass
