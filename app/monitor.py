"""
SafetyMonitor - per-stream orchestrator that turns each processed frame into
durable History events, Evidence (snapshot + clip), and Alerts.

One SafetyMonitor per stream (camera connection / video job / image request);
it shares the app-level Database and AlertManager singletons but keeps its own
cooldown + clip-recorder state so streams never interfere.

`handle()` is synchronous and cheap on the DB side; the only network work
(Telegram) is pushed to a daemon thread so a slow API can never stall the frame
loop. Designed to be called from a worker thread (run_in_threadpool) or
directly inside the synchronous video pass.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from app.alerting import AlertManager
from app.evidence import ClipRecorder, save_snapshot
from app.settings import Settings
from app.storage import Database
from src.pipeline import summarize_result

logger = logging.getLogger("construction_safety.monitor")

_FRAME_FLUSH_EVERY = 10


class SafetyMonitor:
    def __init__(
        self,
        source: str,
        settings: Settings,
        db: Database,
        alert_manager: AlertManager,
        recorder: Optional[ClipRecorder] = None,
    ):
        self.source = source
        self.settings = settings
        self.db = db
        self.alert_manager = alert_manager
        self.recorder = recorder or ClipRecorder(
            settings.evidence_dir,
            source,
            pre_frames=settings.clip_pre_frames,
            post_frames=settings.clip_post_frames,
            fps=settings.clip_fps,
        )

        self._last_event_ts = 0.0
        self._last_evidence_ts = 0.0
        self._awaiting_clip_event_id: Optional[int] = None
        self._pending_frames = 0
        self._lock = threading.Lock()

    # ----------------------------------------------------------
    # cooldowns
    # ----------------------------------------------------------

    def _event_due(self, risk: str, now: float) -> bool:
        if risk not in self.settings.record_risk_levels:
            return False
        return (now - self._last_event_ts) >= self.settings.event_cooldown_s

    def _evidence_due(self, risk: str, now: float) -> bool:
        if risk not in self.settings.evidence_risk_levels:
            return False
        return (now - self._last_evidence_ts) >= self.settings.evidence_cooldown_s

    def _flush_frames(self) -> None:
        if self._pending_frames <= 0:
            return
        try:
            self.db.increment_counter("frames_processed", self._pending_frames)
            self.db.increment_counter(f"frames_{self.source}", self._pending_frames)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Counter flush failed: %s", exc.__class__.__name__)
        self._pending_frames = 0

    # ----------------------------------------------------------
    # main entry point
    # ----------------------------------------------------------

    def handle(self, result: Dict[str, Any], annotated: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Process one frame's engine result. Returns a small dict describing what
        was recorded (useful for tests/debugging):
            {"recorded": bool, "event_id": int|None, "alert": dict|None,
             "evidence_image": str|None, "risk_level": str}
        """
        now = time.time()
        summary = summarize_result(result)
        risk = summary.get("risk_level", "SAFE")
        violation_counts = result.get("violation_counts", {}) or {}

        with self._lock:
            self._pending_frames += 1

            # Keep the clip pre-buffer fresh every frame.
            if annotated is not None:
                try:
                    self.recorder.push(annotated)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Clip push failed: %s", exc.__class__.__name__)

            # Attach any clip that finished since the last frame.
            self._collect_finished_clips()

            due_event = self._event_due(risk, now)
            due_evidence = self._evidence_due(risk, now)
            record = due_event or due_evidence

            if not record:
                if self._pending_frames >= _FRAME_FLUSH_EVERY:
                    self._flush_frames()
                return {"recorded": False, "event_id": None, "alert": None,
                        "evidence_image": None, "risk_level": risk}

            # --- evidence (snapshot first, so the event row can reference it) ---
            snapshot_rel: Optional[str] = None
            if due_evidence:
                self._last_evidence_ts = now
                snapshot_rel = save_snapshot(annotated, self.settings.evidence_dir, self.source, risk)

            # --- event row ---
            event_id: Optional[int] = None
            self._last_event_ts = now
            try:
                event_id = self.db.insert_event(
                    {
                        "ts": now,
                        "source": self.source,
                        "risk_level": risk,
                        "violation_count": summary.get("violation_count", 0),
                        "persons": summary.get("person_count", 0),
                        "machines": summary.get("machine_count", 0),
                        "zones": summary.get("danger_zones", 0),
                        "fire": summary.get("fire_count", 0),
                        "smoke": summary.get("smoke_count", 0),
                        "ppe": summary.get("ppe_violations", 0),
                        "violation_counts": violation_counts,
                        "summary": summary,
                        "evidence_image": snapshot_rel,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Event insert failed: %s", exc.__class__.__name__)

            # --- arm clip recording (attaches to this event when finished) ---
            if due_evidence and event_id is not None:
                try:
                    if self.recorder.trigger():
                        self._awaiting_clip_event_id = event_id
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Clip trigger failed: %s", exc.__class__.__name__)

            self._flush_frames()

        # --- alerts (DB row synchronous; Telegram dispatch async) ---
        alert = None
        try:
            alert = self.alert_manager.evaluate_and_record(
                self.source, risk, summary, violation_counts, event_id
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Alert evaluation failed: %s", exc.__class__.__name__)

        if alert and self.alert_manager.notifiers:
            image_abs = (
                str(self.settings.evidence_dir / snapshot_rel) if snapshot_rel else None
            )
            self._dispatch_async(alert, image_abs)

        return {"recorded": True, "event_id": event_id, "alert": alert,
                "evidence_image": snapshot_rel, "risk_level": risk}

    # ----------------------------------------------------------
    # helpers
    # ----------------------------------------------------------

    def _collect_finished_clips(self) -> None:
        try:
            finished = self.recorder.take_finished()
        except Exception:  # noqa: BLE001
            return
        if not finished:
            return
        clip_rel = finished[-1]
        event_id = self._awaiting_clip_event_id
        if event_id is not None:
            try:
                self.db.update_event_evidence(event_id, None, clip_rel)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Clip attach failed: %s", exc.__class__.__name__)
            self._awaiting_clip_event_id = None

    def _dispatch_async(self, alert: Dict[str, Any], image_abs: Optional[str]) -> None:
        def _run() -> None:
            try:
                self.alert_manager.dispatch(alert, image_path=image_abs)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Alert dispatch failed: %s", exc.__class__.__name__)

        threading.Thread(target=_run, daemon=True).start()

    def close(self) -> None:
        with self._lock:
            self._flush_frames()
        try:
            self.recorder.close()
            # A clip finalised on close may still need attaching.
            self._collect_finished_clips()
        except Exception:  # noqa: BLE001
            pass
