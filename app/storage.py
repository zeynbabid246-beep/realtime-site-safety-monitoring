"""
SQLite persistence for events, alerts, and counters.

Thread-safe by design: a single connection opened with
`check_same_thread=False` guarded by an RLock, in WAL mode so readers do not
block the writer. Writes happen from the asyncio threadpool (video jobs) and
from async endpoints alike; the lock serialises them safely.

Schema is intentionally flat and denormalised (per-event risk/violation
counters stored as columns + a JSON blob) so the Statistics and Reports
endpoints can aggregate without joins.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("construction_safety.storage")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL    NOT NULL,
    source          TEXT    NOT NULL,
    risk_level      TEXT    NOT NULL,
    violation_count INTEGER NOT NULL DEFAULT 0,
    persons         INTEGER NOT NULL DEFAULT 0,
    machines        INTEGER NOT NULL DEFAULT 0,
    zones           INTEGER NOT NULL DEFAULT 0,
    fire            INTEGER NOT NULL DEFAULT 0,
    smoke           INTEGER NOT NULL DEFAULT 0,
    ppe             INTEGER NOT NULL DEFAULT 0,
    violation_counts_json TEXT,
    summary_json    TEXT,
    evidence_image  TEXT,
    evidence_clip   TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_risk ON events(risk_level);

CREATE TABLE IF NOT EXISTS alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    event_id    INTEGER,
    level       TEXT    NOT NULL,
    title       TEXT    NOT NULL,
    message     TEXT    NOT NULL,
    channel     TEXT    NOT NULL DEFAULT 'dashboard',
    status      TEXT    NOT NULL DEFAULT 'new',
    notified    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(ts);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);

CREATE TABLE IF NOT EXISTS counters (
    name  TEXT PRIMARY KEY,
    value INTEGER NOT NULL DEFAULT 0
);
"""


class Database:
    """A small, lock-guarded wrapper around one SQLite connection."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass

    # ----------------------------------------------------------
    # events
    # ----------------------------------------------------------

    def insert_event(self, event: Dict[str, Any]) -> int:
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO events (
                    ts, source, risk_level, violation_count, persons, machines,
                    zones, fire, smoke, ppe, violation_counts_json, summary_json,
                    evidence_image, evidence_clip
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    float(event.get("ts", time.time())),
                    event.get("source", "unknown"),
                    event.get("risk_level", "SAFE"),
                    int(event.get("violation_count", 0)),
                    int(event.get("persons", 0)),
                    int(event.get("machines", 0)),
                    int(event.get("zones", 0)),
                    int(event.get("fire", 0)),
                    int(event.get("smoke", 0)),
                    int(event.get("ppe", 0)),
                    json.dumps(event.get("violation_counts", {}) or {}),
                    json.dumps(event.get("summary", {}) or {}),
                    event.get("evidence_image"),
                    event.get("evidence_clip"),
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def update_event_evidence(self, event_id: int, image: Optional[str], clip: Optional[str]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE events SET evidence_image = COALESCE(?, evidence_image), "
                "evidence_clip = COALESCE(?, evidence_clip) WHERE id = ?",
                (image, clip, event_id),
            )
            self._conn.commit()

    def list_events(
        self,
        limit: int = 100,
        risk: Optional[str] = None,
        source: Optional[str] = None,
        since: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM events"
        clauses: List[str] = []
        params: List[Any] = []
        if risk:
            clauses.append("risk_level = ?")
            params.append(risk)
        if source:
            clauses.append("source = ?")
            params.append(source)
        if since is not None:
            clauses.append("ts >= ?")
            params.append(float(since))
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY ts DESC LIMIT ?"
        params.append(int(limit))

        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_event(r) for r in rows]

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        for key in ("violation_counts_json", "summary_json"):
            raw = d.pop(key, None)
            target = "violation_counts" if key == "violation_counts_json" else "summary"
            try:
                d[target] = json.loads(raw) if raw else {}
            except (TypeError, ValueError):
                d[target] = {}
        return d

    def prune_events(self, keep: int) -> None:
        with self._lock:
            count = self._conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()["c"]
            if count <= keep:
                return
            excess = count - keep
            self._conn.execute(
                "DELETE FROM events WHERE id IN (SELECT id FROM events ORDER BY ts ASC LIMIT ?)",
                (excess,),
            )
            self._conn.commit()

    # ----------------------------------------------------------
    # alerts
    # ----------------------------------------------------------

    def insert_alert(self, alert: Dict[str, Any]) -> int:
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO alerts (ts, event_id, level, title, message, channel, status, notified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    float(alert.get("ts", time.time())),
                    alert.get("event_id"),
                    alert.get("level", "HIGH"),
                    alert.get("title", "Safety alert"),
                    alert.get("message", ""),
                    alert.get("channel", "dashboard"),
                    alert.get("status", "new"),
                    int(alert.get("notified", 0)),
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def mark_alert_notified(self, alert_id: int, notified: bool = True) -> None:
        with self._lock:
            self._conn.execute("UPDATE alerts SET notified = ? WHERE id = ?", (1 if notified else 0, alert_id))
            self._conn.commit()

    def ack_alert(self, alert_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE alerts SET status = 'acknowledged' WHERE id = ? AND status != 'acknowledged'",
                (alert_id,),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def list_alerts(
        self,
        limit: int = 100,
        status: Optional[str] = None,
        level: Optional[str] = None,
        since: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM alerts"
        clauses: List[str] = []
        params: List[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if level:
            clauses.append("level = ?")
            params.append(level)
        if since is not None:
            clauses.append("ts >= ?")
            params.append(float(since))
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY ts DESC LIMIT ?"
        params.append(int(limit))

        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def count_alerts(self, status: Optional[str] = None) -> int:
        with self._lock:
            if status:
                row = self._conn.execute("SELECT COUNT(*) AS c FROM alerts WHERE status = ?", (status,)).fetchone()
            else:
                row = self._conn.execute("SELECT COUNT(*) AS c FROM alerts").fetchone()
        return int(row["c"])

    # ----------------------------------------------------------
    # counters
    # ----------------------------------------------------------

    def increment_counter(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO counters (name, value) VALUES (?, ?)
                ON CONFLICT(name) DO UPDATE SET value = value + excluded.value
                """,
                (name, int(amount)),
            )
            self._conn.commit()

    def get_counter(self, name: str) -> int:
        with self._lock:
            row = self._conn.execute("SELECT value FROM counters WHERE name = ?", (name,)).fetchone()
        return int(row["value"]) if row else 0

    # ----------------------------------------------------------
    # statistics
    # ----------------------------------------------------------

    def events_since(self, since_ts: float) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE ts >= ? ORDER BY ts ASC", (float(since_ts),)
            ).fetchall()
        return [self._row_to_event(r) for r in rows]


# --------------------------------------------------------------
# module-level singleton
# --------------------------------------------------------------

_DB: Optional[Database] = None
_DB_LOCK = threading.Lock()


def init_db(path: Path) -> Database:
    """Create (or recreate the handle to) the application database."""
    global _DB
    with _DB_LOCK:
        if _DB is not None:
            _DB.close()
        _DB = Database(path)
        logger.info("SQLite database ready at %s", path)
        return _DB


def get_db() -> Database:
    if _DB is None:
        raise RuntimeError("Database not initialised - call init_db() first.")
    return _DB
