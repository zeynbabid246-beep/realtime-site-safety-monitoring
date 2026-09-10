"""
Report generation from the SQLite event/alert history.

A report is a period aggregate (24h / 7d / 30d / all): totals, breakdowns by
risk level, violation type, source, and day, plus evidence coverage and the
most severe events. `events_to_csv` flattens the raw events in the window for a
downloadable spreadsheet.
"""

from __future__ import annotations

import csv
import io
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.settings import RISK_ORDER
from app.storage import Database

RANGES: Dict[str, Optional[float]] = {
    "24h": 24 * 3600.0,
    "7d": 7 * 24 * 3600.0,
    "30d": 30 * 24 * 3600.0,
    "all": None,
}


def _since_ts(range_key: str) -> Optional[float]:
    seconds = RANGES.get(range_key, RANGES["24h"])
    if seconds is None:
        return None
    return time.time() - seconds


def _day_key(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().strftime("%Y-%m-%d")


def build_report(db: Database, range_key: str = "24h") -> Dict[str, Any]:
    since = _since_ts(range_key)
    events = db.events_since(since) if since is not None else db.list_events(limit=100000)
    # list_events returns DESC; normalise to ASC for timeline building.
    events = sorted(events, key=lambda e: e.get("ts", 0.0))

    alerts = db.list_alerts(limit=100000, since=since)

    by_risk: Dict[str, int] = {level: 0 for level in RISK_ORDER}
    by_type: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    by_day: Dict[str, int] = {}
    events_with_image = 0
    events_with_clip = 0

    for event in events:
        risk = event.get("risk_level", "SAFE")
        by_risk[risk] = by_risk.get(risk, 0) + 1

        source = event.get("source", "unknown")
        by_source[source] = by_source.get(source, 0) + 1

        by_day[_day_key(event.get("ts", 0.0))] = by_day.get(_day_key(event.get("ts", 0.0)), 0) + 1

        if event.get("evidence_image"):
            events_with_image += 1
        if event.get("evidence_clip"):
            events_with_clip += 1

        for vtype, count in (event.get("violation_counts", {}) or {}).items():
            by_type[vtype] = by_type.get(vtype, 0) + int(count)

    acknowledged = sum(1 for a in alerts if a.get("status") == "acknowledged")

    severity_rank = {level: i for i, level in enumerate(RISK_ORDER)}
    top_events = sorted(
        events,
        key=lambda e: (severity_rank.get(e.get("risk_level", "SAFE"), 0), e.get("violation_count", 0)),
        reverse=True,
    )[:10]

    return {
        "range": range_key,
        "generated_at": time.time(),
        "since": since,
        "totals": {
            "events": len(events),
            "alerts": len(alerts),
            "alerts_acknowledged": acknowledged,
            "alerts_unacknowledged": len(alerts) - acknowledged,
            "frames_processed": db.get_counter("frames_processed"),
            "events_with_image": events_with_image,
            "events_with_clip": events_with_clip,
        },
        "by_risk": by_risk,
        "by_violation_type": dict(sorted(by_type.items(), key=lambda kv: kv[1], reverse=True)),
        "by_source": by_source,
        "by_day": dict(sorted(by_day.items())),
        "top_events": [
            {
                "id": e.get("id"),
                "ts": e.get("ts"),
                "source": e.get("source"),
                "risk_level": e.get("risk_level"),
                "violation_count": e.get("violation_count"),
                "violation_counts": e.get("violation_counts", {}),
                "evidence_image": e.get("evidence_image"),
                "evidence_clip": e.get("evidence_clip"),
            }
            for e in top_events
        ],
    }


_CSV_COLUMNS = [
    "id", "ts", "datetime", "source", "risk_level", "violation_count",
    "persons", "machines", "zones", "fire", "smoke", "ppe",
    "violation_counts", "evidence_image", "evidence_clip",
]


def events_to_csv(db: Database, range_key: str = "24h") -> str:
    since = _since_ts(range_key)
    events = db.events_since(since) if since is not None else db.list_events(limit=100000)
    events = sorted(events, key=lambda e: e.get("ts", 0.0))

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(_CSV_COLUMNS)
    for e in events:
        ts = e.get("ts", 0.0)
        writer.writerow(
            [
                e.get("id"),
                f"{ts:.3f}",
                datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().isoformat(),
                e.get("source"),
                e.get("risk_level"),
                e.get("violation_count"),
                e.get("persons"),
                e.get("machines"),
                e.get("zones"),
                e.get("fire"),
                e.get("smoke"),
                e.get("ppe"),
                "; ".join(f"{k}:{v}" for k, v in (e.get("violation_counts", {}) or {}).items()),
                e.get("evidence_image") or "",
                e.get("evidence_clip") or "",
            ]
        )
    return buffer.getvalue()
