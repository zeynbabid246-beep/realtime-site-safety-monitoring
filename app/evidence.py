"""
Evidence capture: annotated JPEG snapshots and short MP4 clips, saved when a
frame reaches an evidence-worthy risk level (default HIGH/CRITICAL).

Files live under `data/evidence/<YYYY-MM-DD>/` and are served read-only by a
StaticFiles mount at `/evidence` (see app/main.py). The DB stores the path
RELATIVE to the evidence dir, so URLs are just `/evidence/<rel>`.

Both captures are best-effort: any failure is logged and swallowed so the
frame pipeline is never interrupted by a disk/encoder problem.

ClipRecorder keeps a rolling pre-buffer of recent annotated frames. On
trigger() it starts an MP4, writes the pre-buffer (context BEFORE the hazard),
then writes `post_frames` subsequent frames as they are pushed, and finalises.
The finished clip path is collected later via take_finished() so the caller can
attach it to the event row once it exists on disk.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional

import cv2
import numpy as np

logger = logging.getLogger("construction_safety.evidence")


def _rel_path(evidence_dir: Path, absolute: Path) -> str:
    try:
        return absolute.relative_to(evidence_dir).as_posix()
    except ValueError:
        return absolute.as_posix()


def save_snapshot(
    annotated: np.ndarray,
    evidence_dir: Path,
    source: str,
    risk_level: str,
) -> Optional[str]:
    """Write one annotated JPEG. Returns the evidence-relative path, or None."""
    if annotated is None:
        return None
    try:
        day = datetime.now().strftime("%Y-%m-%d")
        folder = evidence_dir / day
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%H%M%S") + f"-{int((time.time() % 1) * 1000):03d}"
        name = f"{source}_{stamp}_{risk_level}.jpg"
        absolute = folder / name
        if not cv2.imwrite(str(absolute), annotated):
            logger.warning("cv2.imwrite returned False for %s", absolute)
            return None
        return _rel_path(evidence_dir, absolute)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Snapshot capture failed: %s", exc.__class__.__name__)
        return None


class ClipRecorder:
    """Rolling pre-buffer + armed post-roll MP4 recorder for one stream."""

    def __init__(
        self,
        evidence_dir: Path,
        source: str,
        pre_frames: int = 15,
        post_frames: int = 30,
        fps: float = 10.0,
    ):
        self.evidence_dir = evidence_dir
        self.source = source
        self.pre_frames = max(0, int(pre_frames))
        self.post_frames = max(0, int(post_frames))
        self.fps = float(fps) or 10.0

        self._pre: Deque[np.ndarray] = deque(maxlen=self.pre_frames) if self.pre_frames > 0 else deque(maxlen=1)
        self._writer: Optional[cv2.VideoWriter] = None
        self._remaining = 0
        self._current: Optional[Path] = None
        self._finished: List[str] = []
        self._lock = threading.Lock()

    @property
    def recording(self) -> bool:
        return self._writer is not None

    def push(self, frame: np.ndarray) -> None:
        """Feed every processed (annotated) frame in."""
        if frame is None:
            return
        with self._lock:
            if self._writer is not None:
                try:
                    self._writer.write(frame)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Clip write failed: %s", exc.__class__.__name__)
                    self._finalize_locked()
                    return
                self._remaining -= 1
                if self._remaining <= 0:
                    self._finalize_locked()
            else:
                self._pre.append(frame.copy())

    def trigger(self) -> bool:
        """
        Arm a clip recording using the current pre-buffer for dimensions.
        Returns True if recording started (or was already running).
        """
        with self._lock:
            if self._writer is not None:
                return True  # already recording this incident

            if not self._pre:
                return False

            h, w = self._pre[-1].shape[:2]
            day = datetime.now().strftime("%Y-%m-%d")
            folder = self.evidence_dir / day
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not create evidence folder: %s", exc.__class__.__name__)
                return False

            stamp = time.strftime("%H%M%S")
            path = folder / f"{self.source}_{stamp}_clip.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(str(path), fourcc, self.fps, (w, h))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(path), fourcc, self.fps, (w, h))
            if not writer.isOpened():
                logger.warning("Could not open VideoWriter for %s", path)
                return False

            try:
                for past in self._pre:
                    writer.write(past)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Clip pre-buffer write failed: %s", exc.__class__.__name__)

            self._writer = writer
            self._current = path
            self._remaining = self.post_frames

            if self._remaining <= 0:
                self._finalize_locked()
            return True

    def _finalize_locked(self) -> None:
        if self._writer is None:
            return
        try:
            self._writer.release()
        except Exception:  # noqa: BLE001
            pass
        if self._current is not None:
            self._finished.append(_rel_path(self.evidence_dir, self._current))
        self._writer = None
        self._current = None
        self._remaining = 0

    def take_finished(self) -> List[str]:
        """Pop and return evidence-relative paths of clips finished since last call."""
        with self._lock:
            finished, self._finished = self._finished, []
            return finished

    def close(self) -> None:
        with self._lock:
            self._finalize_locked()


def prune_evidence(evidence_dir: Path, keep: int) -> None:
    """Delete oldest evidence files (jpg + mp4) beyond `keep`."""
    try:
        files = [p for p in evidence_dir.rglob("*") if p.is_file() and p.suffix in (".jpg", ".jpeg", ".mp4")]
    except Exception:  # noqa: BLE001
        return
    if len(files) <= keep:
        return
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in files[keep:]:
        try:
            stale.unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("Could not prune evidence %s: %s", stale, exc)
