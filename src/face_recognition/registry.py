"""
WorkerRegistry - storage for registered workers and their reference faces.

LAYOUT (all paths derived from FaceRecognitionConfig.data_dir):

    <data_dir>/
        workers.json                     # registry metadata (this file)
        input_images/<worker_id>/*.jpg   # reference face crops
        embeddings/<worker_id>.npy       # precomputed embeddings (cache)
        calibration.json                 # calibrated thresholds (optional)

workers.json shape:
    {
      "version": 1,
      "workers": {
        "worker_001": {
          "worker_id": "worker_001",
          "name": "Ahmed Ali",
          "role": "crane operator",
          "created_at": 1735...,
          "updated_at": 1735...,
          "active": true,
          "meta": {"shift": "A"},
          "reference_images": ["<worker_id>/<uuid>.jpg", ...]
        }, ...
      }
    }

Design rules:
- Adding/removing a worker never touches source code: drop files in the
  worker folder and (re)register via the CLI / REST API.
- The registry is thread-safe (one lock) because REST registration and
  frame processing can run concurrently.
- Embeddings are an optional cache; every code path that needs them
  falls back to loading the raw reference images when the cache is
  absent or stale (wrong row count).
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class WorkerRecord:
    """One registered worker (identity + reference image bookkeeping)."""

    worker_id: str
    name: str = ""
    role: str = ""
    active: bool = True
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    meta: Dict = field(default_factory=dict)
    reference_images: List[str] = field(default_factory=list)  # relative paths

    def to_dict(self) -> Dict:
        return {
            "worker_id": self.worker_id,
            "name": self.name,
            "role": self.role,
            "active": self.active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "meta": dict(self.meta),
            "reference_images": list(self.reference_images),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "WorkerRecord":
        return cls(
            worker_id=str(data["worker_id"]),
            name=str(data.get("name", "")),
            role=str(data.get("role", "")),
            active=bool(data.get("active", True)),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            meta=dict(data.get("meta", {}) or {}),
            reference_images=[str(p) for p in data.get("reference_images", [])],
        )


class WorkerRegistry:
    """Load/save/query registered workers and their reference face images."""

    def __init__(self, config):
        self.config = config
        self._lock = threading.RLock()
        self._workers: Dict[str, WorkerRecord] = {}
        self._load()

    # ---------------------------------------------------------------
    # persistence
    # ---------------------------------------------------------------

    def _load(self) -> None:
        path = self.config.registry_path
        if not path.exists():
            logger.info("Worker registry not found (fresh): %s", path)
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            workers = raw.get("workers", {}) or {}
            with self._lock:
                self._workers = {
                    str(worker_id): WorkerRecord.from_dict(data)
                    for worker_id, data in workers.items()
                }
            logger.info("Worker registry loaded: %d worker(s) from %s", len(self._workers), path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load worker registry %s: %s", path, exc)

    def save(self) -> None:
        """Persist the registry atomically (write temp, then replace)."""

        path = self.config.registry_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "saved_at": time.time(),
            "workers": {worker_id: rec.to_dict() for worker_id, rec in self._workers.items()},
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        with self._lock:
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(path)

    # ---------------------------------------------------------------
    # CRUD
    # ---------------------------------------------------------------

    def add_worker(
        self,
        worker_id: str,
        name: str = "",
        role: str = "",
        meta: Optional[Dict] = None,
    ) -> WorkerRecord:
        """Create (or update) a worker record. Idempotent on worker_id."""

        worker_id = _safe_worker_id(worker_id)
        with self._lock:
            existing = self._workers.get(worker_id)
            record = existing or WorkerRecord(worker_id=worker_id)
            if name:
                record.name = name
            if role:
                record.role = role
            if meta:
                record.meta.update(meta)
            record.active = True
            record.updated_at = time.time()
            self._workers[worker_id] = record
        self.save()
        self.config.worker_image_dir(worker_id).mkdir(parents=True, exist_ok=True)
        return record

    def get_worker(self, worker_id: str) -> Optional[WorkerRecord]:
        with self._lock:
            record = self._workers.get(worker_id)
            return WorkerRecord.from_dict(record.to_dict()) if record else None

    def list_workers(self, active_only: bool = True) -> List[WorkerRecord]:
        with self._lock:
            records = list(self._workers.values())
        if active_only:
            records = [r for r in records if r.active]
        return sorted(records, key=lambda r: r.worker_id)

    def worker_ids(self, active_only: bool = True) -> List[str]:
        return [r.worker_id for r in self.list_workers(active_only=active_only)]

    def remove_worker(self, worker_id: str, delete_images: bool = False) -> bool:
        """
        Deactivate a worker (kept in the registry with active=False).

        With delete_images=True the reference folder is deleted too and
        the record is removed entirely.
        """

        worker_id = _safe_worker_id(worker_id)
        with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                return False
            if delete_images:
                del self._workers[worker_id]
            else:
                record.active = False
                record.updated_at = time.time()
        self.save()

        if delete_images:
            image_dir = self.config.worker_image_dir(worker_id)
            try:
                if image_dir.exists():
                    for image_file in image_dir.iterdir():
                        image_file.unlink(missing_ok=True)
                    image_dir.rmdir()
                emb = self.config.embeddings_dir / f"{worker_id}.npy"
                emb.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Could not fully remove worker %s data: %s", worker_id, exc)
        return True

    # ---------------------------------------------------------------
    # reference images
    # ---------------------------------------------------------------

    def add_reference_image(
        self,
        worker_id: str,
        image: np.ndarray,
        filename: Optional[str] = None,
    ) -> Optional[str]:
        """
        Save a BGR image as a new reference face for a worker and update
        the registry. Any prior embedding cache for that worker is
        invalidated. Returns the relative path stored in the registry.
        """

        if image is None or getattr(image, "size", 0) == 0:
            return None

        worker_id = _safe_worker_id(worker_id)
        # Register implicitly so file + record always agree.
        record = self.get_worker(worker_id) or self.add_worker(worker_id)

        image_dir = self.config.worker_image_dir(worker_id)
        image_dir.mkdir(parents=True, exist_ok=True)

        filename = filename or f"{uuid.uuid4().hex}.jpg"
        if not Path(filename).suffix.lower() in ALLOWED_IMAGE_SUFFIXES:
            filename = f"{Path(filename).stem}.jpg"

        abs_path = image_dir / filename
        if not cv2.imwrite(str(abs_path), image):
            logger.error("Failed to write reference image: %s", abs_path)
            return None

        rel_path = f"{worker_id}/{filename}"
        with self._lock:
            record.reference_images.append(rel_path)
            record.updated_at = time.time()
            self._workers[worker_id] = record
        self.save()
        self.invalidate_embeddings(worker_id)
        logger.info("Reference image added for %s: %s", worker_id, abs_path)
        return rel_path

    def remove_reference_image(self, worker_id: str, rel_path: str) -> bool:
        worker_id = _safe_worker_id(worker_id)
        with self._lock:
            record = self._workers.get(worker_id)
            if record is None or rel_path not in record.reference_images:
                return False
            record.reference_images.remove(rel_path)
            record.updated_at = time.time()
        self.save()
        try:
            (self.config.input_images_dir / rel_path).unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not delete reference image %s: %s", rel_path, exc)
        self.invalidate_embeddings(worker_id)
        return True

    def reference_paths(self, worker_id: str, active_only: bool = True) -> List[str]:
        record = self.get_worker(worker_id)
        if record is None or (active_only and not record.active):
            return []
        return list(record.reference_images)

    def load_reference_images(
        self,
        worker_id: str,
        only_paths: Optional[List[str]] = None,
    ) -> List[np.ndarray]:
        """
        Read a worker's reference images from disk as BGR arrays (in
        registry order). Missing/corrupt files are skipped with a warning
        so one bad file never breaks verification.
        """

        paths = only_paths if only_paths is not None else self.reference_paths(worker_id)
        images: List[np.ndarray] = []
        for rel_path in paths:
            abs_path = self.config.input_images_dir / rel_path
            image = cv2.imread(str(abs_path), cv2.IMREAD_COLOR)
            if image is None:
                logger.warning("Skipping unreadable reference image: %s", abs_path)
                continue
            images.append(image)
        return images

    def auto_discover_workers(self) -> List[str]:
        """
        Register any <input_images>/<folder> that contains images but has
        no registry record yet (folder name = worker_id). Lets users add
        workers by simply creating a folder of face photos - no code
        changes.
        """

        discovered: List[str] = []
        root = self.config.input_images_dir
        if not root.exists():
            return discovered
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if self.get_worker(child.name) is not None:
                continue
            if not any(
                p.suffix.lower() in ALLOWED_IMAGE_SUFFIXES for p in child.iterdir() if p.is_file()
            ):
                continue
            self.add_worker(child.name, name=child.name)
            for image_file in sorted(child.iterdir()):
                if image_file.suffix.lower() in ALLOWED_IMAGE_SUFFIXES:
                    with self._lock:
                        record = self._workers[child.name]
                        if str(image_file.name) not in record.reference_images:
                            record.reference_images.append(f"{child.name}/{image_file.name}")
                            record.updated_at = time.time()
                    self.save()
            discovered.append(child.name)
        if discovered:
            logger.info("Auto-registered workers from input_images/: %s", discovered)
        return discovered

    # ---------------------------------------------------------------
    # embedding cache
    # ---------------------------------------------------------------

    def _embedding_cache_path(self, worker_id: str) -> Path:
        return self.config.embeddings_dir / f"{_safe_worker_id(worker_id)}.npy"

    def load_embeddings(self, worker_id: str, expected_count: Optional[int] = None):
        """
        Load cached embeddings for a worker, or None when missing/stale
        (row count no longer matches the reference image count).
        """

        path = self._embedding_cache_path(worker_id)
        if not path.exists():
            return None
        try:
            data = np.load(str(path))
            if expected_count is not None and data.shape[0] != expected_count:
                return None
            return data
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load embeddings for %s: %s", worker_id, exc)
            return None

    def save_embeddings(self, worker_id: str, embeddings: np.ndarray) -> None:
        self.config.embeddings_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(self._embedding_cache_path(worker_id)), np.asarray(embeddings))

    def invalidate_embeddings(self, worker_id: str) -> None:
        try:
            self._embedding_cache_path(worker_id).unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("Could not invalidate embeddings for %s: %s", worker_id, exc)


def _safe_worker_id(worker_id: str) -> str:
    """
    Normalise a worker id to a safe single path segment (no separators,
    no traversal, non-empty). Worker ids are user input; never trust them.
    """

    cleaned = "".join(ch for ch in str(worker_id) if ch.isalnum() or ch in "-_ ").strip()
    cleaned = cleaned.replace(" ", "_")
    if not cleaned:
        cleaned = f"worker_{uuid.uuid4().hex[:8]}"
    return cleaned[:64]
