"""
FaceRecognitionService - the recognition/verification engine.

Given a frame and the person boxes that HazardDetector/YOLO produced,
this service:

    1. crops each person's face region (FaceDetector),
    2. preprocesses crops exactly like the notebook (preprocessing.py),
    3. scores each crop against EVERY registered worker's reference
       images through the Siamese backend,
    4. applies the notebook's two-threshold decision per worker:
           detections  = count(scores > detection_threshold)
           rate        = detections / len(scores)
           verified    = rate > verification_threshold
    5. picks the best verified worker (highest rate, then mean score),
       enforcing the configured runner-up margin,
    6. caches results per track for cache_ttl_s (verification is the
       expensive part of the frame loop).

One FrameIdentity is produced per face; faces that match no worker
return verified=False with worker_id=None and the pipeline/UI renders
them as "Unknown".

The service is thread-safe: the backend serialises model access, and
per-worker reference batches are memoised behind a lock.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.face_recognition.face_detector import FaceCandidate, FaceDetector
from src.face_recognition.preprocessing import batch_preprocess, preprocess_face
from src.face_recognition.registry import WorkerRegistry

logger = logging.getLogger(__name__)


@dataclass
class WorkerMatch:
    """Verification result of one live face against ONE worker."""

    worker_id: str
    worker_name: str
    scores: List[float] = field(default_factory=list)
    detections: int = 0
    rate: float = 0.0
    mean_score: float = 0.0
    verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "worker_name": self.worker_name,
            "rate": round(self.rate, 4),
            "mean_score": round(self.mean_score, 4),
            "detections": self.detections,
            "n_references": len(self.scores),
            "verified": self.verified,
        }


@dataclass
class FrameIdentity:
    """Verification result for one face in one frame."""

    track_id: Optional[int]
    person_index: int
    bbox: List[float]
    verified: bool
    worker_id: Optional[str] = None
    worker_name: Optional[str] = None
    score: float = 0.0            # winning worker's rate (0..1)
    mean_score: float = 0.0       # winning worker's mean pair score
    margin: Optional[float] = None  # rate gap to the runner-up
    matches: List[WorkerMatch] = field(default_factory=list)
    error: Optional[str] = None   # set when verification itself failed

    @property
    def label(self) -> str:
        if self.verified and self.worker_name:
            return self.worker_name
        if self.verified and self.worker_id:
            return self.worker_id
        return "Unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "person_index": self.person_index,
            "bbox": list(self.bbox),
            "verified": self.verified,
            "worker_id": self.worker_id,
            "worker_name": self.worker_name,
            "label": self.label,
            "score": round(float(self.score), 4),
            "mean_score": round(float(self.mean_score), 4),
            "margin": None if self.margin is None else round(float(self.margin), 4),
            "matches": [m.to_dict() for m in self.matches],
            "error": self.error,
        }


class FaceRecognitionService:
    def __init__(
        self,
        config,
        backend=None,
        registry: Optional[WorkerRegistry] = None,
        face_detector: Optional[FaceDetector] = None,
    ):
        self.config = config
        self.registry = registry if registry is not None else WorkerRegistry(config)
        self.face_detector = face_detector or FaceDetector(
            head_region_fraction=config.head_region_fraction,
            face_width_fraction=config.face_width_fraction,
            min_face_size_px=config.min_face_size_px,
            min_person_confidence=config.min_person_confidence,
        )
        self.backend = backend  # injected (built by the app factory)

        # (worker_id, reference-path tuple) -> preprocessed reference batch
        self._reference_cache: Dict[Tuple[str, Tuple[str, ...]], np.ndarray] = {}
        # track/bbox key -> (FrameIdentity, expiry timestamp)
        self._identity_cache: Dict[Any, Tuple[FrameIdentity, float]] = {}
        self._lock = threading.RLock()
        self._warned_no_workers = False

    # ------------------------------------------------------------------
    # reference management (used by registration CLI/REST)
    # ------------------------------------------------------------------

    def add_worker(
        self,
        worker_id: str,
        name: str = "",
        role: str = "",
        meta: Optional[Dict] = None,
        reference_images: Optional[Sequence[np.ndarray]] = None,
    ) -> str:
        """Register (or update) a worker and attach reference face crops."""

        self.registry.add_worker(worker_id, name=name, role=role, meta=meta)
        for image in reference_images or []:
            self.registry.add_reference_image(worker_id, image)
        self._refresh_worker_embeddings(worker_id)
        return worker_id

    def remove_worker(self, worker_id: str, delete_images: bool = False) -> bool:
        removed = self.registry.remove_worker(worker_id, delete_images=delete_images)
        if removed:
            with self._lock:
                self._identity_cache.clear()
                self._drop_reference_cache(worker_id)
        return removed

    def _refresh_worker_embeddings(self, worker_id: str) -> None:
        """(Re)build the on-disk embedding cache for a worker, if the
        backend can embed. Purely an optimisation - never fatal."""

        if self.backend is None or not hasattr(self.backend, "embed_batch"):
            return
        try:
            images = self.registry.load_reference_images(worker_id)
            if not images:
                return
            batch, _ = batch_preprocess(
                images, (self.config.input_size, self.config.input_size)
            )
            if batch is None:
                return
            embeddings = self.backend.embed_batch(batch)
            self.registry.save_embeddings(worker_id, embeddings)
            logger.info(
                "Cached %d embedding(s) for worker %s (dim=%d)",
                embeddings.shape[0], worker_id, embeddings.shape[1],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Embedding cache build failed for %s (falls back to raw images): %s",
                worker_id, exc,
            )

    # ------------------------------------------------------------------
    # per-frame entry point
    # ------------------------------------------------------------------

    def identify_faces(
        self,
        frame: np.ndarray,
        persons: List[Dict[str, Any]],
    ) -> List[FrameIdentity]:
        """
        Verify every detectable face among `persons` against all
        registered workers. Returns one FrameIdentity per processed
        face, in the same order as the input persons.
        """

        candidates = self.face_detector.extract_face_candidates(frame, persons)
        if not candidates:
            return []

        worker_ids = self.registry.worker_ids(active_only=True)
        if not worker_ids:
            if not self._warned_no_workers:
                logger.warning(
                    "Face recognition active but no workers registered - "
                    "all faces will be 'Unknown'. Register workers via "
                    "scripts/register_worker.py or POST /face/workers."
                )
                self._warned_no_workers = True
            return []
        self._warned_no_workers = False

        now = time.time()
        identities: List[FrameIdentity] = []

        for candidate in candidates[: self.config.max_faces_per_frame]:
            cache_key = self._cache_key(candidate)
            cached = self._identity_cache.get(cache_key)
            if cached is not None and cached[1] > now:
                identities.append(cached[0])
                continue

            # No backend (disabled/failed to load): only cached identities
            # can be served; new candidates cannot be verified.
            if self.backend is None:
                continue

            identity = self._verify_candidate(candidate)
            if self.config.cache_ttl_s > 0:
                with self._lock:
                    self._identity_cache[cache_key] = (
                        identity,
                        now + self.config.cache_ttl_s,
                    )
            identities.append(identity)

        return identities

    # ------------------------------------------------------------------
    # single-crop verification
    # ------------------------------------------------------------------

    def verify_crop(
        self,
        crop: np.ndarray,
        track_id: Optional[int] = None,
        person_index: int = -1,
        bbox: Optional[Sequence[float]] = None,
    ) -> FrameIdentity:
        """Verify one BGR face crop against all registered workers."""

        identity = self._verify_candidate(
            FaceCandidate(
                person_index=person_index,
                track_id=track_id,
                bbox=list(bbox) if bbox is not None else [],
                crop=crop,
            )
        )
        return identity

    def _verify_candidate(self, candidate: FaceCandidate) -> FrameIdentity:
        matches: List[WorkerMatch] = []
        error: Optional[str] = None

        try:
            live_batch = preprocess_face(
                candidate.crop, (self.config.input_size, self.config.input_size)
            )
            if live_batch is None:
                error = "invalid face crop"
            else:
                for worker_id in self.registry.worker_ids(active_only=True):
                    matches.append(self._verify_against_worker(live_batch[0], worker_id))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Face verification failed")
            error = f"{exc.__class__.__name__}: {exc}"

        return self._decide(candidate, matches, error)

    def _verify_against_worker(self, live_image: np.ndarray, worker_id: str) -> WorkerMatch:
        record = self.registry.get_worker(worker_id)
        worker_name = record.name if record else worker_id

        scores = self._scores_for_worker(live_image, worker_id)
        scores = [float(s) for s in scores]

        detections = sum(1 for s in scores if s > self.config.detection_threshold)
        rate = (detections / len(scores)) if scores else 0.0
        mean_score = (sum(scores) / len(scores)) if scores else 0.0

        return WorkerMatch(
            worker_id=worker_id,
            worker_name=worker_name or worker_id,
            scores=scores,
            detections=detections,
            rate=rate,
            mean_score=mean_score,
            verified=bool(scores) and rate > self.config.verification_threshold,
        )

    def _scores_for_worker(self, live_image: np.ndarray, worker_id: str) -> List[float]:
        """Pair scores between one live image and all of a worker's refs."""

        reference_batch, ref_paths = self._reference_batch(worker_id)
        if reference_batch is None or reference_batch.shape[0] == 0:
            return []

        # Fast path: cached reference embeddings + closed-form score
        # (sigmoid of the classifier applied to the L1 distance).
        embeddings = self.registry.load_embeddings(
            worker_id, expected_count=reference_batch.shape[0]
        )
        if (
            embeddings is not None
            and self.backend is not None
            and hasattr(self.backend, "score_from_embeddings")
        ):
            try:
                live_embedding = self.backend.embed_batch(live_image[np.newaxis, ...])[0]
                return list(self.backend.score_from_embeddings(live_embedding, embeddings))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Embedding fast path failed (%s); full pass instead.", exc)

        if self.backend is None or not hasattr(self.backend, "verify_batch"):
            return []

        tiled = np.repeat(live_image[np.newaxis, ...], reference_batch.shape[0], axis=0)
        return list(self.backend.verify_batch(tiled, reference_batch))

    def _reference_batch(self, worker_id: str):
        """
        Preprocessed reference batch for a worker, memoised on the exact
        reference-path tuple so registry edits invalidate automatically.
        Returns (batch_or_None, paths).
        """

        paths = tuple(self.registry.reference_paths(worker_id))
        key = (worker_id, paths)
        with self._lock:
            cached = self._reference_cache.get(key)
        if cached is not None:
            return cached, list(paths)

        images = self.registry.load_reference_images(worker_id)
        batch, _ = batch_preprocess(images, (self.config.input_size, self.config.input_size))

        with self._lock:
            # Drop any stale entries for this worker first.
            self._drop_reference_cache(worker_id)
            if batch is not None:
                self._reference_cache[key] = batch
        return batch, list(paths)

    def _drop_reference_cache(self, worker_id: str) -> None:
        self._reference_cache = {
            k: v for k, v in self._reference_cache.items() if k[0] != worker_id
        }

    # ------------------------------------------------------------------
    # decision
    # ------------------------------------------------------------------

    def _decide(
        self,
        candidate: FaceCandidate,
        matches: List[WorkerMatch],
        error: Optional[str],
    ) -> FrameIdentity:
        verified_matches = [m for m in matches if m.verified]
        verified_matches.sort(key=lambda m: (m.rate, m.mean_score), reverse=True)

        identity = FrameIdentity(
            track_id=candidate.track_id,
            person_index=candidate.person_index,
            bbox=list(candidate.bbox),
            verified=False,
            matches=matches,
            error=error,
        )

        if error is not None:
            return identity

        if not verified_matches:
            return identity

        best = verified_matches[0]
        runner_up = verified_matches[1] if len(verified_matches) > 1 else None

        if runner_up is not None and self.config.min_margin > 0:
            denom = max(runner_up.rate, 1e-6)
            margin = (best.rate - runner_up.rate) / denom
            if margin < self.config.min_margin:
                # Ambiguous between two workers - safer to report Unknown.
                identity.margin = margin
                return identity
            identity.margin = margin

        identity.verified = True
        identity.worker_id = best.worker_id
        identity.worker_name = best.worker_name
        identity.score = best.rate
        identity.mean_score = best.mean_score
        return identity

    # ------------------------------------------------------------------
    # cache helpers
    # ------------------------------------------------------------------

    def last_identities(
        self,
        frame: np.ndarray,
        persons: List[Dict[str, Any]],
    ) -> List[FrameIdentity]:
        """
        Cached identities for the current persons, without re-running the
        model. Used on stride-skipped frames: labels stay visible and
        stable between verification passes instead of flickering off.
        Expired entries are dropped, not refreshed (the next due frame
        refreshes them).
        """

        now = time.time()
        identities: List[FrameIdentity] = []
        candidates = self.face_detector.extract_face_candidates(frame, persons)
        for candidate in candidates[: self.config.max_faces_per_frame]:
            cached = self._identity_cache.get(self._cache_key(candidate))
            if cached is not None and cached[1] > now:
                identities.append(cached[0])
        return identities

    @staticmethod
    def _cache_key(candidate: FaceCandidate):
        # Track id when ByteTrack gave us one; otherwise the person's
        # box position (stable enough frame-to-frame for a short TTL).
        if candidate.track_id is not None:
            return ("track", candidate.track_id)
        bbox = candidate.bbox or []
        return ("bbox",) + tuple(int(v) // 16 for v in bbox[:4])

    def clear_caches(self) -> None:
        with self._lock:
            self._identity_cache.clear()
            self._reference_cache.clear()

    def stats(self) -> Dict[str, Any]:
        return {
            "workers_registered": len(self.registry.worker_ids(active_only=True)),
            "cache_entries": len(self._identity_cache),
        }
