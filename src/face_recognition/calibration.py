"""
Threshold calibration for the Siamese verifier.

The notebook used fixed thresholds (detection 0.9, verification 0.7).
calibrate_threshold() measures how the trained model actually behaves on
YOUR data and proposes thresholds from score distributions:

    genuine  : (live, same-worker-reference) pairs  -> want HIGH scores
    impostor : (live, other-worker-reference) pairs -> want LOW scores

Proposed detection_threshold = max(genuine floor, impostor ceiling)
(with a safety margin), and verification_threshold = fraction-of-refs
sweep that maximises (true accept rate - false accept rate).

Writes calibration.json next to workers.json; FaceRecognitionConfig.
apply_calibration() consumes it at startup, so calibrated values become
the runtime defaults without touching source code.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.face_recognition.preprocessing import batch_preprocess

logger = logging.getLogger(__name__)


def collect_pairs(
    registry,
    input_size: int,
    backend,
    max_images_per_worker: int = 12,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
    """
    Build genuine and impostor pair sets from the registered workers.

    Returns (genuine, impostor) where each entry is
        {"live": (H,W,3) preprocessed, "scores": [...]} - no, simpler:
    each entry is {"worker": id, "live": preprocessed image, "refs":
    preprocessed refs of the OTHER set}. The caller scores them.
    """

    workers = registry.worker_ids(active_only=True)
    genuine: List[Dict[str, np.ndarray]] = []
    impostor: List[Dict[str, np.ndarray]] = []

    per_worker: Dict[str, np.ndarray] = {}
    for worker_id in workers:
        images = registry.load_reference_images(worker_id)[:max_images_per_worker]
        batch, _ = batch_preprocess(images, (input_size, input_size))
        if batch is None:
            continue
        per_worker[worker_id] = batch

    ids = list(per_worker.keys())
    for worker_id in ids:
        batch = per_worker[worker_id]
        for index, live in enumerate(batch):
            # Genuine: live vs the worker's OTHER references (leave-one-out
            # so a reference is never scored against itself).
            other = np.delete(batch, index, axis=0)
            if other.shape[0] > 0:
                genuine.append({"worker": worker_id, "live": live, "refs": other})
        # Impostor: live vs every other worker's references.
        for other_id in ids:
            if other_id == worker_id:
                continue
            for live in per_worker[worker_id]:
                impostor.append(
                    {"worker": worker_id, "live": live, "refs": per_worker[other_id]}
                )

    return genuine, impostor


def calibrate_threshold(
    backend,
    registry,
    input_size: int = 100,
    detection_margin: float = 0.05,
    max_images_per_worker: int = 12,
) -> Dict:
    """
    Run the calibration sweep. Returns a result dict:

        {
          "genuine_scores": [...], "impostor_scores": [...],
          "suggested_detection_threshold": float,
          "suggested_verification_threshold": float,
          "true_accept_rate_at_suggestion": float,
          "false_accept_rate_at_suggestion": float,
          "n_genuine_pairs": int, "n_impostor_pairs": int,
          "calibrated_at": epoch seconds,
        }
    """

    genuine_sets, impostor_sets = collect_pairs(
        registry, input_size, backend, max_images_per_worker=max_images_per_worker
    )

    if not genuine_sets:
        raise ValueError(
            "Not enough registered workers/images to calibrate. Need at least "
            "one worker with 2+ reference images (and ideally 2+ workers)."
        )

    genuine_scores: List[float] = []
    for item in genuine_sets:
        refs = item["refs"]
        tiled = np.repeat(item["live"][np.newaxis, ...], refs.shape[0], axis=0)
        genuine_scores.extend(backend.verify_batch(tiled, refs).tolist())

    impostor_scores: List[float] = []
    for item in impostor_sets:
        refs = item["refs"]
        tiled = np.repeat(item["live"][np.newaxis, ...], refs.shape[0], axis=0)
        impostor_scores.extend(backend.verify_batch(tiled, refs).tolist())

    genuine_arr = np.asarray(genuine_scores, dtype=np.float64)
    impostor_arr = np.asarray(impostor_scores, dtype=np.float64)

    if impostor_arr.size == 0:
        # Single registered worker: only genuine pairs exist. Suggest the
        # notebook threshold unless the model's own scores argue lower.
        suggested_detection = float(max(0.60, np.percentile(genuine_arr, 5)))
        return _result(
            genuine_arr,
            impostor_arr,
            suggested_detection,
            0.70,
            _rates(genuine_arr, impostor_arr, suggested_detection, 0.70),
        )

    # Detection threshold: impostors must stay BELOW it, genuines ABOVE.
    impostor_high = float(np.percentile(impostor_arr, 99))
    genuine_low = float(np.percentile(genuine_arr, 5))
    if genuine_low > impostor_high:
        # Clean separation - sit in the gap.
        suggested_detection = min(
            0.99, (genuine_low + impostor_high) / 2.0
        )
    else:
        # Overlapping distributions - trust the model's positive class:
        # genuine pairs above ~median are true matches, so set the floor
        # there, and never below the impostor ceiling + margin.
        suggested_detection = float(
            max(np.median(genuine_arr), impostor_high + detection_margin)
        )
    suggested_detection = float(min(max(suggested_detection, 0.30), 0.99))

    # Verification (fraction-of-refs) sweep for the best TAR - FAR.
    best = None
    for det in (suggested_detection, 0.90, 0.85, 0.80, 0.75, 0.70):
        for ver in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
            tar, far = _rates(genuine_arr, impostor_arr, det, ver)
            score = tar - far
            if best is None or score > best[0]:
                best = (score, det, ver, tar, far)

    _, det, ver, tar, far = best
    return _result(genuine_arr, impostor_arr, det, ver, (tar, far))


def _rates(
    genuine: np.ndarray,
    impostor: np.ndarray,
    detection_threshold: float,
    verification_threshold: float,
) -> Tuple[float, float]:
    """(true accept rate, false accept rate) for a threshold pair."""

    def accept(scores: np.ndarray) -> bool:
        if scores.size == 0:
            return False
        rate = float((scores > detection_threshold).mean())
        return rate > verification_threshold

    tar = float(np.mean([accept(np.asarray([s])) for s in genuine])) if genuine.size else 0.0
    far = float(np.mean([accept(np.asarray([s])) for s in impostor])) if impostor.size else 0.0
    return tar, far


def _result(genuine, impostor, det, ver, rates) -> Dict:
    tar, far = rates
    return {
        "suggested_detection_threshold": round(float(det), 4),
        "suggested_verification_threshold": round(float(ver), 4),
        "true_accept_rate_at_suggestion": round(tar, 4),
        "false_accept_rate_at_suggestion": round(far, 4),
        "genuine_min": round(float(genuine.min()), 4) if genuine.size else None,
        "genuine_mean": round(float(genuine.mean()), 4) if genuine.size else None,
        "impostor_max": round(float(impostor.max()), 4) if impostor.size else None,
        "n_genuine_pairs": int(genuine.size),
        "n_impostor_pairs": int(impostor.size),
        "calibrated_at": time.time(),
    }


def save_calibration(config, result: Dict) -> Path:
    """Write calibration.json (consumed by FaceRecognitionConfig.apply_calibration)."""

    config.calibration_path.parent.mkdir(parents=True, exist_ok=True)
    config.calibration_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return config.calibration_path


def load_calibration(config) -> Optional[Dict]:
    path = config.calibration_path
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read calibration file %s: %s", path, exc)
        return None
