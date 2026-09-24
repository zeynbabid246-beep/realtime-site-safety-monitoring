"""
calibrate_threshold.py - calibrate verification thresholds from YOUR data.

The notebook used detection=0.9 / verification=0.7. Those were picked
for the notebook author's single-worker setup; this script measures how
the trained Siamese model behaves on YOUR registered workers and writes
data/face_data/calibration.json. FaceRecognitionConfig picks that file
up automatically (app + CLI), so calibrated thresholds become the
runtime defaults with zero code changes.

It scores:
    genuine  pairs: live crop vs OTHER references of the same worker
    impostor pairs: live crop vs references of every other worker

Prerequisites: at least one worker with 2+ reference images (and
preferably 2+ workers for a meaningful impostor set). Requires the TF
environment used for the model.

Usage:
    python scripts/calibrate_threshold.py
    python scripts/calibrate_threshold.py --data-dir data/face_data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.face_recognition.config import FaceRecognitionConfig
from src.face_recognition.registry import WorkerRegistry
from src.face_recognition import calibration as calib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate Siamese verification thresholds.")
    parser.add_argument(
        "--data-dir", default=None, help="Face data root (default: FACE_DATA_DIR env)."
    )
    parser.add_argument(
        "--max-images-per-worker", type=int, default=12,
        help="Cap on reference images used per worker (keeps the sweep fast).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print results without writing calibration.json."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = FaceRecognitionConfig()
    if args.data_dir:
        config.data_dir = Path(args.data_dir)

    registry = WorkerRegistry(config)
    discovered = registry.auto_discover_workers()
    if discovered:
        print(f"Auto-registered workers found on disk: {discovered}")

    workers = registry.worker_ids(active_only=True)
    if not workers:
        print(
            "ERROR: no registered workers. Register at least one worker with "
            "2+ reference images first:\n"
            "  python scripts/register_worker.py --worker-id worker_001 "
            "--name \"Name\" --images crops/*.jpg",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 66)
    print("SIAMESE THRESHOLD CALIBRATION")
    print("=" * 66)
    print(f"Model     : {config.model_path}")
    print(f"Data dir  : {config.data_dir}")
    print(f"Workers   : {workers}")
    print()

    from src.face_recognition.backends import create_backend

    backend = create_backend(config)
    print("Scoring genuine / impostor pairs (this runs the model a lot)...")
    result = calib.calibrate_threshold(
        backend,
        registry,
        input_size=config.input_size,
        max_images_per_worker=args.max_images_per_worker,
    )

    print()
    print(json.dumps(result, indent=2))
    print()
    print(
        f"Suggested: detection_threshold={result['suggested_detection_threshold']} "
        f"verification_threshold={result['suggested_verification_threshold']} "
        f"TAR={result['true_accept_rate_at_suggestion']} "
        f"FAR={result['false_accept_rate_at_suggestion']}"
    )

    if args.dry_run:
        print("(dry run - calibration.json NOT written)")
        return

    path = calib.save_calibration(config, result)
    print(f"Saved -> {path}")
    print("The app and CLI scripts use these thresholds automatically.")


if __name__ == "__main__":
    main()
