"""
face_verify_webcam.py - standalone multi-worker face verification demo.

This is the production replacement for the notebook's single-person
verify() + webcam loop:

    webcam frame -> hazard YOLO (Person boxes) -> face crops
                 -> Siamese model vs EVERY registered worker
                 -> verified worker name / UNKNOWN, drawn live

The safety pipeline is NOT started here (pure identity demo); use
scripts/live_webcam.py --face-recognition for the combined
safety + identity view, or the web dashboard for everything.

Usage:
    python scripts/face_verify_webcam.py
    python scripts/face_verify_webcam.py --camera 1
    python scripts/face_verify_webcam.py --thresholds 0.85 0.70
    python scripts/face_verify_webcam.py --list-registered
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.face_recognition.config import FaceRecognitionConfig
from src.face_recognition.registry import WorkerRegistry
from src.face_recognition.service import FaceRecognitionService
from src.face_recognition import calibration as face_calibration
from src.face_recognition.overlay import draw_identity_annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live multi-worker face verification.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--hazard-model", default="models/hazard/best.pt")
    parser.add_argument(
        "--thresholds", nargs=2, type=float, metavar=("DET", "VER"),
        default=None, help="Override detection/verification thresholds.",
    )
    parser.add_argument("--data-dir", default=None, help="Face data root override.")
    parser.add_argument("--min-face", type=int, default=None, help="Min face crop height (px).")
    parser.add_argument("--list-registered", action="store_true", help="List workers and exit.")
    return parser.parse_args()


def open_camera(index: int) -> cv2.VideoCapture:
    backends = [cv2.CAP_DSHOW] if sys.platform.startswith("win") else []
    backends.append(cv2.CAP_ANY)
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()
    print(f"ERROR: could not open camera {index}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    args = parse_args()

    config = FaceRecognitionConfig()
    if args.data_dir:
        config.data_dir = Path(args.data_dir)
    if args.min_face:
        config.min_face_size_px = args.min_face
    if args.thresholds:
        config.detection_threshold, config.verification_threshold = args.thresholds
    else:
        # Pick up calibrated thresholds when available.
        config.apply_calibration(face_calibration.load_calibration(config))

    registry = WorkerRegistry(config)
    registry.auto_discover_workers()
    workers = registry.worker_ids(active_only=True)

    if args.list_registered:
        for worker in registry.list_workers(active_only=False):
            state = "active" if worker.active else "INACTIVE"
            print(f"{worker.worker_id:<20} {worker.name:<24} {len(worker.reference_images):>3} refs  {state}")
        return

    if not workers:
        print(
            "ERROR: no workers registered. Register workers first:\n"
            "  python scripts/register_worker.py --worker-id worker_001 "
            "--name \"Ahmed\" --capture",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 62)
    print("LIVE FACE VERIFICATION (Siamese, multi-worker)")
    print("=" * 62)
    print(f"Workers     : {workers}")
    print(f"Thresholds  : detection={config.detection_threshold} "
          f"verification={config.verification_threshold}")
    print(f"Model       : {config.model_path}")
    print()

    from src.face_recognition.backends import create_backend
    from src.hazard.hazard_detector import HazardDetector

    print("Loading models...")
    backend = create_backend(config)
    service = FaceRecognitionService(config, backend=backend, registry=registry)
    hazard = HazardDetector(args.hazard_model, confidence=0.40)
    print("Ready. Press Q to quit.\n")

    cap = open_camera(args.camera)
    window = "Face verification - Q to quit"
    frame_index = 0
    fps = 0.0
    t0 = time.time()
    frames_since = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            frame_index += 1
            try:
                persons = hazard.extract_persons(hazard.predict(frame))
                identities = service.identify_faces(frame, persons)
            except Exception as exc:  # noqa: BLE001
                print(f"  verification failed: {exc}", file=sys.stderr)
                persons, identities = [], []

            annotated = draw_identity_annotations(frame, identities)

            names = [i.label for i in identities] or ["-"]
            frames_since += 1
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                fps = frames_since / elapsed
                frames_since, t0 = 0, time.time()

            hud = f"persons={len(persons)} faces={len(identities)} {fps:.1f} FPS | " + ", ".join(names)
            cv2.putText(annotated, hud, (12, annotated.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, annotated)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
