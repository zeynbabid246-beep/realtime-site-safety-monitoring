"""
register_worker.py - add or update a registered worker's identity.

WHY
---
Identities live on disk (workers.json + reference images), never in
source code. This script is the primary non-API way to add/remove
workers; the REST API (POST /face/workers) does the same thing while the
server is running.

WHERE THE DATA GOES
-------------------
    data/face_data/
        workers.json                      # registry metadata (auto-managed)
        input_images/<worker_id>/*.jpg    # reference face crops
        embeddings/<worker_id>.npy        # embedding cache (auto-managed)

USAGE
-----
    # From webcam captures already saved as files (recommended: 5-15
    # frontal 250x250-ish face crops, like the notebook's anchor flow):
    python scripts/register_worker.py --worker-id worker_001 \
        --name "Ahmed Ali" --role "crane operator" \
        --images photos/ahmed1.jpg photos/ahmed2.jpg

    # ...or a whole folder of crops:
    python scripts/register_worker.py --worker-id worker_001 \
        --name "Ahmed Ali" --images "data/my_crops/*.jpg"

    # Live registration straight from the webcam (press C to capture a
    # reference crop of the worker's face, R to finish):
    python scripts/register_worker.py --worker-id worker_002 \
        --name "Sara Hassan" --capture

    # List / remove:
    python scripts/register_worker.py --list
    python scripts/register_worker.py --worker-id worker_002 --remove
    python scripts/register_worker.py --worker-id worker_002 --purge

    # Zero-code alternative: create data/face_data/input_images/<id>/
    # with jpgs inside; the app auto-registers that folder on startup.

NOTE: run with the SAME python environment used for the app
(venv/Scripts/python.exe). TensorFlow is only needed for --recalibrate;
plain registration works without it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.face_recognition.config import FaceRecognitionConfig
from src.face_recognition.registry import WorkerRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register a worker identity for face verification.")
    parser.add_argument("--worker-id", default=None, help="Stable unique id, e.g. worker_001.")
    parser.add_argument("--name", default="", help="Human-readable name shown on overlays/alerts.")
    parser.add_argument("--role", default="", help="Job role / description (optional).")
    parser.add_argument(
        "--images", nargs="*", default=[],
        help="Reference face image files or glob patterns (jpg/png).",
    )
    parser.add_argument(
        "--capture", action="store_true",
        help="Capture reference crops live from the webcam (C=capture, R=finish).",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index for --capture.")
    parser.add_argument("--list", action="store_true", help="List registered workers and exit.")
    parser.add_argument("--remove", action="store_true", help="Deactivate the worker (keep files).")
    parser.add_argument("--purge", action="store_true", help="Delete the worker AND their images.")
    parser.add_argument(
        "--data-dir", default=None, help="Override the face data root (default: FACE_DATA_DIR env)."
    )
    return parser.parse_args()


def _expand_images(patterns):
    files = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            files.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}))
        else:
            import glob as _glob

            files.extend(sorted(Path(p) for p in _glob.glob(pattern)))
    unique = []
    for f in files:
        if f.is_file() and f not in unique:
            unique.append(f)
    return unique


def _load_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"  WARNING: could not read {path} - skipped")
    return image


def capture_from_camera(camera_index: int, worker_id: str, config: FaceRecognitionConfig):
    """Interactive webcam capture of reference face crops."""
    from src.face_recognition.face_detector import FaceDetector

    backends = [cv2.CAP_DSHOW] if sys.platform.startswith("win") else []
    backends.append(cv2.CAP_ANY)
    cap = None
    for backend in backends:
        candidate = cv2.VideoCapture(camera_index, backend)
        if candidate.isOpened():
            cap = candidate
            break
        candidate.release()
    if cap is None or not cap.isOpened():
        print(f"ERROR: could not open camera {camera_index}", file=sys.stderr)
        sys.exit(1)

    # Face crops come from the person detector if available; otherwise
    # fall back to a centred crop region like the notebook's anchor flow.
    face_zone = None
    try:
        from src.hazard.hazard_detector import HazardDetector

        hazard = HazardDetector(Path("models/hazard/best.pt"), confidence=0.4)
        detector = FaceDetector(
            head_region_fraction=config.head_region_fraction,
            face_width_fraction=config.face_width_fraction,
            min_face_size_px=config.min_face_size_px,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"(Hazard model unavailable: {exc} - using fixed centred crop region)")
        hazard, detector = None, None

    print("WEBECAM REGISTRATION - position the worker's face in view.")
    print("  C = capture reference crop   R = finish   Q = quit without saving")
    captured = []
    window = "Register worker (C=capture, R=done, Q=quit)"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            candidates = []
            if hazard is not None:
                try:
                    persons = hazard.extract_persons(hazard.predict(frame))
                    candidates = detector.extract_face_candidates(frame, persons)
                except Exception:  # noqa: BLE001
                    candidates = []

            display = frame.copy()
            if candidates:
                best = max(candidates, key=lambda c: c.crop.shape[0] * c.crop.shape[1])
                x1, y1, x2, y2 = (int(v) for v in best.bbox)
                cv2.rectangle(display, (x1, y1), (x2, y2), (80, 220, 80), 2)
                cv2.putText(display, f"captured: {len(captured)}", (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 80), 2)
            else:
                h, w = frame.shape[:2]
                face_zone = (int(w * 0.3), int(h * 0.12), int(w * 0.7), int(h * 0.55))
                cv2.rectangle(display, face_zone[:2], face_zone[2:], (80, 220, 80), 2)
                cv2.putText(display, "align face in the box", (face_zone[0], face_zone[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 80), 2)

            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                return captured
            if key in (ord("r"), ord("R")):
                break
            if key in (ord("c"), ord("C")):
                if candidates:
                    best = max(candidates, key=lambda c: c.crop.shape[0] * c.crop.shape[1])
                    crop = best.crop
                elif face_zone:
                    crop = frame[face_zone[1]:face_zone[3], face_zone[0]:face_zone[2]]
                else:
                    continue
                if crop is not None and crop.size:
                    captured.append(crop)
                    print(f"  captured crop #{len(captured)} ({crop.shape[1]}x{crop.shape[0]})")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return captured


def main() -> None:
    args = parse_args()

    config = FaceRecognitionConfig()
    if args.data_dir:
        config.data_dir = Path(args.data_dir)
    config.ensure_dirs()
    registry = WorkerRegistry(config)

    if args.list:
        workers = registry.list_workers(active_only=False)
        if not workers:
            print("No workers registered yet.")
            return
        print(f"{'worker_id':<20} {'name':<24} {'role':<18} {'refs':<5} active")
        for worker in workers:
            print(
                f"{worker.worker_id:<20} {worker.name:<24} {worker.role:<18} "
                f"{len(worker.reference_images):<5} {'yes' if worker.active else 'NO'}"
            )
        return

    if not args.worker_id:
        print("ERROR: --worker-id is required (or use --list).", file=sys.stderr)
        sys.exit(1)

    if args.remove or args.purge:
        removed = registry.remove_worker(args.worker_id, delete_images=args.purge)
        print(("Removed" if args.purge else "Deactivated") if removed else "Worker not found:",
              args.worker_id)
        return

    images = [_load_image(p) for p in _expand_images(args.images)]
    images = [img for img in images if img is not None]

    if args.capture:
        images.extend(capture_from_camera(args.camera, args.worker_id, config))

    if images:
        registry.add_worker(args.worker_id, name=args.name, role=args.role)
        added = 0
        for image in images:
            if registry.add_reference_image(args.worker_id, image):
                added += 1
        record = registry.get_worker(args.worker_id)
        print(f"\nRegistered worker '{args.worker_id}' ({record.name or 'unnamed'})")
        print(f"  new reference images : {added}")
        print(f"  total references     : {len(record.reference_images)}")
        print(f"  data dir             : {config.worker_image_dir(args.worker_id)}")
        if added:
            print("\nNext steps:")
            print("  1. (Recommended) python scripts/calibrate_threshold.py")
            print("  2. python scripts/face_verify_webcam.py")
    else:
        record = registry.add_worker(args.worker_id, name=args.name, role=args.role)
        print(f"Worker '{args.worker_id}' saved (no reference images attached yet).")
        print("Add reference face images with --images, --capture, or by dropping")
        print(f"jpgs into: {config.worker_image_dir(args.worker_id)}")


if __name__ == "__main__":
    main()
