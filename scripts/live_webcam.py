"""
Live webcam safety pipeline.

    WEBCAM -> VideoCapture -> SafetyPipeline (SAME code path as the API
    and analyze_video.py) -> LIVE VIDEO  (press Q to quit)

The pipeline (src/pipeline.py::SafetyPipeline) runs hazard YOLO +
ByteTrack, the stateless detection filter, track confirmation, fire/smoke
YOLO + confirmation, the SafetyEngine (zones / perspective distance / PPE
/ proximity / fire), and the overlay - identical to the server, so the
webcam and uploaded videos behave the same.

Usage (from project root):

    python scripts/live_webcam.py
    python scripts/live_webcam.py --camera 1
    python scripts/live_webcam.py --skip-fire
    python scripts/live_webcam.py --disable-perspective-distance
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hazard.hazard_detector import HazardDetector
from src.safety.rules import SafetyConfig
from src.pipeline import SafetyPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live webcam construction-safety pipeline.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default 0).")
    parser.add_argument("--hazard-model", default="models/hazard/best.pt")
    parser.add_argument("--hazard-conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--skip-fire", action="store_true",
                        help="Run hazard-only (no Fire-Smoke model).")
    parser.add_argument("--fire-model", default="models/fire_smoke/best.pt")
    parser.add_argument("--fire-conf", type=float, default=0.20)
    parser.add_argument("--machine-distance-m", type=float, default=2.5,
                        help="Depth-aware person<->machine proximity threshold in metres.")
    parser.add_argument("--disable-perspective-distance", action="store_true",
                        help="Use the pure-pixel proximity threshold instead of the metre estimate.")
    parser.add_argument("--disable-static-machinery-filter", action="store_true",
                        help="Keep perfectly-static machinery/vehicle tracks (e.g. a background "
                             "building misclassified as machinery) instead of suppressing them.")
    parser.add_argument("--disable-filters", action="store_true",
                        help="Skip size/confidence/aspect-ratio and track-confirmation gates.")
    parser.add_argument("--face-recognition", action="store_true",
                        help="Enable worker identification (Siamese face verification).")
    parser.add_argument("--face-frame-stride", type=int, default=None,
                        help="Verify identities every Nth frame (default: FACE_FRAME_STRIDE env or 1).")
    return parser.parse_args()


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    backends = []
    if sys.platform.startswith("win"):
        backends.append(cv2.CAP_DSHOW)
    backends.append(cv2.CAP_ANY)

    cap = None
    for backend in backends:
        candidate = cv2.VideoCapture(index, backend)
        if candidate.isOpened():
            cap = candidate
            break
        candidate.release()

    if cap is None or not cap.isOpened():
        print(f"ERROR: could not open webcam index {index}.", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _maybe_face_service(args):
    """Build the shared FaceRecognitionService for --face-recognition."""

    if not args.face_recognition:
        return None

    try:
        from src.face_recognition.config import FaceRecognitionConfig
        from src.face_recognition.registry import WorkerRegistry
        from src.face_recognition.service import FaceRecognitionService
        from src.face_recognition import calibration as face_calibration
        from src.face_recognition.backends import create_backend

        face_config = FaceRecognitionConfig()
        face_config.apply_calibration(face_calibration.load_calibration(face_config))
        face_config.ensure_dirs()
        registry = WorkerRegistry(face_config)
        registry.auto_discover_workers()
        backend = create_backend(face_config)
        service = FaceRecognitionService(face_config, backend=backend, registry=registry)
        workers = registry.worker_ids(active_only=True)
        print(f"Face recognition: ON  workers={workers} "
              f"thresholds=({face_config.detection_threshold:.2f}, "
              f"{face_config.verification_threshold:.2f})")
        if not workers:
            print("  (no workers registered - all faces will show as UNKNOWN; "
                  "register via scripts/register_worker.py)")
        return service
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: face recognition unavailable ({exc}) - continuing without.",
              file=sys.stderr)
        return None


def main() -> None:
    args = parse_args()

    hazard_model_path = Path(args.hazard_model)
    if not hazard_model_path.exists():
        print(f"ERROR: hazard model not found: {hazard_model_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 62)
    print("LIVE WEBCAM SAFETY PIPELINE")
    print("=" * 62)

    hazard_detector = HazardDetector(
        hazard_model_path, confidence=args.hazard_conf, image_size=args.imgsz,
        tracker="bytetrack.yaml",
    )
    print(f"Hazard classes: {hazard_detector.class_names}")

    fire_detector = None
    if not args.skip_fire:
        from src.fire.fire_detector import FireDetector

        fire_model_path = Path(args.fire_model)
        if not fire_model_path.exists():
            print(f"ERROR: fire model not found: {fire_model_path}", file=sys.stderr)
            sys.exit(1)
        fire_detector = FireDetector(fire_model_path, confidence=args.fire_conf, image_size=args.imgsz)
        print(f"Fire/smoke classes: {fire_detector.class_names}")
    else:
        print("Fire-Smoke model: OFF (--skip-fire)")

    config = SafetyConfig(
        machine_distance_threshold_m=args.machine_distance_m,
        use_perspective_distance=not args.disable_perspective_distance,
    )

    from src.hazard.track_confirmation import TrackConfirmationTracker

    track_confirmation = None
    if not args.disable_filters:
        track_confirmation = TrackConfirmationTracker(
            min_hits=3,
            filter_static_machinery=not args.disable_static_machinery_filter,
        )

    pipeline = SafetyPipeline(
        hazard_detector,
        fire_detector,
        config=config,
        enable_detection_filter=not args.disable_filters,
        enable_track_confirmation=track_confirmation is not None,
        enable_fire_confirmation=fire_detector is not None,
        track_confirmation=track_confirmation,
        face_service=_maybe_face_service(args),
        face_frame_stride=args.face_frame_stride or 1,
    )

    cap = open_camera(args.camera, args.width, args.height)
    print(f"Webcam {args.camera} open. Press Q to quit.\n")

    window_name = "Construction Safety — LIVE"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    t0 = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("WARNING: failed to read frame from webcam.", file=sys.stderr)
                continue

            persons_n = machines_n = zones_n = 0
            risk_level = "SAFE"
            live = frame

            try:
                fr = pipeline.process_frame(frame, track=True, draw=True)
                live = fr.annotated
                persons_n = len(fr.persons)
                machines_n = len(fr.machines)
                zones_n = len(fr.result.get("danger_zones", []))
                risk_level = fr.result.get("risk_level", "SAFE")
            except Exception as exc:  # noqa: BLE001
                print(f"  pipeline failed: {exc}", file=sys.stderr)

            frame_count += 1
            now = time.time()
            elapsed = now - t0
            if elapsed >= 0.5:
                fps = frame_count / elapsed if elapsed else 0.0
                frame_count = 0
                t0 = now

            hud = (
                f"LIVE  {fps:.1f} FPS  |  persons={persons_n}  machines={machines_n}  "
                f"zones={zones_n}  |  {risk_level}   [Q quit]"
            )
            cv2.putText(
                live, hud, (12, live.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
            )

            cv2.imshow(window_name, live)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam closed.")


if __name__ == "__main__":
    main()
