"""
Live webcam safety pipeline.

    WEBCAM
      │
      ▼
    VideoCapture(0)
      │
      ▼
    Current Frame
      │
      ├──────────────────────────────┐
      ▼                              ▼
    Hazard YOLO                   Fire-Smoke YOLO
    (Person / Machinery / PPE)    (Fire / Smoke)
      │                              │
      ▼                              ▼
    ByteTrack → Person IDs      confirmation gate
      │                              │
      └──────────┬───────────────────┘
                 ▼
         Distance / Zones
                 ▼
           Safety Engine
                 ▼
    SAFE / LOW / MEDIUM / HIGH / CRITICAL
                 ▼
            LIVE VIDEO  (press Q to quit)

Usage (from project root):

    python scripts/live_webcam.py
    python scripts/live_webcam.py --camera 1
    python scripts/live_webcam.py --skip-fire
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hazard.hazard_detector import HazardDetector
from src.hazard.detection_filter import filter_detections
from src.hazard.track_confirmation import TrackConfirmationTracker
from src.safety.safety_engine import SafetyEngine
from src.safety.geometry import DangerZoneTracker
from src.safety.rules import SafetyConfig
from src.safety.overlay import draw_safety_overlay


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
    parser.add_argument("--disable-filters", action="store_true",
                        help="Skip size/confidence/aspect-ratio and track-confirmation gates.")
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


def main() -> None:
    args = parse_args()

    hazard_model_path = Path(args.hazard_model)
    if not hazard_model_path.exists():
        print(f"ERROR: hazard model not found: {hazard_model_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 62)
    print("LIVE WEBCAM SAFETY PIPELINE")
    print("=" * 62)
    print("  WEBCAM -> VideoCapture -> Hazard YOLO + Fire-Smoke YOLO")
    print("         -> ByteTrack / confirmation -> Distance/Zones")
    print("         -> Safety Engine -> LIVE VIDEO")
    print()

    hazard_detector = HazardDetector(
        hazard_model_path,
        confidence=args.hazard_conf,
        image_size=args.imgsz,
        tracker="bytetrack.yaml",
    )
    print(f"Hazard classes: {hazard_detector.class_names}")

    fire_detector = None
    fire_tracker = None
    if not args.skip_fire:
        from src.fire.fire_detector import FireDetector
        from src.fire.fire_confirmation import FireConfirmationTracker

        fire_model_path = Path(args.fire_model)
        if not fire_model_path.exists():
            print(f"ERROR: fire model not found: {fire_model_path}", file=sys.stderr)
            sys.exit(1)
        fire_detector = FireDetector(fire_model_path, confidence=args.fire_conf, image_size=args.imgsz)
        fire_tracker = FireConfirmationTracker()
        print(f"Fire/smoke classes: {fire_detector.class_names}")
    else:
        print("Fire-Smoke model: OFF (--skip-fire)")

    zone_tracker = DangerZoneTracker()
    track_confirmation = None if args.disable_filters else TrackConfirmationTracker(min_hits=3)
    engine = SafetyEngine(config=SafetyConfig(), zone_tracker=zone_tracker)

    cap = open_camera(args.camera, args.width, args.height)
    print(f"Webcam {args.camera} open. Press Q to quit.\n")

    window_name = "Construction Safety — LIVE"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_index = 0
    t0 = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("WARNING: failed to read frame from webcam.", file=sys.stderr)
                continue

            frame_index += 1

            persons = []
            machines = []
            result = {"risk_level": "SAFE", "danger_zones": []}
            live = frame

            try:
                # Hazard YOLO + ByteTrack (persist IDs across frames)
                detections = hazard_detector.track(frame, persist=True)

                if not args.disable_filters:
                    detections = filter_detections(detections, frame_shape=frame.shape[:2])
                    detections = track_confirmation.update(detections, frame_shape=frame.shape[:2])

                persons = hazard_detector.extract_persons(detections)
                machines = hazard_detector.extract_machines(detections)

                fire_detections = []
                if fire_detector is not None:
                    raw_fire = fire_detector.predict(frame)
                    fire_detections = fire_tracker.update(raw_fire)

                result = engine.analyze(
                    detections=detections,
                    persons=persons,
                    machines=machines,
                    fire_detections=fire_detections,
                    frame_width=float(frame.shape[1]),
                )

                live = draw_safety_overlay(
                    frame, result, detections, hazard_detector.class_names
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [FRAME {frame_index}] pipeline failed: {exc}", file=sys.stderr)

            now = time.time()
            elapsed = now - t0
            if elapsed >= 0.5:
                fps = frame_index / elapsed if elapsed else 0.0
                frame_index = 0
                t0 = now

            hud = (
                f"LIVE  {fps:.1f} FPS  |  persons={len(persons)}  "
                f"machines={len(machines)}  "
                f"zones={len(result.get('danger_zones', []))}  "
                f"|  {result.get('risk_level', 'SAFE')}   [Q quit]"
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
