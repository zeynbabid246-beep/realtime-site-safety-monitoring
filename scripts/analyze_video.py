"""
analyze_video.py

Standalone real-video validation script for the construction safety
pipeline. Unlike the FastAPI endpoints, this runs directly against a
video file on disk with no server involved - useful for the
"does this actually work on real footage" validation pass.

For every frame it:
    - runs HazardDetector.track() (persons/PPE/cones/machinery/poles/vehicles)
    - runs FireDetector.predict() (fire/smoke)
    - runs them through one persistent SafetyEngine (with a
      DangerZoneTracker, so zone ids are stable across the whole video)
    - draws the overlay and writes it to the output video
    - logs a row to a CSV: frame_index, risk_level, violation_counts, etc.

Usage:
    python scripts/analyze_video.py --video data/videos/your_test_video.mp4

    python scripts/analyze_video.py --video data/videos/your_test_video.mp4 \\
        --output data/output/videos/annotated.mp4 \\
        --csv data/output/videos/summary.csv \\
        --hazard-model models/hazard/best.pt \\
        --fire-model models/fire_smoke/best.pt \\
        --hazard-conf 0.25 --fire-conf 0.25 \\
        --machine-distance 250 --pole-distance 250 \\
        --ppe-overlap 0.40 \\
        --show-progress-every 30

Notes for validation:
    - Watch the CSV's `risk_level` column over time: does it match what
      you see happening in the video? A risk_level of CRITICAL/HIGH
      with zero corresponding visible danger is a sign a threshold
      needs tuning (most likely --ppe-overlap or --machine-distance).
    - `zone_ids_seen` growing every frame (instead of staying flat)
      means danger-zone stabilization ISN'T actually stabilizing on
      this footage - cones may be flickering in/out of detection too
      much for the IoU-matching to work, and you may need to lower
      cone_min_cluster_size or improve cone detection confidence first.
    - `frames_failed` should be 0 or very low. Any failures are logged
      to stderr with the frame number and exception, so you can go
      inspect exactly what that frame looked like.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

# Allow running this script directly (python scripts/analyze_video.py)
# from the project root without needing to `pip install -e .` first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hazard.hazard_detector import HazardDetector
from src.hazard.detection_filter import filter_detections_with_stats
from src.hazard.track_confirmation import TrackConfirmationTracker
from src.fire.fire_detector import FireDetector
from src.fire.fire_confirmation import FireConfirmationTracker
from src.safety.safety_engine import SafetyEngine
from src.safety.geometry import DangerZoneTracker
from src.safety.rules import SafetyConfig
from src.safety.overlay import draw_safety_overlay, draw_unconfirmed_fire_debug
from src.fire.fire_confirmation import detection_kind


RISK_ORDER = ["SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full safety pipeline on a real video.")

    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument("--output", default=None, help="Path for the annotated output video.")
    parser.add_argument("--csv", default=None, help="Path for the per-frame CSV summary.")

    parser.add_argument("--hazard-model", default="models/hazard/best.pt")
    parser.add_argument("--fire-model", default="models/fire_smoke/best.pt")

    parser.add_argument("--hazard-conf", type=float, default=0.25)
    parser.add_argument("--fire-conf", type=float, default=0.20,
                         help="YOLO conf for the Fire-Smoke model (independent of Person).")
    parser.add_argument("--hazard-imgsz", type=int, default=640)
    parser.add_argument("--fire-imgsz", type=int, default=640,
                         help="Inference size for Fire-Smoke. Raise (e.g. 960) if distant "
                              "plumes vanish after the 640 resize.")

    parser.add_argument("--machine-distance", type=float, default=250.0,
                         help="Person<->machinery proximity threshold, in pixels.")
    parser.add_argument("--pole-distance", type=float, default=250.0,
                         help="Machinery/vehicle<->pole proximity threshold, in pixels.")
    parser.add_argument("--ppe-overlap", type=float, default=0.40,
                         help="Minimum fraction of a PPE box that must fall inside a person's "
                              "matched head/torso region (anatomical association, not whole-bbox "
                              "overlap - see rules.py). Default matches SafetyConfig's own default.")
    parser.add_argument("--violation-confidence", type=float, default=0.30,
                         help="Minimum confidence for NO_HARDHAT/NO_SAFETY_VEST detections.")
    parser.add_argument("--fire-violation-confidence", type=float, default=0.25)
    parser.add_argument("--cone-min-cluster-size", type=int, default=3)

    parser.add_argument("--fire-confirm-min-confidence", type=float, default=0.45,
                         help="Minimum confidence for a FIRE box to even be a candidate.")
    parser.add_argument("--fire-confirm-min-area", type=float, default=800.0,
                         help="Minimum bbox area (pixels^2) for a FIRE candidate.")
    parser.add_argument("--fire-confirm-min-frames", type=int, default=5,
                         help="Consecutive frames the SAME fire region must be seen "
                              "before it's treated as real.")
    parser.add_argument("--smoke-confirm-min-confidence", type=float, default=0.20,
                         help="Smoke-only confirmation confidence (do not share with Person/Fire).")
    parser.add_argument("--smoke-confirm-min-area", type=float, default=200.0,
                         help="Smoke-only minimum bbox area (pixels^2). Diffuse plumes are smaller "
                              "in score-map terms than compact fire.")
    parser.add_argument("--smoke-confirm-min-frames", type=int, default=2,
                         help="Smoke-only persistence. Keep this low: smoke boxes morph and "
                              "IoU-match poorly compared to fire.")
    parser.add_argument("--disable-fire-confirmation", action="store_true",
                         help="Skip the confirmation gate entirely - every raw fire/smoke "
                              "detection above --fire-conf reaches the engine immediately. "
                              "Use this to test whether the MODEL sees smoke at all.")
    parser.add_argument("--draw-raw-fire", action="store_true",
                         help="Draw cyan dashed boxes for Fire-Smoke detections the gate dropped. "
                              "If you see dashed Smoke but no solid Smoke, the model fired and "
                              "confirmation filtered it. If you see neither, the model missed.")

    parser.add_argument("--track-confirm-min-hits", type=int, default=3,
                         help="Consecutive-ish frames a track_id must be seen before it counts "
                              "as a real object (person, machinery, etc.), not a one-frame "
                              "flicker false positive.")
    parser.add_argument("--track-confirm-max-missed", type=int, default=5,
                         help="How many frames a track_id can go missing before it's dropped "
                              "and has to re-earn confirmation from scratch.")
    parser.add_argument("--disable-track-confirmation", action="store_true",
                         help="Skip track confirmation - every raw tracked detection (even a "
                              "single-frame flicker) reaches the engine immediately. Not "
                              "recommended; mainly for debugging tracking itself.")

    parser.add_argument("--disable-detection-filter", action="store_true",
                         help="Skip per-class confidence/size/aspect-ratio filtering entirely - "
                              "every raw hazard-model detection reaches the engine. Use this to "
                              "get a BEFORE baseline, then compare against a normal run for an "
                              "objective before/after comparison (see detection_filter.py).")

    parser.add_argument("--skip-fire", action="store_true",
                         help="Skip fire/smoke inference entirely (faster, hazard-only run).")

    parser.add_argument("--show-progress-every", type=int, default=30,
                         help="Print a progress line every N frames.")
    parser.add_argument("--max-frames", type=int, default=None,
                         help="Stop after N frames (useful for a quick first pass).")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    hazard_model_path = Path(args.hazard_model)
    if not hazard_model_path.exists():
        print(f"ERROR: hazard model not found: {hazard_model_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else (
        video_path.parent / f"{video_path.stem}_safety_annotated.mp4"
    )
    csv_path = Path(args.csv) if args.csv else (
        video_path.parent / f"{video_path.stem}_safety_summary.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("REAL-VIDEO SAFETY PIPELINE VALIDATION")
    print("=" * 70)
    print(f"Video           : {video_path}")
    print(f"Output video    : {output_path}")
    print(f"CSV summary     : {csv_path}")
    print(f"Hazard model    : {hazard_model_path}")
    print(f"Fire model      : {'SKIPPED' if args.skip_fire else args.fire_model}")
    print()

    print("Loading models...")
    hazard_detector = HazardDetector(
        hazard_model_path, confidence=args.hazard_conf, image_size=args.hazard_imgsz
    )
    print(f"Hazard classes: {hazard_detector.class_names}")

    fire_detector = None
    if not args.skip_fire:
        fire_model_path = Path(args.fire_model)
        if not fire_model_path.exists():
            print(f"ERROR: fire model not found: {fire_model_path}", file=sys.stderr)
            sys.exit(1)
        fire_detector = FireDetector(
            fire_model_path, confidence=args.fire_conf, image_size=args.fire_imgsz
        )
        print(f"Fire/smoke classes: {fire_detector.class_names}")
        print(f"Fire-Smoke YOLO conf={args.fire_conf} imgsz={args.fire_imgsz}")

    config = SafetyConfig(
        machine_distance_threshold=args.machine_distance,
        pole_distance_threshold=args.pole_distance,
        violation_confidence=args.violation_confidence,
        fire_violation_confidence=args.fire_violation_confidence,
        cone_min_cluster_size=args.cone_min_cluster_size,
        ppe_overlap_threshold=args.ppe_overlap,
    )

    zone_tracker = DangerZoneTracker()

    fire_tracker = None
    if not args.disable_fire_confirmation:
        fire_tracker = FireConfirmationTracker(
            min_confidence=args.fire_confirm_min_confidence,
            min_area_px=args.fire_confirm_min_area,
            min_consecutive_frames=args.fire_confirm_min_frames,
            smoke_min_confidence=args.smoke_confirm_min_confidence,
            smoke_min_area_px=args.smoke_confirm_min_area,
            smoke_min_consecutive_frames=args.smoke_confirm_min_frames,
        )
        print(
            f"Fire confirmation : conf>={args.fire_confirm_min_confidence} "
            f"area>={args.fire_confirm_min_area}px^2 "
            f"frames>={args.fire_confirm_min_frames}"
        )
        print(
            f"Smoke confirmation: conf>={args.smoke_confirm_min_confidence} "
            f"area>={args.smoke_confirm_min_area}px^2 "
            f"frames>={args.smoke_confirm_min_frames}"
        )
    else:
        print("Fire confirmation: DISABLED (--disable-fire-confirmation)")

    track_confirmation = None
    if not args.disable_track_confirmation:
        track_confirmation = TrackConfirmationTracker(
            min_hits=args.track_confirm_min_hits,
            max_missed_frames=args.track_confirm_max_missed,
        )
        print(
            f"Track confirmation: min_hits={args.track_confirm_min_hits} "
            f"max_missed_frames={args.track_confirm_max_missed}"
        )
    else:
        print("Track confirmation: DISABLED (--disable-track-confirmation)")

    if args.disable_detection_filter:
        print("Detection filter (confidence/size/aspect-ratio): DISABLED (--disable-detection-filter)")
    else:
        print("Detection filter (confidence/size/aspect-ratio): ENABLED (see detection_filter.py)")

    engine = SafetyEngine(config=config, zone_tracker=zone_tracker)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: could not open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"\nResolution: {width}x{height}  FPS: {fps:.1f}  Frames: {total_frames}\n")
    print("Processing...\n")

    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame", "risk_level", "violation_count", "violation_types",
        "persons", "machines", "danger_zones", "zone_ids",
        "confirmed_fire_smoke",
        "raw_smoke", "raw_fire",
        "max_raw_smoke_conf", "max_raw_smoke_area",
        "dropped_confidence", "dropped_area", "dropped_persistence",
        "confirmed_smoke", "confirmed_fire",
        "fire_model_ran",
    ])

    frame_index = 0
    frames_failed = 0
    max_risk_seen = "SAFE"
    all_zone_ids_seen = set()
    risk_level_counts = {level: 0 for level in RISK_ORDER}
    raw_fire_frame_count = 0
    confirmed_fire_frame_count = 0
    frames_with_raw_smoke = 0
    frames_with_raw_fire = 0
    frames_with_confirmed_smoke = 0
    frames_with_confirmed_fire = 0
    frames_fire_model_ran = 0
    smoke_gate_totals = {
        "dropped_confidence": 0,
        "dropped_area": 0,
        "dropped_persistence": 0,
        "max_raw_smoke_conf": 0.0,
        "max_raw_smoke_area": 0.0,
    }

    # Accumulated across the whole video - this is your objective
    # "before/after" measurement for the detection filter (point 13 in
    # the false-positive-reduction guide: "Problem | Before | After").
    filter_stats_total = {
        "total": 0, "kept": 0,
        "rejected_confidence": 0, "rejected_size": 0,
        "rejected_aspect_ratio": 0, "rejected_roi": 0,
        "rejected_missing_bbox": 0,
    }

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1

        if args.max_frames and frame_index > args.max_frames:
            print(f"\nReached --max-frames={args.max_frames}, stopping early.")
            break

        try:
            raw_detections = hazard_detector.track(frame)

            # 1. Filter: confidence/size/aspect-ratio - drops most
            #    single-frame junk (ID:714 Person 0.42, tiny hardhats,
            #    etc.) before it ever reaches tracking.
            if args.disable_detection_filter:
                detections = raw_detections
            else:
                detections, frame_filter_stats = filter_detections_with_stats(
                    raw_detections, frame_shape=frame.shape[:2]
                )
                for key, value in frame_filter_stats.items():
                    filter_stats_total[key] = filter_stats_total.get(key, 0) + value

            # 2. Track confirmation: requires a track_id to persist for
            #    several frames and filters out stationary poles/tripods.
            if track_confirmation is not None:
                detections = track_confirmation.update(
                    detections, frame_shape=frame.shape[:2]
                )

            fire_model_ran = fire_detector is not None
            raw_fire_detections = fire_detector.predict(frame) if fire_model_ran else []
            if fire_model_ran:
                frames_fire_model_ran += 1

            gate_stats = {
                "raw_smoke": sum(1 for d in raw_fire_detections if detection_kind(d) == "smoke"),
                "raw_fire": sum(1 for d in raw_fire_detections if detection_kind(d) == "fire"),
                "dropped_confidence": 0,
                "dropped_area": 0,
                "dropped_persistence": 0,
                "confirmed_smoke": 0,
                "confirmed_fire": 0,
                "max_raw_smoke_conf": max(
                    (d.get("confidence", 0.0) for d in raw_fire_detections
                     if detection_kind(d) == "smoke"),
                    default=0.0,
                ),
                "max_raw_smoke_area": 0.0,
            }

            fire_detections = (
                fire_tracker.update(raw_fire_detections)
                if fire_tracker is not None
                else raw_fire_detections
            )
            if fire_tracker is not None:
                gate_stats = fire_tracker.last_stats
            else:
                def _box_area(det):
                    box = det.get("bbox") or [0, 0, 0, 0]
                    return max(0.0, float(box[2]) - float(box[0])) * max(
                        0.0, float(box[3]) - float(box[1])
                    )

                gate_stats["confirmed_smoke"] = sum(
                    1 for d in fire_detections if detection_kind(d) == "smoke"
                )
                gate_stats["confirmed_fire"] = sum(
                    1 for d in fire_detections if detection_kind(d) == "fire"
                )
                gate_stats["max_raw_smoke_area"] = max(
                    (_box_area(d) for d in raw_fire_detections if detection_kind(d) == "smoke"),
                    default=0.0,
                )

            if raw_fire_detections:
                raw_fire_frame_count += 1
            if fire_detections:
                confirmed_fire_frame_count += 1
            if gate_stats.get("raw_smoke", 0) > 0:
                frames_with_raw_smoke += 1
            if gate_stats.get("raw_fire", 0) > 0:
                frames_with_raw_fire += 1
            if gate_stats.get("confirmed_smoke", 0) > 0:
                frames_with_confirmed_smoke += 1
            if gate_stats.get("confirmed_fire", 0) > 0:
                frames_with_confirmed_fire += 1
            smoke_gate_totals["dropped_confidence"] += gate_stats.get("dropped_confidence", 0)
            smoke_gate_totals["dropped_area"] += gate_stats.get("dropped_area", 0)
            smoke_gate_totals["dropped_persistence"] += gate_stats.get("dropped_persistence", 0)
            smoke_gate_totals["max_raw_smoke_conf"] = max(
                smoke_gate_totals["max_raw_smoke_conf"],
                float(gate_stats.get("max_raw_smoke_conf") or 0.0),
            )
            smoke_gate_totals["max_raw_smoke_area"] = max(
                smoke_gate_totals["max_raw_smoke_area"],
                float(gate_stats.get("max_raw_smoke_area") or 0.0),
            )

            persons = hazard_detector.extract_persons(detections)
            machines = hazard_detector.extract_machines(detections)

            result = engine.analyze(
                detections=detections,
                persons=persons,
                machines=machines,
                fire_detections=fire_detections,
                frame_width=float(width),
            )

            annotated = draw_safety_overlay(
                frame, result, detections, hazard_detector.class_names
            )
            if args.draw_raw_fire:
                draw_unconfirmed_fire_debug(annotated, raw_fire_detections, fire_detections)
            writer.write(annotated)

            risk_level = result.get("risk_level", "SAFE")
            risk_level_counts[risk_level] = risk_level_counts.get(risk_level, 0) + 1

            if RISK_ORDER.index(risk_level) > RISK_ORDER.index(max_risk_seen):
                max_risk_seen = risk_level

            zone_ids = [z["zone_id"] for z in result.get("danger_zones", [])]
            all_zone_ids_seen.update(zone_ids)

            violation_types = sorted({v.get("type") for v in result.get("violations", [])})

            csv_writer.writerow([
                frame_index,
                risk_level,
                result.get("violation_count", 0),
                "|".join(violation_types),
                len(persons),
                len(machines),
                len(zone_ids),
                "|".join(str(z) for z in zone_ids),
                len(fire_detections),
                gate_stats.get("raw_smoke", 0),
                gate_stats.get("raw_fire", 0),
                f"{gate_stats.get('max_raw_smoke_conf', 0.0):.3f}",
                f"{gate_stats.get('max_raw_smoke_area', 0.0):.1f}",
                gate_stats.get("dropped_confidence", 0),
                gate_stats.get("dropped_area", 0),
                gate_stats.get("dropped_persistence", 0),
                gate_stats.get("confirmed_smoke", 0),
                gate_stats.get("confirmed_fire", 0),
                int(fire_model_ran),
            ])

        except Exception as exc:  # noqa: BLE001
            frames_failed += 1
            print(f"  [FRAME {frame_index}] FAILED: {exc}", file=sys.stderr)
            writer.write(frame)  # write the raw frame so output stays in sync
            csv_writer.writerow(
                [frame_index, "ERROR", 0, str(exc), 0, 0, 0, "", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            )
            continue

        if frame_index % args.show_progress_every == 0:
            elapsed = time.time() - start_time
            progress = (frame_index / total_frames * 100) if total_frames else 0
            print(
                f"  Frame {frame_index}/{total_frames} ({progress:.1f}%) "
                f"| risk={risk_level:8s} | violations={result.get('violation_count', 0)} "
                f"| zones_seen_so_far={len(all_zone_ids_seen)} "
                f"| {elapsed:.1f}s elapsed"
            )

    cap.release()
    writer.release()
    csv_file.close()

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Frames processed : {frame_index}")
    print(f"Frames failed    : {frames_failed}")
    print(f"Processing time  : {elapsed:.1f}s ({frame_index / elapsed:.1f} fps)" if elapsed > 0 else "")
    print(f"Max risk seen    : {max_risk_seen}")
    if not args.skip_fire:
        print()
        print("SMOKE / FIRE DIAGNOSIS (independent of Person filters)")
        print(f"  Fire-Smoke model ran on every processed frame : "
              f"{'YES' if frames_fire_model_ran == frame_index and frame_index else 'NO'}"
              f"  ({frames_fire_model_ran}/{frame_index})")
        print(f"  Video resolution                              : {width}x{height}")
        print(f"  Fire-Smoke inference size                     : {args.fire_imgsz}")
        print(f"  YOLO --fire-conf                              : {args.fire_conf}")
        print(f"  Frames with RAW Smoke                         : {frames_with_raw_smoke}")
        print(f"  Frames with RAW Fire                          : {frames_with_raw_fire}")
        print(f"  Frames with CONFIRMED Smoke                   : {frames_with_confirmed_smoke}")
        print(f"  Frames with CONFIRMED Fire                    : {frames_with_confirmed_fire}")
        print(f"  Peak RAW smoke confidence / area              : "
              f"{smoke_gate_totals['max_raw_smoke_conf']:.3f} / "
              f"{smoke_gate_totals['max_raw_smoke_area']:.0f}px^2")
        print(f"  Gate drops (conf / area / persistence)        : "
              f"{smoke_gate_totals['dropped_confidence']} / "
              f"{smoke_gate_totals['dropped_area']} / "
              f"{smoke_gate_totals['dropped_persistence']}")
        print()
        if frames_fire_model_ran < frame_index:
            print("  Likely cause: fire model skipped on some frames (--skip-fire or load failure).")
        elif frames_with_raw_smoke == 0 and frames_with_confirmed_smoke == 0:
            print("  Likely cause: Fire-Smoke model never produced a Smoke box.")
            print("  Check appearance vs training data, --fire-conf, --fire-imgsz,")
            print("  and re-run with --disable-fire-confirmation (won't help if RAW is already 0).")
        elif frames_with_raw_smoke > 0 and frames_with_confirmed_smoke == 0:
            print("  Likely cause: model SAW smoke, confirmation gate dropped it.")
            print("  Lower --smoke-confirm-min-confidence / --smoke-confirm-min-area /")
            print("  --smoke-confirm-min-frames (do NOT lower Person confidence).")
            print("  Re-run with --draw-raw-fire to see cyan dashed RAW boxes.")
        else:
            print("  Smoke reached the safety engine on at least some frames.")
        print(f"  Combined RAW fire/smoke frames                : {raw_fire_frame_count}")
        print(f"  Combined CONFIRMED fire/smoke frames          : {confirmed_fire_frame_count}")
    print(f"Distinct zone ids seen across whole video: {sorted(all_zone_ids_seen)}")
    print("  (if this list keeps growing roughly 1-per-frame instead of staying")
    print("   small and stable, zone stabilization isn't matching well on this")
    print("   footage - see the module docstring for what to check.)")
    print()

    if not args.disable_detection_filter:
        print("Detection filter results (across all frames, all classes):")
        print(f"  Total raw detections           : {filter_stats_total['total']}")
        print(f"  Kept                            : {filter_stats_total['kept']}")
        print(f"  Rejected - low confidence       : {filter_stats_total['rejected_confidence']}")
        print(f"  Rejected - too small            : {filter_stats_total['rejected_size']}")
        print(f"  Rejected - bad aspect ratio     : {filter_stats_total['rejected_aspect_ratio']}")
        print(f"  Rejected - outside ROI          : {filter_stats_total['rejected_roi']}")
        print(f"  Rejected - missing bbox         : {filter_stats_total['rejected_missing_bbox']}")
        if filter_stats_total["total"] > 0:
            reject_pct = 100 * (1 - filter_stats_total["kept"] / filter_stats_total["total"])
            print(f"  Overall rejection rate          : {reject_pct:.1f}%")
        print("  (compare this against a run with --disable-detection-filter to see")
        print("   exactly how many detections the filter is removing, and adjust")
        print("   MIN_CONFIDENCE_BY_CLASS / MIN_SIZE_BY_CLASS in detection_filter.py")
        print("   if a category looks over- or under-aggressive.)")
        print()

    print("Risk level distribution:")
    for level in RISK_ORDER:
        count = risk_level_counts.get(level, 0)
        pct = (count / frame_index * 100) if frame_index else 0
        print(f"  {level:10s}: {count:6d} frames ({pct:5.1f}%)")
    print()
    print(f"Annotated video : {output_path.resolve()}")
    print(f"CSV summary     : {csv_path.resolve()}")


if __name__ == "__main__":
    main()