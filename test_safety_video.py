from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.safety.safety_engine import SafetyEngine
from src.safety.rules import SafetyConfig


# ============================================================
# CONFIGURATION
# ============================================================

# Hazard detection model
MODEL_PATH = Path("models/hazard/best.pt")

# ============================================================
# IMPORTANT:
# ONLY THIS VIDEO WILL BE PROCESSED
# ============================================================

VIDEO_PATH = Path("data/videos/engine_test.mp4")

# Output directory
OUTPUT_DIR = Path("data/output/videos")

# Detection confidence
CONFIDENCE = 0.25

# YOLO image size
IMAGE_SIZE = 640


# ============================================================
# PROCESS VIDEO
# ============================================================

def process_video(
    video_path: Path,
    model: YOLO,
    safety_engine: SafetyEngine
):

    print()
    print("=" * 70)
    print("PROCESSING VIDEO")
    print("=" * 70)

    print()
    print("Input:")
    print(video_path.resolve())

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path.resolve())
    )

    if not cap.isOpened():

        print()
        print("[ERROR] Could not open video:")
        print(video_path.resolve())

        return False

    # --------------------------------------------------------
    # Video information
    # --------------------------------------------------------

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print()
    print("Video information:")
    print(f"  Resolution : {width} x {height}")
    print(f"  FPS        : {fps:.2f}")
    print(f"  Frames     : {total_frames}")

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Output filename
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / "engine_test_safety_result.mp4"
    )

    print()
    print("Output:")
    print(output_path.resolve())

    # --------------------------------------------------------
    # Video writer
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        print()
        print(
            "[ERROR] Could not create output video."
        )

        cap.release()

        return False

    # ========================================================
    # STATISTICS
    # ========================================================

    frame_number = 0

    max_persons = 0
    max_machines = 0
    max_zones = 0
    max_violations = 0

    total_violation_frames = 0

    # ========================================================
    # FRAME LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # ====================================================
        # YOLO DETECTION + TRACKING
        # ====================================================

        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            verbose=False
        )

        # ----------------------------------------------------
        # If no result
        # ----------------------------------------------------

        if not results:

            writer.write(frame)

            continue

        result = results[0]

        # ====================================================
        # YOLO DETECTIONS
        #
        # Format:
        #
        # [x1, y1, x2, y2, confidence, class_id]
        # ====================================================

        detections = []

        if result.boxes is not None:

            boxes = result.boxes

            for i in range(
                len(boxes)
            ):

                xyxy = (
                    boxes.xyxy[i]
                    .cpu()
                    .numpy()
                )

                x1 = float(
                    xyxy[0]
                )

                y1 = float(
                    xyxy[1]
                )

                x2 = float(
                    xyxy[2]
                )

                y2 = float(
                    xyxy[3]
                )

                confidence = float(
                    boxes.conf[i]
                    .cpu()
                    .item()
                )

                class_id = int(
                    boxes.cls[i]
                    .cpu()
                    .item()
                )

                detections.append(
                    [
                        x1,
                        y1,
                        x2,
                        y2,
                        confidence,
                        class_id
                    ]
                )

        # ====================================================
        # PERSONS
        # ====================================================

        persons = []

        if result.boxes is not None:

            boxes = result.boxes

            for i in range(
                len(boxes)
            ):

                class_id = int(
                    boxes.cls[i]
                    .cpu()
                    .item()
                )

                # Person = class 5
                if class_id != 5:
                    continue

                xyxy = (
                    boxes.xyxy[i]
                    .cpu()
                    .numpy()
                )

                person = {
                    "bbox": [
                        float(xyxy[0]),
                        float(xyxy[1]),
                        float(xyxy[2]),
                        float(xyxy[3])
                    ],
                    "confidence": float(
                        boxes.conf[i]
                        .cpu()
                        .item()
                    ),
                    "class_id": 5,
                    "track_id": None
                }

                # ------------------------------------------------
                # Tracking ID
                # ------------------------------------------------

                if boxes.id is not None:

                    person["track_id"] = int(
                        boxes.id[i]
                        .cpu()
                        .item()
                    )

                persons.append(
                    person
                )

        # ====================================================
        # MACHINES / VEHICLES
        # ====================================================

        machines = []

        if result.boxes is not None:

            boxes = result.boxes

            for i in range(
                len(boxes)
            ):

                class_id = int(
                    boxes.cls[i]
                    .cpu()
                    .item()
                )

                # Machinery = 8
                # Vehicle = 10
                if class_id not in [8, 10]:
                    continue

                xyxy = (
                    boxes.xyxy[i]
                    .cpu()
                    .numpy()
                )

                machine = {
                    "bbox": [
                        float(xyxy[0]),
                        float(xyxy[1]),
                        float(xyxy[2]),
                        float(xyxy[3])
                    ],
                    "confidence": float(
                        boxes.conf[i]
                        .cpu()
                        .item()
                    ),
                    "class_id": class_id
                }

                machines.append(
                    machine
                )

        # ====================================================
        # SAFETY ENGINE
        # ====================================================

        safety_result = safety_engine.analyze(
            detections=detections,
            persons=persons,
            machines=machines
        )

        # ====================================================
        # GET RESULTS
        # ====================================================

        risk_level = safety_result.get(
            "risk_level",
            "SAFE"
        )

        violations = safety_result.get(
            "violations",
            []
        )

        danger_zones = safety_result.get(
            "danger_zones",
            []
        )

        distance_results = safety_result.get(
            "distance_results",
            []
        )

        people_inside_zones = safety_result.get(
            "people_inside_zones",
            []
        )

        # ====================================================
        # STATISTICS
        # ====================================================

        max_persons = max(
            max_persons,
            len(persons)
        )

        max_machines = max(
            max_machines,
            len(machines)
        )

        max_zones = max(
            max_zones,
            len(danger_zones)
        )

        max_violations = max(
            max_violations,
            len(violations)
        )

        if len(violations) > 0:

            total_violation_frames += 1

        # ====================================================
        # CREATE ANNOTATED FRAME
        # ====================================================

        annotated = frame.copy()

        # ====================================================
        # DRAW ALL YOLO DETECTIONS
        # ====================================================

        for detection in detections:

            (
                x1,
                y1,
                x2,
                y2,
                confidence,
                class_id
            ) = detection

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            class_name = model.names.get(
                class_id,
                str(class_id)
            )

            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )

            cv2.putText(
                annotated,
                label,
                (
                    x1,
                    max(
                        20,
                        y1 - 8
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

        # ====================================================
        # DRAW PERSON TRACKING
        # ====================================================

        for person in persons:

            x1, y1, x2, y2 = map(
                int,
                person["bbox"]
            )

            track_id = person.get(
                "track_id"
            )

            # Bottom center
            bottom_x = int(
                (x1 + x2) / 2
            )

            bottom_y = int(
                y2
            )

            # ------------------------------------------------
            # Person ID
            # ------------------------------------------------

            if track_id is not None:

                person_text = (
                    f"Person ID: {track_id}"
                )

            else:

                person_text = "Person"

            cv2.putText(
                annotated,
                person_text,
                (
                    x1,
                    min(
                        height - 10,
                        y2 + 22
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # ------------------------------------------------
            # Bottom center point
            # ------------------------------------------------

            cv2.circle(
                annotated,
                (
                    bottom_x,
                    bottom_y
                ),
                5,
                (255, 255, 255),
                -1
            )

        # ====================================================
        # DRAW DANGER ZONES
        # ====================================================

        for zone_index, polygon in enumerate(
            danger_zones
        ):

            if polygon is None:
                continue

            try:

                coordinates = list(
                    polygon.exterior.coords
                )

                points = [
                    [
                        int(x),
                        int(y)
                    ]
                    for x, y in coordinates
                ]

                if len(points) < 3:
                    continue

                pts = np.array(
                    points,
                    dtype=np.int32
                )

                cv2.polylines(
                    annotated,
                    [pts],
                    True,
                    (255, 255, 255),
                    3
                )

                # Zone label
                label_x = points[0][0]
                label_y = points[0][1]

                cv2.putText(
                    annotated,
                    f"DANGER ZONE {zone_index + 1}",
                    (
                        label_x,
                        max(
                            25,
                            label_y - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

            except Exception:
                continue

        # ====================================================
        # PEOPLE INSIDE DANGER ZONES
        # ====================================================

        for item in people_inside_zones:

            person_index = item.get(
                "person_index"
            )

            if person_index is None:
                continue

            if person_index >= len(persons):
                continue

            person = persons[
                person_index
            ]

            x1, y1, x2, y2 = map(
                int,
                person["bbox"]
            )

            track_id = person.get(
                "track_id"
            )

            # Highlight person
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                4
            )

            if track_id is not None:

                danger_text = (
                    f"DANGER - Person {track_id}"
                )

            else:

                danger_text = (
                    "DANGER - Person"
                )

            cv2.putText(
                annotated,
                danger_text,
                (
                    x1,
                    max(
                        25,
                        y1 - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        # ====================================================
        # PERSON ↔ MACHINE DISTANCES
        # ====================================================

        for distance_data in distance_results:

            person_index = distance_data.get(
                "person_index"
            )

            machine_index = distance_data.get(
                "machine_index"
            )

            distance = distance_data.get(
                "distance"
            )

            # Compatibility with distance_calculator.py
            if distance is None:

                distance = distance_data.get(
                    "distance_pixels"
                )

            if distance is None:
                continue

            if person_index is None:
                continue

            if machine_index is None:
                continue

            if person_index >= len(persons):
                continue

            if machine_index >= len(machines):
                continue

            person_bbox = persons[
                person_index
            ]["bbox"]

            machine_bbox = machines[
                machine_index
            ]["bbox"]

            # ------------------------------------------------
            # Person bottom center
            # ------------------------------------------------

            person_x = int(
                (
                    person_bbox[0]
                    + person_bbox[2]
                ) / 2
            )

            person_y = int(
                person_bbox[3]
            )

            # ------------------------------------------------
            # Machine bottom center
            # ------------------------------------------------

            machine_x = int(
                (
                    machine_bbox[0]
                    + machine_bbox[2]
                ) / 2
            )

            machine_y = int(
                machine_bbox[3]
            )

            # ------------------------------------------------
            # Draw distance line
            # ------------------------------------------------

            cv2.line(
                annotated,
                (
                    person_x,
                    person_y
                ),
                (
                    machine_x,
                    machine_y
                ),
                (255, 255, 255),
                2
            )

            # ------------------------------------------------
            # Distance text
            # ------------------------------------------------

            text_x = int(
                (
                    person_x
                    + machine_x
                ) / 2
            )

            text_y = int(
                (
                    person_y
                    + machine_y
                ) / 2
            )

            cv2.putText(
                annotated,
                f"{distance:.0f} px",
                (
                    text_x,
                    text_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

        # ====================================================
        # SAFETY INFORMATION
        # ====================================================

        cv2.putText(
            annotated,
            f"RISK: {risk_level}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            3
        )

        cv2.putText(
            annotated,
            f"Persons: {len(persons)}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated,
            f"Machines: {len(machines)}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated,
            f"Danger zones: {len(danger_zones)}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated,
            f"Violations: {len(violations)}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated,
            f"Frame: {frame_number}/{total_frames}",
            (20, 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        # ====================================================
        # DRAW VIOLATIONS
        # ====================================================

        violation_y = 230

        for violation in violations[:5]:

            violation_type = violation.get(
                "type",
                "VIOLATION"
            )

            cv2.putText(
                annotated,
                str(violation_type),
                (
                    20,
                    violation_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            violation_y += 25

        # ====================================================
        # WRITE RESULT FRAME
        # ====================================================

        writer.write(
            annotated
        )

        # ====================================================
        # TERMINAL PROGRESS
        # ====================================================

        if frame_number % 50 == 0:

            progress = (
                (frame_number / total_frames) * 100
                if total_frames > 0
                else 0
            )

            print(
                f"Frame {frame_number}/{total_frames} "
                f"({progress:.1f}%) | "
                f"Persons: {len(persons)} | "
                f"Machines: {len(machines)} | "
                f"Zones: {len(danger_zones)} | "
                f"Violations: {len(violations)} | "
                f"Risk: {risk_level}"
            )

    # ========================================================
    # RELEASE
    # ========================================================

    cap.release()
    writer.release()

    # IMPORTANT:
    # No cv2.imshow()
    # No cv2.waitKey()
    # No OpenCV window

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print("=" * 70)
    print("VIDEO PROCESSING COMPLETED")
    print("=" * 70)

    print()
    print("Input:")
    print(
        video_path.resolve()
    )

    print()
    print("Output:")
    print(
        output_path.resolve()
    )

    print()
    print("Statistics:")
    print(
        f"  Frames processed       : {frame_number}"
    )

    print(
        f"  Maximum persons/frame  : {max_persons}"
    )

    print(
        f"  Maximum machines/frame : {max_machines}"
    )

    print(
        f"  Maximum zones/frame    : {max_zones}"
    )

    print(
        f"  Maximum violations/frame: {max_violations}"
    )

    print(
        f"  Frames with violations : {total_violation_frames}"
    )

    print()
    print("RESULT SAVED SUCCESSFULLY.")

    print("=" * 70)

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CONSTRUCTION SAFETY - ENGINE TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        print()
        print("[ERROR] Hazard model not found:")

        print(
            MODEL_PATH.resolve()
        )

        return

    # --------------------------------------------------------
    # CHECK VIDEO
    # --------------------------------------------------------

    if not VIDEO_PATH.exists():

        print()
        print("[ERROR] Video not found:")

        print(
            VIDEO_PATH.resolve()
        )

        return

    # --------------------------------------------------------
    # Make output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("Loading hazard model...")

    print(
        MODEL_PATH.resolve()
    )

    model = YOLO(
        str(
            MODEL_PATH.resolve()
        )
    )

    print()
    print(
        "Hazard model loaded successfully."
    )

    # --------------------------------------------------------
    # Safety configuration
    # --------------------------------------------------------

    config = SafetyConfig(
        machine_distance_threshold=250.0,
        pole_distance_threshold=250.0,
        violation_confidence=0.30,

        # At least 3 cones
        cone_min_cluster_size=3,

        # Current geometry implementation
        cone_min_samples=1
    )

    # --------------------------------------------------------
    # Safety Engine
    # --------------------------------------------------------

    safety_engine = SafetyEngine(
        config=config
    )

    print()
    print(
        "Safety Engine loaded successfully."
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # PROCESS ONLY engine_test.mp4
    # --------------------------------------------------------

    print()
    print("Target video:")
    print(
        VIDEO_PATH.resolve()
    )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    success = process_video(
        video_path=VIDEO_PATH,
        model=model,
        safety_engine=safety_engine
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()

    if success:

        print("=" * 70)
        print("TEST FINISHED SUCCESSFULLY")
        print("=" * 70)

        print()
        print("Input video:")
        print(
            VIDEO_PATH.resolve()
        )

        print()
        print("Detected result:")
        print(
            (
                OUTPUT_DIR
                / "engine_test_safety_result.mp4"
            ).resolve()
        )

        print("=" * 70)

    else:

        print("=" * 70)
        print("TEST FAILED")
        print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()