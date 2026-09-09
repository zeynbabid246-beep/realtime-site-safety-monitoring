"""
Test dangerous-zone detection on a real construction video.

Pipeline:
    YOLO hazard model
        ↓
    Person + Safety Cone + Machinery detection
        ↓
    Safety cone clustering
        ↓
    Dangerous-zone polygon
        ↓
    Person bottom-center
        ↓
    Inside / Outside dangerous zone
        ↓
    Annotated output video
"""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.safety.geometry import (
    PERSON_CLASS_ID,
    SAFETY_CONE_CLASS_ID,
    MACHINERY_CLASS_ID,
    build_danger_zones,
    find_people_inside_zones,
    get_bottom_center,
)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path("models/hazard/best.pt")

VIDEO_PATH = Path("data/videos/hazard_test1.mp4")

OUTPUT_DIR = Path("data/output/videos/danger_zone_test")

OUTPUT_PATH = OUTPUT_DIR / "danger_zone_result.mp4"

CONFIDENCE = 0.25

IMAGE_SIZE = 640

# Process every frame.
# Change to 2 or 3 later if the video is too slow.
FRAME_SKIP = 1


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "Hardhat",
    1: "Mask",
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
    5: "Person",
    6: "Safety Cone",
    7: "Safety Vest",
    8: "machinery",
    9: "utility pole",
    10: "vehicle",
}


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print("DANGEROUS ZONE VIDEO TEST")
    print("========================================")

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Check video
    # --------------------------------------------------------

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load YOLO model
    # --------------------------------------------------------

    print(f"\nLoading model:")
    print(MODEL_PATH)

    model = YOLO(str(MODEL_PATH))

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print("\nVideo information:")
    print(f"Resolution : {width} x {height}")
    print(f"FPS        : {fps:.2f}")
    print(f"Frames     : {total_frames}")

    # --------------------------------------------------------
    # Video writer
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            f"Could not create output video: {OUTPUT_PATH}"
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    frame_number = 0

    processed_frames = 0

    frames_with_cones = 0

    frames_with_zones = 0

    frames_with_people_in_zone = 0

    total_people_in_zone = 0

    max_people_in_zone = 0

    # ========================================================
    # PROCESS VIDEO
    # ========================================================

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # ----------------------------------------------------
        # Frame skipping
        # ----------------------------------------------------

        if FRAME_SKIP > 1:

            if (frame_number - 1) % FRAME_SKIP != 0:

                writer.write(frame)

                continue

        processed_frames += 1

        # ----------------------------------------------------
        # YOLO inference
        # ----------------------------------------------------

        results = model.predict(
            source=frame,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            verbose=False,
        )

        result = results[0]

        detections = []

        # ----------------------------------------------------
        # Convert YOLO detections into our standard format
        #
        # [x1, y1, x2, y2, confidence, class_id]
        # ----------------------------------------------------

        if result.boxes is not None:

            boxes = result.boxes

            for i in range(len(boxes)):

                xyxy = boxes.xyxy[i].cpu().numpy()

                confidence = float(
                    boxes.conf[i].cpu().item()
                )

                class_id = int(
                    boxes.cls[i].cpu().item()
                )

                x1, y1, x2, y2 = map(
                    float,
                    xyxy
                )

                detections.append(
                    [
                        x1,
                        y1,
                        x2,
                        y2,
                        confidence,
                        class_id,
                    ]
                )

        # ----------------------------------------------------
        # Count important classes
        # ----------------------------------------------------

        persons = [
            d for d in detections
            if int(d[5]) == PERSON_CLASS_ID
        ]

        cones = [
            d for d in detections
            if int(d[5]) == SAFETY_CONE_CLASS_ID
        ]

        machinery = [
            d for d in detections
            if int(d[5]) == MACHINERY_CLASS_ID
        ]

        # ----------------------------------------------------
        # Build dangerous zones
        # ----------------------------------------------------

        danger_zones = build_danger_zones(
            detections,
            min_cluster_size=3,
            min_samples=1,
        )

        # ----------------------------------------------------
        # Find people inside zones
        # ----------------------------------------------------

        people_inside = find_people_inside_zones(
            detections,
            danger_zones,
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        if len(cones) >= 3:
            frames_with_cones += 1

        if danger_zones:
            frames_with_zones += 1

        if people_inside:
            frames_with_people_in_zone += 1

        people_in_zone_count = len(
            people_inside
        )

        total_people_in_zone += (
            people_in_zone_count
        )

        max_people_in_zone = max(
            max_people_in_zone,
            people_in_zone_count,
        )

        # ====================================================
        # DRAW DETECTIONS
        # ====================================================

        for detection in detections:

            x1, y1, x2, y2, confidence, class_id = detection

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            class_id = int(class_id)

            class_name = CLASS_NAMES.get(
                class_id,
                f"class_{class_id}",
            )

            # ------------------------------------------------
            # Person
            # ------------------------------------------------

            if class_id == PERSON_CLASS_ID:

                # Normal person box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 0),
                    2,
                )

                label = (
                    f"Person {confidence:.2f}"
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

                # Bottom-center point
                point = get_bottom_center(
                    [x1, y1, x2, y2]
                )

                px = int(point[0])
                py = int(point[1])

                cv2.circle(
                    frame,
                    (px, py),
                    5,
                    (255, 255, 0),
                    -1,
                )

            # ------------------------------------------------
            # Safety Cone
            # ------------------------------------------------

            elif class_id == SAFETY_CONE_CLASS_ID:

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    2,
                )

                label = (
                    f"Cone {confidence:.2f}"
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )

                point = get_bottom_center(
                    [x1, y1, x2, y2]
                )

                cv2.circle(
                    frame,
                    (
                        int(point[0]),
                        int(point[1]),
                    ),
                    5,
                    (0, 255, 255),
                    -1,
                )

            # ------------------------------------------------
            # Machinery
            # ------------------------------------------------

            elif class_id == MACHINERY_CLASS_ID:

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 165, 255),
                    2,
                )

                label = (
                    f"Machinery {confidence:.2f}"
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2,
                )

            # ------------------------------------------------
            # Other classes
            # ------------------------------------------------

            else:

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (180, 180, 180),
                    1,
                )

        # ====================================================
        # DRAW DANGEROUS ZONES
        # ====================================================

        for zone_index, polygon in enumerate(
            danger_zones
        ):

            coordinates = np.array(
                [
                    [
                        int(x),
                        int(y),
                    ]
                    for x, y in polygon.exterior.coords
                ],
                dtype=np.int32,
            )

            # Draw polygon outline
            cv2.polylines(
                frame,
                [coordinates],
                isClosed=True,
                color=(0, 0, 255),
                thickness=4,
            )

            # Transparent zone overlay
            overlay = frame.copy()

            cv2.fillPoly(
                overlay,
                [coordinates],
                (0, 0, 255),
            )

            frame = cv2.addWeighted(
                overlay,
                0.20,
                frame,
                0.80,
                0,
            )

            # Zone label
            if len(coordinates) > 0:

                zone_x = int(
                    coordinates[0][0]
                )

                zone_y = int(
                    coordinates[0][1]
                )

                cv2.putText(
                    frame,
                    f"DANGER ZONE {zone_index + 1}",
                    (zone_x, max(30, zone_y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    3,
                )

        # ====================================================
        # HIGHLIGHT PEOPLE INSIDE ZONES
        # ====================================================

        for violation in people_inside:

            person_index = violation[
                "person_index"
            ]

            zone_index = violation[
                "zone_index"
            ]

            if person_index >= len(
                detections
            ):
                continue

            detection = detections[
                person_index
            ]

            bbox = detection[:4]

            x1, y1, x2, y2 = map(
                int,
                bbox
            )

            point = get_bottom_center(
                bbox
            )

            px = int(point[0])
            py = int(point[1])

            # Red person box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                4,
            )

            # Red ground point
            cv2.circle(
                frame,
                (px, py),
                8,
                (0, 0, 255),
                -1,
            )

            # Warning
            cv2.putText(
                frame,
                "DANGER: PERSON IN ZONE",
                (
                    max(10, x1),
                    max(30, y1 - 15),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                3,
            )

        # ====================================================
        # INFORMATION PANEL
        # ====================================================

        panel_height = 145

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (430, panel_height),
            (0, 0, 0),
            -1,
        )

        frame = cv2.addWeighted(
            overlay,
            0.60,
            frame,
            0.40,
            0,
        )

        cv2.putText(
            frame,
            f"Frame: {frame_number}/{total_frames}",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Persons: {len(persons)}",
            (15, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Cones: {len(cones)}",
            (15, 79),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Machinery: {len(machinery)}",
            (15, 106),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Danger zones: {len(danger_zones)}",
            (220, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"People in zone: {people_in_zone_count}",
            (220, 79),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255)
            if people_in_zone_count > 0
            else (255, 255, 255),
            2,
        )

        # ----------------------------------------------------
        # Write frame
        # ----------------------------------------------------

        writer.write(frame)

        # ----------------------------------------------------
        # Console progress
        # ----------------------------------------------------

        if (
            frame_number == 1
            or frame_number % 30 == 0
        ):

            print(
                f"Frame {frame_number}: "
                f"persons={len(persons)}, "
                f"cones={len(cones)}, "
                f"machines={len(machinery)}, "
                f"zones={len(danger_zones)}, "
                f"people_in_zone={people_in_zone_count}"
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    writer.release()

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n========================================")
    print("TEST COMPLETED")
    print("========================================")

    print(
        f"Processed frames           : "
        f"{processed_frames}"
    )

    print(
        f"Frames with >=3 cones       : "
        f"{frames_with_cones}"
    )

    print(
        f"Frames with danger zones    : "
        f"{frames_with_zones}"
    )

    print(
        f"Frames with person in zone  : "
        f"{frames_with_people_in_zone}"
    )

    print(
        f"Total people-in-zone events : "
        f"{total_people_in_zone}"
    )

    print(
        f"Maximum people in zone      : "
        f"{max_people_in_zone}"
    )

    print(
        f"\nOutput video:"
    )

    print(OUTPUT_PATH)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()