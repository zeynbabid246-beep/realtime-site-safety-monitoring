from ultralytics import YOLO
from pathlib import Path
import cv2

from src.safety.distance_calculator import (
    calculate_ground_distance
)


# ============================================================
# CONFIGURATION
# ============================================================

# PPE model used for Person detection + tracking
PERSON_MODEL_PATH = Path(
    "person_tracking_test/models/ppe_best.pt"
)

# SiteSense construction machine model
MACHINE_MODEL_PATH = Path(
    "models/machine/best.pt"
)

# Videos to process
VIDEO_DIR = Path(
    "data/videos/machine_detection_test"
)

# Output directory
OUTPUT_DIR = Path(
    "data/output/videos/distance_test"
)

CONFIDENCE_PERSON = 0.25
CONFIDENCE_MACHINE = 0.40

IMAGE_SIZE = 640

# Person class in the PPE model
PERSON_CLASS_ID = 6


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHECK FILES
# ============================================================

if not PERSON_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Person model not found: {PERSON_MODEL_PATH}"
    )

if not MACHINE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Machine model not found: {MACHINE_MODEL_PATH}"
    )

if not VIDEO_DIR.exists():
    raise FileNotFoundError(
        f"Video directory not found: {VIDEO_DIR}"
    )


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 70)
print("LOADING MODELS")
print("=" * 70)

print("\nLoading person tracking model...")
person_model = YOLO(
    str(PERSON_MODEL_PATH)
)

print("Person model loaded successfully!")

print("\nLoading machine detection model...")
machine_model = YOLO(
    str(MACHINE_MODEL_PATH)
)

print("Machine model loaded successfully!")


# ============================================================
# DISPLAY MACHINE CLASSES
# ============================================================

print("\nMachine classes:")

for class_id, class_name in machine_model.names.items():
    print(
        f"  {class_id}: {class_name}"
    )


# ============================================================
# FIND VIDEOS
# ============================================================

video_extensions = [
    "*.mp4",
    "*.avi",
    "*.mov",
    "*.mkv",
    "*.webm"
]

videos = []

for extension in video_extensions:
    videos.extend(
        VIDEO_DIR.glob(extension)
    )

videos = sorted(videos)


if not videos:
    print("\nNo videos found in:")
    print(VIDEO_DIR.resolve())
    exit()


print("\n" + "=" * 70)
print(
    f"Found {len(videos)} video(s)"
)
print("=" * 70)


# ============================================================
# PROCESS EACH VIDEO
# ============================================================

for video_index, video_path in enumerate(
    videos,
    start=1
):

    print("\n")
    print("=" * 70)
    print(
        f"VIDEO {video_index}/{len(videos)}"
    )
    print(
        f"Name: {video_path.name}"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            "ERROR: Could not open video."
        )

        continue


    # --------------------------------------------------------
    # VIDEO INFORMATION
    # --------------------------------------------------------

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

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

    if fps <= 0:
        fps = 30


    print(
        f"Resolution : {width} x {height}"
    )

    print(
        f"FPS        : {fps:.2f}"
    )

    print(
        f"Frames     : {total_frames}"
    )


    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR /
        f"{video_path.stem}_distance.mp4"
    )

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

        print(
            "ERROR: Could not create output video."
        )

        cap.release()

        continue


    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    frame_count = 0

    unique_person_ids = set()

    total_distance_measurements = 0


    # ========================================================
    # FRAME LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1


        # ====================================================
        # 1. PERSON TRACKING
        # ====================================================

        person_results = person_model.track(

            source=frame,

            persist=True,

            tracker="bytetrack.yaml",

            conf=CONFIDENCE_PERSON,

            imgsz=IMAGE_SIZE,

            classes=[
                PERSON_CLASS_ID
            ],

            verbose=False
        )

        person_result = person_results[0]


        # ====================================================
        # 2. MACHINE DETECTION
        # ====================================================

        machine_results = machine_model.predict(

            source=frame,

            conf=CONFIDENCE_MACHINE,

            imgsz=IMAGE_SIZE,

            verbose=False
        )

        machine_result = machine_results[0]


        # ====================================================
        # CREATE MACHINE LIST
        # ====================================================

        machines = []

        if machine_result.boxes is not None:

            for box in machine_result.boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].cpu().tolist()
                )

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                class_name = machine_model.names[
                    class_id
                ]

                machines.append({

                    "class_name": class_name,

                    "confidence": confidence,

                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2
                    ]
                })


        # ====================================================
        # DRAW MACHINE DETECTIONS
        # ====================================================

        annotated_frame = (
            machine_result.plot()
        )


        # ====================================================
        # 3. PROCESS TRACKED PERSONS
        # ====================================================

        if (
            person_result.boxes is not None
            and
            person_result.boxes.id is not None
        ):

            boxes = person_result.boxes

            track_ids = (
                boxes.id
                .int()
                .cpu()
                .tolist()
            )


            for index, track_id in enumerate(
                track_ids
            ):

                unique_person_ids.add(
                    track_id
                )


                # --------------------------------------------
                # PERSON BOUNDING BOX
                # --------------------------------------------

                person_box = boxes.xyxy[
                    index
                ]

                person_bbox = list(
                    map(
                        int,
                        person_box
                        .cpu()
                        .tolist()
                    )
                )


                # --------------------------------------------
                # PERSON BOTTOM CENTER
                # --------------------------------------------

                px1, py1, px2, py2 = (
                    person_bbox
                )

                person_point = (
                    int((px1 + px2) / 2),
                    int(py2)
                )


                # --------------------------------------------
                # FIND CLOSEST MACHINE
                # --------------------------------------------

                closest_machine = None

                minimum_distance = float(
                    "inf"
                )


                for machine in machines:

                    distance = calculate_ground_distance(

                        person_bbox,

                        machine["bbox"]

                    )


                    if distance < minimum_distance:

                        minimum_distance = distance

                        closest_machine = machine


                # --------------------------------------------
                # DRAW PERSON
                # --------------------------------------------

                cv2.rectangle(

                    annotated_frame,

                    (px1, py1),

                    (px2, py2),

                    (255, 255, 255),

                    2
                )


                # --------------------------------------------
                # DRAW PERSON ID
                # --------------------------------------------

                cv2.putText(

                    annotated_frame,

                    f"Person ID: {track_id}",

                    (px1, max(py1 - 10, 20)),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.65,

                    (255, 255, 255),

                    2
                )


                # --------------------------------------------
                # DRAW PERSON POINT
                # --------------------------------------------

                cv2.circle(

                    annotated_frame,

                    person_point,

                    5,

                    (255, 255, 255),

                    -1
                )


                # =================================================
                # DISTANCE RESULT
                # =================================================

                if closest_machine is not None:

                    distance = minimum_distance

                    total_distance_measurements += 1


                    machine_bbox = (
                        closest_machine["bbox"]
                    )


                    mx1, my1, mx2, my2 = (
                        machine_bbox
                    )


                    # --------------------------------------------
                    # MACHINE BOTTOM CENTER
                    # --------------------------------------------

                    machine_point = (

                        int((mx1 + mx2) / 2),

                        int(my2)

                    )


                    # --------------------------------------------
                    # DRAW MACHINE POINT
                    # --------------------------------------------

                    cv2.circle(

                        annotated_frame,

                        machine_point,

                        5,

                        (255, 255, 255),

                        -1
                    )


                    # --------------------------------------------
                    # DRAW DISTANCE LINE
                    # --------------------------------------------

                    cv2.line(

                        annotated_frame,

                        person_point,

                        machine_point,

                        (255, 255, 255),

                        2
                    )


                    # --------------------------------------------
                    # DISTANCE TEXT
                    # --------------------------------------------

                    middle_x = int(
                        (
                            person_point[0]
                            +
                            machine_point[0]
                        ) / 2
                    )

                    middle_y = int(
                        (
                            person_point[1]
                            +
                            machine_point[1]
                        ) / 2
                    )


                    cv2.putText(

                        annotated_frame,

                        f"{distance:.0f} px",

                        (
                            middle_x,
                            middle_y
                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.65,

                        (255, 255, 255),

                        2
                    )


                    # --------------------------------------------
                    # MACHINE NAME
                    # --------------------------------------------

                    cv2.putText(

                        annotated_frame,

                        (
                            f"Nearest: "
                            f"{closest_machine['class_name']}"
                        ),

                        (
                            px1,
                            min(py2 + 25, height - 10)
                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.55,

                        (255, 255, 255),

                        2
                    )


        # ====================================================
        # FRAME INFORMATION
        # ====================================================

        cv2.putText(

            annotated_frame,

            f"Frame: {frame_count}/{total_frames}",

            (20, 35),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (255, 255, 255),

            2
        )


        cv2.putText(

            annotated_frame,

            f"People tracked: {len(unique_person_ids)}",

            (20, 70),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2
        )


        cv2.putText(

            annotated_frame,

            f"Machines: {len(machines)}",

            (20, 105),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2
        )


        # ====================================================
        # WRITE FRAME
        # ====================================================

        writer.write(
            annotated_frame
        )


        # ====================================================
        # PROGRESS
        # ====================================================

        if frame_count % 100 == 0:

            progress = (

                frame_count /
                total_frames *
                100

                if total_frames > 0

                else 0

            )

            print(

                f"Processed "
                f"{frame_count}/{total_frames} "
                f"({progress:.1f}%) | "
                f"Tracked IDs: "
                f"{len(unique_person_ids)}"

            )


    # ========================================================
    # RELEASE
    # ========================================================

    cap.release()

    writer.release()


    print("\nFinished!")

    print(
        f"Unique persons tracked: "
        f"{len(unique_person_ids)}"
    )

    print(
        f"Distance measurements: "
        f"{total_distance_measurements}"
    )

    print(
        f"Output: "
        f"{output_path.resolve()}"
    )


# ============================================================
# FINISHED
# ============================================================

print("\n")

print("=" * 70)

print("ALL VIDEOS COMPLETED")

print("=" * 70)

print(
    "\nResults are available in:"
)

print(
    OUTPUT_DIR.resolve()
)