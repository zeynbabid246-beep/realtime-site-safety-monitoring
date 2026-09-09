from ultralytics import YOLO
from pathlib import Path
import cv2


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path("models/ppe_best.pt")
VIDEO_DIR = Path("videos")
OUTPUT_DIR = Path("output")

CONFIDENCE = 0.25
IMAGE_SIZE = 640

# Person class in our PPE model
PERSON_CLASS_ID = 6


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))

print("=" * 60)
print("PERSON TRACKING")
print("=" * 60)

print("\nModel classes:")

for class_id, class_name in model.names.items():
    print(f"{class_id}: {class_name}")


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
    videos.extend(VIDEO_DIR.glob(extension))

videos = sorted(videos)

if not videos:
    print("\nNo videos found.")
    exit()


print(f"\nFound {len(videos)} video(s).")


# ============================================================
# PROCESS VIDEOS
# ============================================================

for video_path in videos:

    print("\n" + "=" * 60)
    print(f"Processing: {video_path.name}")
    print("=" * 60)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    if fps <= 0:
        fps = 30


    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR /
        f"{video_path.stem}_tracked.mp4"
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


    frame_count = 0

    unique_person_ids = set()


    # ========================================================
    # FRAME LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1


        # ----------------------------------------------------
        # TRACKING
        # ----------------------------------------------------

        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            classes=[PERSON_CLASS_ID],
            verbose=False
        )


        result = results[0]


        # ----------------------------------------------------
        # DRAW TRACKING
        # ----------------------------------------------------

        annotated_frame = result.plot()


        # ----------------------------------------------------
        # GET TRACK IDs
        # ----------------------------------------------------

        if result.boxes is not None:

            if result.boxes.id is not None:

                track_ids = (
                    result.boxes.id
                    .int()
                    .cpu()
                    .tolist()
                )

                for track_id in track_ids:

                    unique_person_ids.add(
                        track_id
                    )


        # ----------------------------------------------------
        # DISPLAY NUMBER OF PEOPLE
        # ----------------------------------------------------

        current_people = 0

        if result.boxes is not None:

            current_people = len(
                result.boxes
            )


        cv2.putText(
            annotated_frame,
            f"People: {current_people}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )


        cv2.putText(
            annotated_frame,
            f"Tracked IDs: {len(unique_person_ids)}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # WRITE FRAME
        # ----------------------------------------------------

        writer.write(
            annotated_frame
        )


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if frame_count % 100 == 0:

            print(
                f"Processed {frame_count} frames | "
                f"Unique IDs: {len(unique_person_ids)}"
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
        f"Output: "
        f"{output_path.resolve()}"
    )


print("\n" + "=" * 60)
print("ALL VIDEOS COMPLETED")
print("=" * 60)