from ultralytics import YOLO
from pathlib import Path
import cv2
import time


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path("models/machine/best.pt")
VIDEO_DIR = Path("data/videos/machine_detection_test")
OUTPUT_DIR = Path("data/output/videos")

# Detection confidence threshold
CONFIDENCE = 0.25

# YOLO image size
IMAGE_SIZE = 640


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


# ============================================================
# CHECK VIDEOS DIRECTORY
# ============================================================

if not VIDEO_DIR.exists():
    raise FileNotFoundError(
        f"Videos directory not found: {VIDEO_DIR}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("Loading YOLO model...")
print("=" * 60)

model = YOLO(str(MODEL_PATH))

print("\nModel loaded successfully!")

print("\nClasses:")
for class_id, class_name in model.names.items():
    print(f"  {class_id}: {class_name}")


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


if len(videos) == 0:
    print("\nNo videos found in:")
    print(VIDEO_DIR.resolve())
    exit()


print("\n" + "=" * 60)
print(f"Found {len(videos)} video(s)")
print("=" * 60)


# ============================================================
# PROCESS EACH VIDEO
# ============================================================

for video_index, video_path in enumerate(videos, start=1):

    print("\n")
    print("=" * 60)
    print(f"VIDEO {video_index}/{len(videos)}")
    print(f"Name: {video_path.name}")
    print("=" * 60)

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        continue


    # --------------------------------------------------------
    # Video information
    # --------------------------------------------------------

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )


    if fps <= 0:
        fps = 30


    duration = total_frames / fps if fps > 0 else 0


    print(f"Resolution : {width} x {height}")
    print(f"FPS        : {fps:.2f}")
    print(f"Frames     : {total_frames}")
    print(f"Duration   : {duration:.2f} seconds")


    # --------------------------------------------------------
    # Output path
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR /
        f"{video_path.stem}_detected.mp4"
    )


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
        print("ERROR: Could not create output video.")
        cap.release()
        continue


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    frame_count = 0

    total_detection_count = 0

    class_detection_count = {}

    max_confidence = {}

    start_time = time.time()


    # ========================================================
    # FRAME LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break


        frame_count += 1


        # ----------------------------------------------------
        # YOLO prediction
        # ----------------------------------------------------

        results = model.predict(
            source=frame,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            verbose=False
        )


        result = results[0]


        # ----------------------------------------------------
        # Analyze detections
        # ----------------------------------------------------

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                class_name = model.names[
                    class_id
                ]


                # Total detections
                total_detection_count += 1


                # Count per class
                if class_name not in class_detection_count:

                    class_detection_count[
                        class_name
                    ] = 0

                class_detection_count[
                    class_name
                ] += 1


                # Maximum confidence
                if class_name not in max_confidence:

                    max_confidence[
                        class_name
                    ] = confidence

                else:

                    max_confidence[
                        class_name
                    ] = max(
                        max_confidence[class_name],
                        confidence
                    )


        # ----------------------------------------------------
        # Draw detections
        # ----------------------------------------------------

        annotated_frame = result.plot()


        # ----------------------------------------------------
        # Add information on frame
        # ----------------------------------------------------

        cv2.putText(
            annotated_frame,
            f"Frame: {frame_count}/{total_frames}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Write frame
        # ----------------------------------------------------

        writer.write(
            annotated_frame
        )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if frame_count % 100 == 0:

            progress = (
                frame_count / total_frames * 100
                if total_frames > 0
                else 0
            )

            print(
                f"Progress: "
                f"{progress:.1f}% "
                f"({frame_count}/{total_frames})"
            )


    # ========================================================
    # RELEASE VIDEO
    # ========================================================

    cap.release()
    writer.release()


    # ========================================================
    # RESULTS
    # ========================================================

    elapsed_time = time.time() - start_time


    print("\n")
    print("-" * 60)
    print("VIDEO COMPLETED")
    print("-" * 60)

    print(
        f"Processed frames : {frame_count}"
    )

    print(
        f"Total detections : {total_detection_count}"
    )

    print(
        f"Processing time  : {elapsed_time:.2f} seconds"
    )


    if elapsed_time > 0:

        processing_fps = (
            frame_count / elapsed_time
        )

        print(
            f"Processing FPS   : "
            f"{processing_fps:.2f}"
        )


    # ========================================================
    # CLASS RESULTS
    # ========================================================

    print("\nDetected classes:")

    if len(class_detection_count) == 0:

        print("  No objects detected.")

    else:

        for class_name in sorted(
            class_detection_count
        ):

            count = class_detection_count[
                class_name
            ]

            confidence = max_confidence[
                class_name
            ]

            print(
                f"  {class_name}: "
                f"{count} detections "
                f"(max confidence: "
                f"{confidence:.3f})"
            )


    print(
        f"\nOutput saved to:"
    )

    print(
        output_path.resolve()
    )


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 60)
print("ALL VIDEOS PROCESSED")
print("=" * 60)

print(
    f"\nResults are available in:"
)

print(
    OUTPUT_DIR.resolve()
)