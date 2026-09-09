import cv2

from src.hazard.hazard_detector import HazardDetector


VIDEO_PATH = "data/videos/hazard_test.mp4"


detector = HazardDetector(
    confidence=0.25,
    image_size=640,
)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")


frame_count = 0

while True:
    success, frame = cap.read()

    if not success:
        break

    detections = detector.track(frame)

    persons = [
        d for d in detections
        if d["class_id"] == detector.PERSON_CLASS
    ]

    machines = [
        d for d in detections
        if d["class_id"] == detector.MACHINERY_CLASS
    ]

    cones = [
        d for d in detections
        if d["class_id"] == detector.CONE_CLASS
    ]

    print(
        f"Frame {frame_count}: "
        f"persons={len(persons)}, "
        f"machines={len(machines)}, "
        f"cones={len(cones)}"
    )

    frame_count += 1

cap.release()

print("TEST COMPLETED")