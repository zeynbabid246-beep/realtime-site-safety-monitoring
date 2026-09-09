from ultralytics import YOLO

# =========================
# 1. Load Fire/Smoke model
# =========================
model = YOLO("models/fire_smoke/best.pt")

print("\nMODEL CLASSES:")
print(model.names)

print("\nMODEL PATH:")
print("models/fire_smoke/best.pt")

# =========================
# 2. Video source
# =========================
video_path = "data/videos/fire_test2.mp4"

# =========================
# 3. Run video detection
# =========================
results = model.predict(
    source=video_path,
    conf=0.25,
    imgsz=640,
    save=True,
    stream=True
)

# =========================
# 4. Display detections
# =========================
print("\nVIDEO DETECTIONS:\n")

for frame_number, result in enumerate(results, start=1):

    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        detections.append(
            f"{class_name}: {confidence:.2f}"
        )

    if detections:
        print(f"Frame {frame_number}:")
        for detection in detections:
            print(f"  {detection}")