from ultralytics import YOLO

# Load trained PPE model
model = YOLO("models/ppe/best.pt")

# Video source
video_path = "data/videos/test3.mp4"

# Run prediction
results = model.predict(
    source=video_path,
    conf=0.25,
    imgsz=640,
    save=True,
    stream=True
)

print("\nVIDEO DETECTIONS:\n")

for frame_number, result in enumerate(results, start=1):

    print(f"Frame {frame_number}:")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        print(f"  {class_name}: {confidence:.2f}")