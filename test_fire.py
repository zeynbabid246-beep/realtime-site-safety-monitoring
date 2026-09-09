from ultralytics import YOLO

model = YOLO("models/fire_smoke/best.pt")

results = model.predict(
    source="data/images/test.webp",
    conf=0.25,
    save=True
)

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        print(f"{class_name}: {confidence:.2f}")