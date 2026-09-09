from ultralytics import YOLO

# =========================
# 1. Load trained PPE model
# =========================
model = YOLO("models/ppe/best.pt")

# =========================
# 2. Display model classes
# =========================
print("\nMODEL CLASSES:")
print(model.names)

print("\nMODEL PATH:")
print("models/ppe/best.pt")

# =========================
# 3. Run prediction
# =========================
results = model.predict(
    source="data/images/ppe-inference-im0 (1).png",
    conf=0.25,
    imgsz=640,
    save=True
)

# =========================
# 4. Display detections
# =========================
print("\nDETECTIONS:")

for result in results:
    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        print(f"{class_name}: {confidence:.3f}")