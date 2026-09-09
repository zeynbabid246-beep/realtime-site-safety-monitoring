from ultralytics import YOLO

model = YOLO("models/machine/best.pt")

print(model.names)

results = model.predict(
    source="data/images/pexels-ikbalphoto-10421758.jpg",
    conf=0.25,
    imgsz=640,
    save=True
)