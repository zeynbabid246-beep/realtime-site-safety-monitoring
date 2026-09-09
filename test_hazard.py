from ultralytics import YOLO

MODEL_PATH = "models/hazard/best.pt"
VIDEO_PATH = "data/videos/hazard_test3.mp4"

model = YOLO(MODEL_PATH)

results = model.predict(
    source=VIDEO_PATH,
    imgsz=640,
    conf=0.25,
    save=True,
    stream=True,
    max_det=495
)

for result in results:
    pass

print("Hazard detection test completed.")