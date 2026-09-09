from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = Path("models/hazard/best.pt")
SOURCE = Path("data/videos/hazard_test4.mp4")

model = YOLO(MODEL_PATH)

results = model.predict(
    source=SOURCE,
    conf=0.15,
    imgsz=640,
    classes=[6],          # Safety Cone
    save=True,
    verbose=True
)

print("\nSafety Cone test completed.")