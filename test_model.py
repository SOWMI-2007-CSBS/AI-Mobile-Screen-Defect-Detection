from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Predict on an image
results = model.predict(
    source="static/uploads/mobile screen.avif",  # your test image
    save=True,
    conf=0.25
)

print("Prediction Completed!")