from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import glob
import shutil

app = Flask(__name__)

# Load trained model
model = YOLO("models/best.pt")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():

    image = request.files["image"]

    # Save uploaded image
    upload_path = os.path.join("static", "uploads", image.filename)
    os.makedirs("static/uploads", exist_ok=True)
    image.save(upload_path)

    # Run prediction
    results = model.predict(
        source=upload_path,
        save=True,
        conf=0.25
    )

    # Find latest prediction folder
    predict_folders = glob.glob("runs/detect/predict*")
    latest_folder = max(predict_folders, key=os.path.getctime)

    # Find predicted image
    image_files = (
        glob.glob(os.path.join(latest_folder, "*.jpg"))
        + glob.glob(os.path.join(latest_folder, "*.png"))
        + glob.glob(os.path.join(latest_folder, "*.jpeg"))
    )

    predicted_image = image_files[0]

    # Copy predicted image to static/results
    os.makedirs("static/results", exist_ok=True)

    destination = os.path.join(
        "static",
        "results",
        os.path.basename(predicted_image)
    )

    shutil.copy(predicted_image, destination)

    # Class names
    class_names = {
        0: "Defect 1",
        1: "Defect 2",
        2: "Defect 3",
        3: "Defect 4",
        4: "Smartphone"
    }

    # Get defect name and confidence
    if len(results[0].boxes) > 0:

        defects = []

        for box in results[0].boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Ignore Smartphone class
            if class_id != 4:
                defects.append((class_id, confidence))

        # If a defect is detected
        if defects:

            # Select highest confidence defect
            class_id, confidence = max(
                defects,
                key=lambda x: x[1]
            )

            defect = class_names.get(
                class_id,
                "Unknown"
            )

            confidence = round(
                confidence * 100,
                2
            )

        else:

            defect = "No Defect"
            confidence = 0

    else:

        defect = "No Defect"
        confidence = 0

    return render_template(
        "result.html",
        image_path="/" + destination.replace("\\", "/"),
        defect=defect,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)