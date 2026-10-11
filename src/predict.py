
from pathlib import Path
import json
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "traffic_sign_model.keras"
LABELS_PATH = PROJECT_DIR / "class_indices.json"

# Check image path
if len(sys.argv) < 2:
    print("Usage: python src\\predict.py <image_path>")
    sys.exit(1)

IMAGE_PATH = Path(sys.argv[1])

if not IMAGE_PATH.exists():
    print(f"Image not found: {IMAGE_PATH}")
    sys.exit(1)

# Load trained model
model = tf.keras.models.load_model(str(MODEL_PATH))

# Load class labels
with open(LABELS_PATH, "r") as file:
    class_indices = json.load(file)

# Reverse mapping: model output index -> class ID
index_to_class = {
    index: class_id
    for class_id, index in class_indices.items()
}

# Prepare image
image = Image.open(IMAGE_PATH).convert("RGB")
image = image.resize((30, 30))

image_array = np.array(image, dtype=np.float32) / 255.0
image_array = np.expand_dims(image_array, axis=0)

# Predict
predictions = model.predict(image_array, verbose=0)
predicted_index = int(np.argmax(predictions[0]))
confidence = float(predictions[0][predicted_index]) * 100

traffic_sign_names = {
    "0": "Speed limit (20 km/h)",
    "1": "Speed limit (30 km/h)",
    "2": "Speed limit (50 km/h)",
    "3": "Speed limit (60 km/h)",
    "4": "Speed limit (70 km/h)",
    "5": "Speed limit (80 km/h)",
    "6": "End of speed limit (80 km/h)",
    "7": "Speed limit (100 km/h)",
    "8": "Speed limit (120 km/h)",
    "9": "No passing",
    "10": "No passing for vehicles over 3.5 metric tons",
    "11": "Right-of-way at the next intersection",
    "12": "Priority road",
    "13": "Yield",
    "14": "Stop",
    "15": "No vehicles",
    "16": "Vehicles over 3.5 metric tons prohibited",
    "17": "No entry",
    "18": "General caution",
    "19": "Dangerous curve to the left",
    "20": "Dangerous curve to the right",
    "21": "Double curve",
    "22": "Bumpy road",
    "23": "Slippery road",
    "24": "Road narrows on the right",
    "25": "Road work",
    "26": "Traffic signals",
    "27": "Pedestrians",
    "28": "Children crossing",
    "29": "Bicycles crossing",
    "30": "Beware of ice/snow",
    "31": "Wild animals crossing",
    "32": "End of all speed and passing limits",
    "33": "Turn right ahead",
    "34": "Turn left ahead",
    "35": "Ahead only",
    "36": "Go straight or right",
    "37": "Go straight or left",
    "38": "Keep right",
    "39": "Keep left",
    "40": "Roundabout mandatory",
    "41": "End of no passing",
    "42": "End of no passing by vehicles over 3.5 metric tons",
}


predicted_class = index_to_class[predicted_index]

print("\n--- Traffic Sign Prediction ---")
print(f"Predicted Class ID: {predicted_class}")
print(f"Traffic Sign: {traffic_sign_names[predicted_class]}")
print(f"Confidence: {confidence:.2f}%")



