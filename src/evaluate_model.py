
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset" / "archive (1)"
TEST_DIR = DATASET_DIR / "Test"
CSV_PATH = DATASET_DIR / "Test.csv"

# Load trained model
model = tf.keras.models.load_model(
    str(PROJECT_DIR / "traffic_sign_model.keras")
)

# Load class mapping created during training
with open(PROJECT_DIR / "class_indices.json", "r") as f:
    class_indices = json.load(f)

# Read test labels
df = pd.read_csv(CSV_PATH)
print("CSV columns:", df.columns.tolist())
print("Test images:", len(df))

# Map actual class IDs to model output indices
images = []
labels = []

for _, row in df.iterrows():
    image_path = DATASET_DIR / str(row["Path"])
    if not image_path.exists():
        image_path = TEST_DIR / Path(str(row["Path"])).name

    if not image_path.exists():
        continue

    image = Image.open(image_path).convert("RGB")
    image = image.resize((30, 30))
    images.append(np.array(image, dtype=np.float32) / 255.0)

    class_id = str(int(row["ClassId"]))
    labels.append(class_indices[class_id])

if not images:
    raise ValueError("No test images loaded. Check the image paths in Test.csv.")

X_test = np.array(images)
y_test = np.array(labels)

# Evaluate model
loss, accuracy = model.evaluate(X_test, tf.keras.utils.to_categorical(
    y_test, num_classes=len(class_indices)
), verbose=1)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
