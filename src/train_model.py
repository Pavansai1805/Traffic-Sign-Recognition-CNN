

from pathlib import Path
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Project and dataset paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_DIR / "dataset" / "archive (1)" / "Train"

# Check dataset
if not TRAIN_DIR.exists():
    raise FileNotFoundError(f"Training folder not found: {TRAIN_DIR}")

print("Training folder:", TRAIN_DIR)

# Load and preprocess images
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    str(TRAIN_DIR),
    target_size=(30, 30),
    batch_size=64,
    class_mode="categorical",
    subset="training",
    seed=42
)

validation_data = datagen.flow_from_directory(
    str(TRAIN_DIR),
    target_size=(30, 30),
    batch_size=64,
    class_mode="categorical",
    subset="validation",
    seed=42
)

num_classes = train_data.num_classes
print("Number of classes:", num_classes)
print("Training images:", train_data.samples)
print("Validation images:", validation_data.samples)

# Build CNN model
model = models.Sequential([
    layers.Input(shape=(30, 30, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train model
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=5
)

# Save trained model and class labels
model_path = PROJECT_DIR / "traffic_sign_model.keras"
model.save(str(model_path))

labels_path = PROJECT_DIR / "class_indices.json"
with open(labels_path, "w", encoding="utf-8") as f:
    json.dump(train_data.class_indices, f, indent=4)

print("\nModel saved to:", model_path)
print("Class labels saved to:", labels_path)
print("CNN training completed successfully!")

