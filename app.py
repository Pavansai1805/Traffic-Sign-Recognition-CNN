
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "traffic_sign_model.keras"
LABELS_PATH = BASE_DIR / "class_indices.json"

SIGN_NAMES = {
    0: "Speed limit (20 km/h)",
    1: "Speed limit (30 km/h)",
    2: "Speed limit (50 km/h)",
    3: "Speed limit (60 km/h)",
    4: "Speed limit (70 km/h)",
    5: "Speed limit (80 km/h)",
    6: "End of speed limit (80 km/h)",
    7: "Speed limit (100 km/h)",
    8: "Speed limit (120 km/h)",
    9: "No passing",
    10: "No passing for heavy vehicles",
    11: "Right-of-way at next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Heavy vehicles prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of restrictions",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of heavy-vehicle no passing",
}

st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="centered",
)

st.title("🚦 Traffic Sign Recognition")
st.write(
    "Upload a traffic sign image to predict its category "
    "using a trained Convolutional Neural Network (CNN)."
)

@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model(str(MODEL_PATH))
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)

    index_to_class = {
        int(index): int(class_id)
        for class_id, index in class_indices.items()
    }
    return model, index_to_class

if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    st.error("Model files are missing from the repository.")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a traffic sign image",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded traffic sign", width=350)

    if st.button("Predict Traffic Sign", type="primary"):
        with st.spinner("Analyzing image..."):
            model, index_to_class = load_model_and_labels()

            resized = image.resize((30, 30))
            array = np.asarray(resized, dtype=np.float32) / 255.0
            predictions = model.predict(
                np.expand_dims(array, axis=0), verbose=0
            )[0]

            predicted_index = int(np.argmax(predictions))
            class_id = index_to_class[predicted_index]
            confidence = float(predictions[predicted_index]) * 100

        st.success("Prediction completed!")
        st.subheader(SIGN_NAMES.get(class_id, f"Class {class_id}"))
        st.metric("Prediction confidence", f"{confidence:.2f}%")
        st.caption(
            "Confidence is the model's estimated probability, "
            "not a guarantee that the prediction is correct."
        )

st.divider()
st.caption(
    "Built with Python, TensorFlow/Keras and Streamlit. "
    "Educational demonstration only; not for safety-critical driving."
)
