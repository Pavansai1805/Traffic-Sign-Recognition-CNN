# Traffic Sign Recognition using CNN

## Project Overview
This project uses a Convolutional Neural Network (CNN) to classify traffic sign images into different traffic sign categories. It demonstrates the use of deep learning and computer vision for traffic sign recognition.

## Objectives
- Build a CNN model for traffic sign classification.
- Train and evaluate the model using a traffic sign image dataset.
- Predict traffic sign categories from input images.
- Display the predicted class and confidence score.

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Convolutional Neural Networks (CNN)
- Git and GitHub

## Model Performance
- Test Accuracy: **93.74%**
- Test Loss: **0.2534**
- Validation Accuracy: **90.04%** after 5 epochs

## Project Structure
```text
Traffic-Sign-Recognition-CNN/
├── src/
│   ├── explore_dataset.py
│   ├── view_images.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
├── class_indices.json
├── traffic_sign_model.keras
└── .gitignore
```

## Example Prediction
- Predicted Class ID: 16
- Traffic Sign: Vehicles over 3.5 metric tons prohibited
- Confidence: 99.88%

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Pavansai1805/Traffic-Sign-Recognition-CNN.git
   cd Traffic-Sign-Recognition-CNN
   ```

2. Install the required libraries:
   ```bash
   pip install tensorflow numpy matplotlib pillow
   ```

3. Download and extract the traffic sign dataset into the `dataset/` directory.

4. Train the model:
   ```bash
   python src/train_model.py
   ```

5. Evaluate the model:
   ```bash
   python src/evaluate_model.py
   ```

6. Predict a traffic sign:
   ```bash
   python src/predict.py "path/to/traffic_sign_image.png"
   ```

## Future Improvements
- Improve model accuracy through hyperparameter tuning.
- Add a user-friendly web interface.
- Test the model on real-world traffic sign images.

## Author
Pavan Sai

## Disclaimer
This project is intended for educational purposes and is not designed for safety-critical driving applications.
