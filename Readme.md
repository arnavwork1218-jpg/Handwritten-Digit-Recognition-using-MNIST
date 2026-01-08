# Handwritten Digit Recognition System

## Project Overview
This project implements a machine learning pipeline capable of recognizing handwritten digits (0-9) from grayscale images. The system utilizes a Convolutional Neural Network (CNN) trained on the MNIST dataset. A comparative analysis was conducted against a standard Artificial Neural Network (ANN) to demonstrate the efficacy of feature extraction in computer vision tasks.

## Key Objectives
* **Deep Learning Implementation:** Developed a CNN architecture to extract spatial features from 28x28 pixel images.
* **Comparative Analysis:** Benchmarked the CNN against a baseline Dense Neural Network to evaluate performance gains.
* **Generalization Testing:** Validated the model's robustness on unseen, custom-generated handwritten samples.

## Technical Stack
* **Language:** Python
* **Frameworks:** TensorFlow, Keras
* **Data Processing:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn

## Results and Analysis

### 1. Model Performance Metrics
* **CNN Model:** ~99.0% Accuracy
* **Simple Model (ANN):** ~97.5% Accuracy

The Convolutional Neural Network outperformed the baseline model due to its ability to capture spatial hierarchies such as edges and loops.

### 2. Training Dynamics
The model demonstrates rapid convergence and stability within 5 epochs, as evidenced by the accuracy and loss curves.
![Accuracy and Loss Graphs](Accuracy%20&%20Loss%20Graphs.png.png)

### 3. Confusion Matrix
The confusion matrix below illustrates the classification performance across all classes, highlighting minor misclassifications between geometrically similar digits.
![Confusion Matrix](Confusion%20Matrix.png.png)

---

## Inference and Generalization Test
To ensure the model generalizes well to external data, a custom inference script (`CustomTest.py`) was developed to process raw images created outside the training dataset.

**Test Methodology:**
1. A handwritten digit was created using digital drawing tools.
2. The image was preprocessed (grayscale conversion, resizing, inversion) to match the MNIST data distribution.
3. The model predicted the class with a confidence score.

### Test Input Data
The following image (`digit.png`) was used as the raw input for the test:
![Test Input Image](digit.png)

### Prediction Result
The model correctly identified the input digit with high confidence:
![Inference Result](Real-World%20Test.png.png)

---

## Usage Instructions
1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Train the Primary Model:**
    ```bash
    python maincode.py
    ```
3.  **Run Comparative Analysis:**
    ```bash
    python SimpleModel.py
    ```
4.  **Run Custom Inference:**
    Ensure a sample image named `digit.png` is present in the root directory.
    ```bash
    python CustomTest.py
    ```
