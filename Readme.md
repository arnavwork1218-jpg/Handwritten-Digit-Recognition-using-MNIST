# Handwritten Digit Recognition System (MNIST)

## Project Overview
This project implements a machine learning pipeline capable of recognizing handwritten digits (0-9) from grayscale images. The system utilizes a Convolutional Neural Network (CNN) trained on the MNIST dataset. A comparative analysis was conducted against a standard Artificial Neural Network (ANN) to demonstrate the efficacy of feature extraction in computer vision tasks.

## Objectives
* **Deep Learning Implementation:** Developed a CNN architecture to extract spatial features from 28x28 pixel images.
* **Comparative Analysis:** Benchmarked the CNN against a baseline Dense Neural Network to evaluate performance gains.
* **Generalization Testing:** Validated the model's robustness on unseen, custom-generated handwritten samples created in external software (MS Paint).
* **Metric Evaluation:** Analyzed model performance using Accuracy curves, Loss convergence, and Confusion Matrices.

## Technical Stack
* **Language:** Python 3.12
* **Frameworks:** TensorFlow, Keras
* **Data Processing:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn

## Results and Analysis

### 1. Model Performance Comparison
* **CNN Model (Proposed Method):** ~99.0% Accuracy
* **Simple ANN (Baseline):** ~97.5% Accuracy

The Convolutional Neural Network outperformed the baseline model due to its ability to capture spatial hierarchies, such as edges and loops, which are critical for image recognition.

### 2. Training Dynamics
The model demonstrates rapid convergence and stability within 5 epochs. The plots below show the training and validation accuracy/loss, indicating minimal overfitting.

![Accuracy and Loss Graphs](Accuracy%20&%20Loss%20Graphs.png.png)

### 3. Confusion Matrix
The confusion matrix illustrates the classification performance across all ten classes. The diagonal dominance confirms high precision, while off-diagonal elements highlight minor misclassifications between geometrically similar digits (e.g., 4 and 9).

![Confusion Matrix](Confusion%20Matrix.png.png)

---

## Inference and Generalization Test
To verify that the model generalizes well to new data, a custom inference script (`CustomTest.py`) was used to process raw images created outside the training environment.

**Methodology:**
1. A handwritten digit ("8") was created using digital drawing tools.
2. The image was preprocessed (grayscale conversion, resizing to 28x28, and color inversion) to match the MNIST data distribution.
3. The model predicted the class with a confidence score.

**Prediction Result:**
The model correctly identified the external input digit with high confidence.

![Prediction Result](Real-World%20Test.png.png)

---

## Usage Instructions

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Train the Primary Model (CNN):**
    ```bash
    python maincode.py
    ```

3.  **Run Comparative Analysis (ANN):**
    ```bash
    python SimpleModel.py
    ```

4.  **Run Custom Inference:**
    Ensure a sample image named `digit.png` is present in the root directory.
    ```bash
    python CustomTest.py
    ```
