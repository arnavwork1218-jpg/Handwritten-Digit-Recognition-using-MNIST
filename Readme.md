# Handwritten Digit Recognition Project 

Project Overview
This project builds an AI system capable of recognizing handwritten digits (0-9) from images. I trained a **Convolutional Neural Network (CNN)** using the MNIST dataset and compared its performance against a standard Neural Network to show the importance of deep learning in computer vision.

 Key Features
* **High Accuracy:** The main CNN model achieves **~99% accuracy**.
* **Model Comparison:** Includes a second "Simple Model" script to demonstrate why CNNs are better than basic networks for image tasks.
* **Real-World Testing:** Features a custom script (`CustomTest.py`) that allows the model to predict digits drawn in MS Paint, proving it works outside the training dataset.

 Tech Stack
* **Python** (TensorFlow, Keras)
* **Data Processing:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn

 Results & Analysis

 1. Model Performance
* **CNN Model:** ~99.0% Accuracy (Winner) 
* **Simple Model:** ~97.5% Accuracy

2. Training Graphs
The model learns quickly and stabilizes within 5 epochs.
![Accuracy Graph](Accuracy%20&%20Loss%20Graphs.png.png)

 3. Confusion Matrix
A detailed look at where the model succeeds and where it makes small errors (like confusing 4s and 9s).
![Confusion Matrix](Confusion%20Matrix.png.png)

---

 Real-World Proof
I tested the model on my own handwriting created in MS Paint to ensure it generalizes well.
* **Input:** A handwritten "8".
* **Prediction:** Correctly identified with high confidence.

![Real World Test](Real-World%20Test.png.png)

---

 How to Run
1.  **Install Libraries:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Train the Model:**
    ```bash
    python maincode.py
    ```
3.  **Run the Comparison:**
    ```bash
    python SimpleModel.py
    ```
4.  **Test Your Own Image:**
    (Save a digit as `digit.png` in the folder first)
    ```bash
    python CustomTest.py
    ```




