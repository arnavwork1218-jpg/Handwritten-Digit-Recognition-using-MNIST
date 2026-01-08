import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os

def predict_custom_image(filename):
    if not os.path.exists(filename):
        print(f"ERROR: Could not find '{filename}'")
        return

    print(f"Processing '{filename}'...")
    try:
        model = tf.keras.models.load_model('mnist_digit_recognizer.h5')
    except:
        print("ERROR: Model not found. Run maincode.py first.")
        return

    # Load image, convert to grayscale, resize to 28x28
    img = Image.open(filename).convert('L')
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img)

    # Invert if background is white (MNIST is black background)
    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    # Normalize and Reshape
    img_array = img_array / 255.0
    img_reshaped = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_reshaped)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    print(f"\nPREDICTION: {predicted_digit} (Confidence: {confidence:.2f}%)")
    
    plt.imshow(img_array, cmap='gray')
    plt.title(f"Predicted: {predicted_digit}")
    plt.axis('off')
    plt.show()

# Run the function on your drawing
predict_custom_image('digit.png')