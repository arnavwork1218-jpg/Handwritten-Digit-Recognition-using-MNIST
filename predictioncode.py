import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('mnist_digit_recognizer.h5')

# Load dataset again just to grab a random image
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Pick a random image from the test set
index = np.random.randint(0, 10000)
img = x_test[index]
actual_label = y_test[index]

# Preprocess (Scale & Reshape)
input_img = img.astype("float32") / 255.0
input_img = np.expand_dims(input_img, -1) # (28, 28, 1)
input_img = np.expand_dims(input_img, 0)  # (1, 28, 28, 1) - Batch of 1

# Predict
prediction = model.predict(input_img)
predicted_label = np.argmax(prediction)

# Show result
plt.imshow(img, cmap='gray')
plt.title(f"Actual: {actual_label} | Predicted: {predicted_label}")
plt.show()

print(f"Probability distribution: {np.round(prediction, 2)}")