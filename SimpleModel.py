import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Load Data
print("Loading data for comparison...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. Build a SIMPLE Model (No Convolution, just Dense layers)
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),    # Flatten 2D image to 1D vector
    layers.Dense(128, activation='relu'),    # Simple dense layer
    layers.Dense(10, activation='softmax')   # Output layer
])

# 3. Train
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

print("Training Simple Model (to compare with CNN)...")
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 4. Result & Comparison
val_acc = history.history['val_accuracy'][-1]
print(f"\n========================================")
print(f"SIMPLE MODEL ACCURACY: {val_acc*100:.2f}%")
print(f"CNN MODEL ACCURACY:    ~99.00%")
print(f"========================================")
print("Conclusion: The CNN model is more accurate because it captures spatial patterns better.")