import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. DATA INGESTION & PREPROCESSING
# ==========================================
def load_data():
    print(" Loading MNIST Data...")
    # Load dataset (already split into train and test sets)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalization: Scale pixel values from 0-255 to 0-1
    # This helps the neural network converge faster
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshaping: Add a channel dimension (28, 28) -> (28, 28, 1)
    # Required because CNNs expect (Batch, Height, Width, Channels)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    print(f"Data Loaded. Train shape: {x_train.shape}, Test shape: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)

# ==========================================
# 2. MODEL ARCHITECTURE (CNN)
# ==========================================
def build_cnn_model():
    print(" Building CNN Model...")
    model = models.Sequential([
        # Layer 1: Convolution (Extract Features) + Pooling (Reduce Size)
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),

        # Layer 2: Convolution + Pooling (Deeper features)
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Flatten: Convert 2D maps to 1D vector
        layers.Flatten(),

        # Dense Layers: Classification
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5), # Regularization to prevent overfitting
        layers.Dense(10, activation='softmax') # Output layer (0-9)
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ==========================================
# 3. TRAINING & EVALUATION
# ==========================================
def plot_history(history):
    # Plot training vs validation accuracy
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.show()

def evaluate_model(model, x_test, y_test):
    print("\n Evaluating on Test Data...")
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f" Final Test Accuracy: {acc*100:.2f}%")

    # Confusion Matrix
    y_pred = np.argmax(model.predict(x_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix (Actual vs Predicted)")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# ==========================================
# MAIN EXECUTION FLOW
# ==========================================
if __name__ == "__main__":
    # 1. Load Data
    (x_train, y_train), (x_test, y_test) = load_data()

    # 2. Build Model
    model = build_cnn_model()
    model.summary() # Print architecture details

    # 3. Train Model
    print("\n Starting Training...")
    history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

    # 4. Results
    plot_history(history)
    evaluate_model(model, x_test, y_test)

    # 5. Save Model (The "Product")
    model.save("mnist_digit_recognizer.h5")
    print("\ Model saved as 'mnist_digit_recognizer.h5'")