import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Constants
IMG_SIZE = 50
DATA_DIR = 'train/train'
MODEL_NAME = 'leaf_disease_classifier.keras'

# Labeling function
def label_img(filename):
    label_char = filename[0].lower()
    if label_char == 'h':
        return [1, 0, 0, 0]
    elif label_char == 'b':
        return [0, 1, 0, 0]
    elif label_char == 'v':
        return [0, 0, 1, 0]
    elif label_char == 'l':
        return [0, 0, 0, 1]
    else:
        raise ValueError(f"Unknown label prefix in filename: {filename}")

# Create training data
def create_train_data():
    training_data = []
    for img_name in tqdm(os.listdir(DATA_DIR)):
        try:
            label = label_img(img_name)
            path = os.path.join(DATA_DIR, img_name)
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            training_data.append([img, label])
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue
    np.random.shuffle(training_data)
    return training_data

# Load and preprocess data
train_data = create_train_data()
X = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
Y = np.array([i[1] for i in train_data])

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

# Build the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Save the model
model.save(MODEL_NAME)
print(f"Model saved as {MODEL_NAME}")
