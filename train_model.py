import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
def load_dataset(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    images = []
    labels = []
    for idx, row in df.iterrows():
        img_path = os.path.join(image_dir, row['filename'])
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        images.append(img_array)
        labels.append(row['label'])
    return np.array(images), np.array(labels)

# Define the model
def create_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Main function to train the model
def train_deepfake_detector(csv_path, image_dir, model_save_path):
    # Load dataset
    images, labels = load_dataset(csv_path, image_dir)
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # Normalize images
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # Create and train the model
    model = create_model(X_train[0].shape)
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    
    # Save the model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

# Example usage
csv_path = 'dataset.csv'
image_dir = 'frames'
model_save_path = 'deepfake_detector.h5'
train_deepfake_detector(csv_path, image_dir, model_save_path)