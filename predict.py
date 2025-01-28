import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('deepfake_detector.h5')

# Function to preprocess video frames
def preprocess_frame(frame, target_size=(128, 128)):
    frame = cv2.resize(frame, target_size)
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

# Function to predict if a video is deepfake
def predict_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = preprocess_frame(frame)
        prediction = model.predict(processed_frame)
        predictions.append(prediction[0][0])
    cap.release()
    avg_prediction = np.mean(predictions)
    return avg_prediction > 0.5  # Assuming 0.5 as the threshold

# Example usage
video_path = 'sample1.mp4'
is_deepfake = predict_video(video_path, model)
print(f"The video is {'a deepfake' if is_deepfake else 'real'}.")