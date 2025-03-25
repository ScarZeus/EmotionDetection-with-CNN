import os
import cv2
import numpy as np
import argparse
from tensorflow.keras.models import load_model

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_and_predict_emotion(img_path, model):
    img = cv2.imread(img_path)
    if img is None:
        return "Error: Image not found or invalid path!"
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30,30))
    if len(faces) == 0:
        return "No face detected in the image!"
    
    (x, y, w, h) = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48,48))
    face = face / 255.0
    face = face.reshape(1, 48, 48, 1)
    
    predictions = model.predict(face)[0]
    for label, confidence in zip(emotion_labels, predictions):
        print(f"{label}: {confidence:.4f}")
    
    predicted_emotion = emotion_labels[np.argmax(predictions)]
    return predicted_emotion

def main():
    parser = argparse.ArgumentParser(description="Predict Emotion from an Image")
    parser.add_argument('--model_path', type=str, default='emotion_model.h5', help="Path to the saved model")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the image file for prediction")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    model = load_model(args.model_path)
    predicted_emotion = detect_and_predict_emotion(args.image_path, model)
    print(f"Predicted Emotion: {predicted_emotion}")

if __name__ == '__main__':
    main()
