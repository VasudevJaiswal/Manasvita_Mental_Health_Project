from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask_cors import CORS
import base64
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
model = load_model("fer2013_cnn_model.h5")

# Emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    stress_level = "Neutral"
    predicted_label = None

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = img_to_array(roi) / 255.0
        roi = np.expand_dims(roi, axis=0)

        predictions = model.predict(roi)
        predicted_label = emotion_labels[np.argmax(predictions)]

        if predicted_label in ["Angry", "Fear", "Sad"]:
            stress_level = "High"
        elif predicted_label == "Surprise":
            stress_level = "Moderate"
        else:
            stress_level = "Low"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame, stress_level, predicted_label

@app.route('/videofeed', methods=["POST"])
def video_feed():
    data = request.get_json()
    image_data = data.get("image")

    # Decode Base64 image
    img = np.frombuffer(base64.b64decode(image_data.split(',')[1]), dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    frame, stress_level, predicted_label = detect_emotion(img)

    # Encode processed frame to Base64 for response
    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()

    return jsonify({
        "image": base64.b64encode(frame_bytes).decode("utf-8"),
        "stress_level": stress_level,
        "predicted_label": predicted_label
    }), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
