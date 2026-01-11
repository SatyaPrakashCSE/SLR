import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
from keras.models import model_from_json

app = Flask(__name__)

# Load model
with open("signlanguagedetectionmodel48x48.json", "r") as f:
    model = model_from_json(f.read())

model.load_weights("signlanguagedetectionmodel48x48.h5")

labels = ['A', 'M', 'N', 'S', 'T', 'blank']

def extract_features(image):
    image = image.reshape(1, 48, 48, 1)
    return image / 255.0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    image_data = base64.b64decode(data.split(',')[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # SAME preprocessing as your original code
    roi = frame[40:300, 0:300]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    processed = extract_features(resized)

    pred = model.predict(processed, verbose=0)
    label = labels[np.argmax(pred)]
    confidence = float(np.max(pred) * 100)

    return jsonify({
        "label": label,
        "confidence": round(confidence, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
