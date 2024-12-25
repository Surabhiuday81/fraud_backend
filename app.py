import os
import requests
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS

# Remote model URL
MODEL_URL = "https://drive.google.com/file/d/1FoSvYRTaTxrqnVaflMAA8fMhh98b7hHW/view?usp=drive_link"
MODEL_PATH = "credit_card_fraud_model.pkl"

# Download the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)

# Load the model
model = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    return "Welcome to the Credit Card Fraud Detection API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = [np.array(data['features'])]
    prediction = model.predict(features)
    output = {'prediction': int(prediction[0])}
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
