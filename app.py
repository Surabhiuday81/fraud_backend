import os
import requests
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS

# Remote model URL
MODEL_URL = "https://drive.google.com/uc?export=download&id=1FoSvYRTaTxrqnVaflMAA8fMhh98b7hHW"
MODEL_PATH = "credit_card_fraud_model.pkl"

# Download the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    try:
        response = requests.get(MODEL_URL, timeout=30)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading model: {e}")
        model = None  # Gracefully handle the failure
    else:
        # Only load the model if the download was successful
        model = joblib.load(MODEL_PATH)

if model is None:
    print("Model failed to load. Exiting.")
    exit(1)

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
    app.run(debug=False)  # Set debug=False for production
