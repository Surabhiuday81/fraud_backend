from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the model
model = joblib.load('credit_card_fraud_model.pkl')

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
