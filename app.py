from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import joblib
import logging
import json

# Initialize Flask app
app = Flask(__name__)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configurations
with open('config.json', 'r') as f:
    config = json.load(f)

# Load the pre-trained model and scaler
try:
    model = load_model(config['model_path'])
    scaler = joblib.load(config['scaler_path'])
except Exception as e:
    logger.error(f"Error loading model or scaler: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from POST data
        content = request.json
        features = np.array([[
            content['Air_temperature'],
            content['Process_temperature'],
            content['Rotational_speed'],
            content['Torque'],
            content['Tool_wear']
        ]])

        # Validate input (you can add more validation logic)
        if features.shape != (1, 5):
            return jsonify({"error": "Invalid input shape", "message": "Ensure you have 5 features for prediction."})

        # Scale the features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)
        prediction = "Failure" if prediction > 0.5 else "No Failure"

        return jsonify({"prediction": prediction})

    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        return jsonify({"error": str(e), "message": "An error occurred during prediction."})

if __name__ == '__main__':
    app.run(debug=True)
