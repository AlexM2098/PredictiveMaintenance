from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained model and scaler
model = load_model('trained_model.h5')
scaler = joblib.load('standard_scaler.pkl')

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

        # Scale the features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)
        prediction = "Failure" if prediction > 0.5 else "No Failure"

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e), "message": "An error occurred during prediction."})

if __name__ == '__main__':
    app.run(debug=True)
