import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model
model = joblib.load('model/heatwave_model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Extract features from the request data
        max_temp = data['max_temp']
        min_temp = data['min_temp']
        humidity = data['humidity']
        wind_speed = data['wind_speed']
        previous_temp = data['previous_temp']

        # Create a DataFrame for prediction
        input_data = pd.DataFrame([[
            max_temp,
            min_temp,
            humidity,
            wind_speed,
            previous_temp
        ]], columns=[
            "max_temp",
            "min_temp",
            "humidity",
            "wind_speed",
            "previous_temp"
        ])

        # Make prediction
        prediction = model.predict(input_data)[0]

        result = "Heatwave" if prediction == 1 else "No Heatwave"

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)