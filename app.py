from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('traffic_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Traffic-Prediction-App/predict', methods=['POST'])
def predict():
    hour = int(request.form['hour'])
    day_of_week = int(request.form['day_of_week'])
    junction = int(request.form['junction'])

    # Predict the number of vehicles
    prediction = model.predict(np.array([[hour, day_of_week, junction]]))
    
    return jsonify({'predicted_vehicles': int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
