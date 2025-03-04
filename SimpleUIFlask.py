from flask import Flask, render_template, jsonify, request
import torch
import pandas as pd
import numpy as np
import joblib
import json
from flask_socketio import SocketIO
import nbformat
from nbconvert import PythonExporter

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

# Function to load and execute Jupyter Notebook
def load_notebook(notebook_path):
    with open(notebook_path) as f:
        notebook = nbformat.read(f, as_version=4)
    exporter = PythonExporter()
    python_code, _ = exporter.from_notebook_node(notebook)
    exec(python_code, globals())  # Execute the notebook in the global scope

# Load and execute AeroEncoder.ipynb
load_notebook("AeroEncoder.ipynb")

# Load mock flight data (Replace with live data in production)
def load_flight_data():
    try:
        with open("static/data/flight_data.json", "r") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading flight data: {e}")
        return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_flight_data')
def get_flight_data():
    flights = load_flight_data()
    if not flights:
        return jsonify({"error": "No flight data available"}), 500
    
    df = pd.DataFrame(flights)
    
    # Extract features and preprocess
    features = ['alt', 'gs', 'heading', 'lat', 'lon', 'vertRate', 'altChange_encoded']
    X = df[features].values
    X_scaled = scaler.transform(X)
    
    # Predict anomalies
    anomalies, scores = detector.predict(X_scaled, return_scores=True)
    
    for i, flight in enumerate(flights):
        flight['anomaly'] = bool(anomalies[i])
        flight['score'] = float(scores[i])

    return jsonify(flights)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0')
