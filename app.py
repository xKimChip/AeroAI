# from flask import Flask, render_template, jsonify
# from flask_socketio import SocketIO
# import os
# import json
# import pandas as pd
# import torch
# import numpy as np

# # 1) Import load_model from your AeroEncoder code
# from AeroEncoder import load_model

# app = Flask(__name__)
# socketio = SocketIO(app, async_mode='threading')

# # 2) Load the trained model, scaler, and detector
# model, scaler, detector = load_model()

# def load_flight_data():
#     """
#     Loads flight data from a local JSON file:
#       data/data_MVP.json
#     Must be a list of flight dicts like:
#     [
#       {
#         "id": "Flight123",
#         "alt": 35000,
#         "gs": 450,
#         "heading": 90,
#         "lat": 35.0,
#         "lon": 55.0,
#         "vertRate": 0,
#         "altChange": " ",
#         ...
#       }
#     ]
#     """
#     data_path = os.path.join("data", "combined_MVP.json")

#     if not os.path.exists(data_path):
#         print(f"⚠️  {data_path} not found!")
#         return []

#     try:
#         with open(data_path, "r") as file:
#             flights = json.load(file)
#             if not isinstance(flights, list):
#                 print("⚠️  data_MVP.json must contain a JSON array of flights.")
#                 return []
#             return flights
#     except json.JSONDecodeError as e:
#         print(f"⚠️  JSON parsing error: {e}")
#         return []

# @app.route("/")
# def index():
#     """Renders a front-end UI (templates/index.html) if desired."""
#     return render_template("index.html")

# @app.route("/get_flight_data")
# def get_flight_data():
#     """
#     Returns flight data + anomaly predictions as JSON.

#     The model was trained on 9 columns:
#       1. alt
#       2. gs
#       3. heading
#       4. lat
#       5. lon
#       6. vertRate
#       7. altChange_encoded
#       8. gs_change_rate
#       9. heading_change_rate
#     """
#     flights = load_flight_data()
#     if not flights:
#         return jsonify({"error": "No flight data found"}), 404

#     # Convert to DataFrame for easy manipulation
#     df = pd.DataFrame(flights)

#     # 1) Convert altChange => altChange_encoded
#     if "altChange" in df.columns:
#         df["altChange_encoded"] = df["altChange"].map({" ": 0, "C": 1, "D": -1}).fillna(0)
#     else:
#         df["altChange_encoded"] = 0

#     # 2) Provide placeholder columns if missing
#     if "gs_change_rate" not in df.columns:
#         df["gs_change_rate"] = 0
#     if "heading_change_rate" not in df.columns:
#         df["heading_change_rate"] = 0

#     # 3) The exact 9 features the model/scaler expects
#     features = [
#         "alt",
#         "gs",
#         "heading",
#         "lat",
#         "lon",
#         "vertRate",
#         "altChange_encoded",
#         "gs_change_rate",
#         "heading_change_rate"
#     ]

#     # Fill missing columns with zero
#     for col in features:
#         if col not in df.columns:
#             df[col] = 0

#     # 4) Convert to numpy array & scale
#     X = df[features].values
#     X_scaled = scaler.transform(X)

#     # 5) Predict anomalies
#     anomalies, scores = detector.predict(X_scaled, return_scores=True)

#     # 6) Append results to flights
#     for i, flight in enumerate(flights):
#         flight["anomaly"] = bool(anomalies[i])
#         flight["score"] = float(scores[i])

#     # Return as JSON
#     return jsonify(flights)

# if __name__ == "__main__":
#     # Only inference here (no training). 
#     # The model is already trained & loaded from saved_models/.
#     socketio.run(app, debug=True, host="0.0.0.0", port=5001)



from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
import os
import json
import pandas as pd
import torch
import numpy as np

# Import load_model from AeroEncoder code
from AeroEncoder import load_model

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

# Load the trained model, scaler, and detector at startup
model, scaler, detector = load_model()

# -----------
# GLOBALS for simulating 'one by one' data
# -----------
all_flights = []  # Will hold the entire flight list
current_idx = 0   # Tracks the 'next flight' to serve

def load_all_flights():
    """
    Loads flight data from data/combined_MVP.json (or data_MVP.json)
    and stores them in memory to serve them one by one.
    """
    data_path = os.path.join("data", "combined_MVP.json")
    if not os.path.exists(data_path):
        print(f"⚠️  {data_path} not found!")
        return []

    try:
        with open(data_path, "r") as file:
            flights = json.load(file)
            if not isinstance(flights, list):
                print("⚠️  combined_MVP.json must contain a list of flights.")
                return []
            return flights
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON parsing error: {e}")
        return []

def init_data():
    """
    Loads all_flights into memory once and resets index.
    Call this manually instead of using @app.before_first_request
    to avoid AttributeError in old Flask versions.
    """
    global all_flights, current_idx
    all_flights = load_all_flights()
    current_idx = 0
    print(f"Loaded {len(all_flights)} flights into memory for simulation.")

@app.route("/")
def index():
    """Renders a front-end UI (templates/index.html) if desired."""
    return render_template("index.html")

@app.route("/get_flight_data")
def get_flight_data():
    """
    Returns *all* flight data + anomaly predictions as JSON.

    Model is trained on 9 columns:
      1. alt
      2. gs
      3. heading
      4. lat
      5. lon
      6. vertRate
      7. altChange_encoded
      8. gs_change_rate
      9. heading_change_rate
    """
    if not all_flights:
        return jsonify({"error": "No flight data found"}), 404

    df = pd.DataFrame(all_flights)

    # Convert altChange => altChange_encoded
    if "altChange" in df.columns:
        df["altChange_encoded"] = df["altChange"].map({" ": 0, "C": 1, "D": -1}).fillna(0)
    else:
        df["altChange_encoded"] = 0

    # Provide placeholder columns if missing
    if "gs_change_rate" not in df.columns:
        df["gs_change_rate"] = 0
    if "heading_change_rate" not in df.columns:
        df["heading_change_rate"] = 0

    features = [
        "alt", "gs", "heading", "lat", "lon",
        "vertRate", "altChange_encoded",
        "gs_change_rate", "heading_change_rate"
    ]
    for col in features:
        if col not in df.columns:
            df[col] = 0

    # Scale & Predict
    X = df[features].values
    X_scaled = scaler.transform(X)
    anomalies, scores = detector.predict(X_scaled, return_scores=True)

    # Merge predictions
    flights_copy = []
    for i, flight in enumerate(all_flights):
        flight_copy = dict(flight)  # copy to avoid mutating the global
        flight_copy["anomaly"] = bool(anomalies[i])
        flight_copy["score"] = float(scores[i])
        flights_copy.append(flight_copy)

    return jsonify(flights_copy)

@app.route("/get_next_flight")
def get_next_flight():
    """
    Returns exactly ONE flight record each time it's called,
    simulating a real-time feed. Increments current_idx each time.
    """
    global current_idx, all_flights

    if not all_flights:
        return jsonify({"error": "No flight data found"}), 404
    if current_idx >= len(all_flights):
        return jsonify({"error": "No more flights"}), 404

    flight = all_flights[current_idx]
    current_idx += 1

    # Single-flight DF for anomaly detection
    df = pd.DataFrame([flight])
    if "altChange" in df.columns:
        df["altChange_encoded"] = df["altChange"].map({" ": 0, "C": 1, "D": -1}).fillna(0)
    else:
        df["altChange_encoded"] = 0

    if "gs_change_rate" not in df.columns:
        df["gs_change_rate"] = 0
    if "heading_change_rate" not in df.columns:
        df["heading_change_rate"] = 0

    features = [
        "alt", "gs", "heading", "lat", "lon",
        "vertRate", "altChange_encoded",
        "gs_change_rate", "heading_change_rate"
    ]
    for col in features:
        if col not in df.columns:
            df[col] = 0

    # Scale & Predict
    X = df[features].values
    X_scaled = scaler.transform(X)
    anomalies, scores = detector.predict(X_scaled, return_scores=True)

    flight["anomaly"] = bool(anomalies[0])
    flight["score"] = float(scores[0])

    return jsonify(flight)

@app.route("/reset_sim")
def reset_sim():
    """Resets current_idx so you can simulate again from the start."""
    global current_idx
    current_idx = 0
    return jsonify({"message": "Simulation index reset to 0"})

if __name__ == "__main__":
    # Manually call init_data() before running the server
    init_data()
    socketio.run(app, debug=True, host="0.0.0.0", port=5001)
