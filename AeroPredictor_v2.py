import pandas as pd
import numpy as np
import json
import redis
from AeroEncoder import load_model

# Load the model
model, scaler, detector = load_model()

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)
print("Connected to Redis")
print("Checking sample data...")
sample_key = r.keys('*')[0].decode('utf-8')
print(f"Sample key: {sample_key}")
sample_data = r.lindex(sample_key, 0)
print(f"Sample data: {sample_data}")
# when i tested last, im connecting to redis correctly, but im not processing the anomolies correctly:
# Processed 0 aircraft
# Found anomalies in 0 aircraft

try:
    parsed_data = json.loads(sample_data)
    print("Data is valid JSON")
    print("Keys:", list(parsed_data.keys()))
except Exception as e:
    print(f"Error parsing data: {e}")

def process_redis_data(aircraft_id):
    """
    Process data for a specific aircraft ID from Redis
    """
    # Get all data points for this aircraft
    data_points = []
    list_length = r.llen(aircraft_id)
    
    if list_length == 0:
        return None
    
    # Get all items in the list
    for i in range(list_length):
        bytes_data = r.lindex(aircraft_id, i)
        if bytes_data:
            data_point = json.loads(bytes_data)
            data_points.append(data_point)
    
    df = pd.DataFrame(data_points)
    
    # Check features (add vertRate if missing)
    required_features = ['alt', 'gs', 'heading', 'vertRate', 'pitr']
    for feature in required_features:
        if feature not in df.columns:
            if feature == 'vertRate':
                df[feature] = 0
            else:
                return None
    
    # Convert columns to numeric
    for col in ['alt', 'gs', 'heading', 'vertRate', 'pitr']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle missing values by adding median values
    for col in ['alt', 'gs', 'heading', 'vertRate']:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Create altChange_encoded feature
    if 'altChange' in df.columns:
        df['altChange_encoded'] = df['altChange'].map({' ': 0, 'C': 1, 'D': -1}).fillna(0)
    else:
        df['altChange_encoded'] = 0
    
    # Calculate time differences and rates of change if we have multiple points
    if len(df) > 1:
        # Sort by pitr time
        df = df.sort_values('pitr').reset_index(drop=True)
        
        # Calculate time differences
        df['time_diff'] = df['pitr'].diff().fillna(1)  # First point gets time_diff of 1
        
        # Compute rate of change (change per second)
        df['gs_change_rate'] = df['gs'].diff() / df['time_diff']
        df['heading_change_rate'] = df['heading'].diff().apply(lambda x: (x + 180) % 360 - 180) / df['time_diff']
        
        # Replace infinity or extreme values with 0 or NaN
        df['gs_change_rate'] = df['gs_change_rate'].replace([np.inf, -np.inf], 0).fillna(0)
        df['heading_change_rate'] = df['heading_change_rate'].replace([np.inf, -np.inf], 0).fillna(0)
    else:
        # For single data points, set rates to 0
        df['time_diff'] = 0
        df['gs_change_rate'] = 0
        df['heading_change_rate'] = 0
    
    return df

def detect_anomalies():
    """
    Process all aircraft in Redis and detect anomalies
    """
    # Get all keys (aircraft IDs)
    all_keys = r.keys('*')
    results = {}
    
    for key in all_keys:
        aircraft_id = key.decode('utf-8')
        df = process_redis_data(aircraft_id)
        
        if df is not None and not df.empty:

            print(f"  Successfully processed {len(df)} data points")

            # Prepare features for the model
            features = ['alt', 'gs', 'heading', 'vertRate', 'altChange_encoded']
            
            # Add engineered features if they exist
            if 'gs_change_rate' in df.columns:
                features.append('gs_change_rate')
            if 'heading_change_rate' in df.columns:
                features.append('heading_change_rate')
            
            # Make sure all required features exist
            missing_features = [f for f in features if f not in df.columns]
            for feature in missing_features:
                df[feature] = 0
            
            # Extract features for the model
            X = df[features].values
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Detect anomalies
            anomalies = detector.predict(X_scaled)
            
            # Store results
            results[aircraft_id] = {
                'total_points': len(df),
                'anomalies_detected': int(anomalies.sum()) if hasattr(anomalies, 'sum') else 0,
                'is_anomalous': bool(anomalies.any()) if hasattr(anomalies, 'any') else False,
                'data': df.to_dict('records')
            }
            
            # Update Redis with anomaly flags (optional)
            for i, is_anomaly in enumerate(anomalies):
                if isinstance(is_anomaly, bool) or isinstance(is_anomaly, np.bool_):
                    data_point = json.loads(r.lindex(aircraft_id, i))
                    data_point['is_anomaly'] = bool(is_anomaly)
                    r.lset(aircraft_id, i, json.dumps(data_point))
        else:
            print(f"  No valid data for {aircraft_id} or insufficient data points")
    
    return results

# Run anomaly detection
if __name__ == "__main__":
    results = detect_anomalies()
    print(f"Processed {len(results)} aircraft")
    
    # Print anomaly summary
    anomalous_aircraft = [aid for aid, data in results.items() if data['is_anomalous']]
    print(f"Found anomalies in {len(anomalous_aircraft)} aircraft")
    
    # Print details of anomalous aircraft
    for aid in anomalous_aircraft:
        print(f"Aircraft {aid}: {results[aid]['anomalies_detected']} anomalies out of {results[aid]['total_points']} points")