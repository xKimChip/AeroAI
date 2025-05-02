import redis
import json
import pandas as pd
import torch
import numpy as np
from datetime import datetime
import time
import os

from AeroEncoder import (
    minimal_preprocess,
    load_model,
    flight_prediction,
    EnhancedFlightPrediction,
    AnomalyDetector
)

r = redis.Redis(host='localhost', port=6379, db=0)

model, scaler, detector = load_model(
    model_path='saved_models/flight_anomaly_model.pth',
    scaler_path='saved_models/scaler.joblib',
    detector_path='saved_models/detector.joblib'
)

def get_all_flights_from_redis():
    """
    Get all current flight data from Redis.
    Returns a dictionary with flight IDs as keys and flight data as values.
    """
    flights = {}
    # get keys
    all_keys = r.keys('*')
    
    for key in all_keys:
        flight_id = key.decode('utf-8')

        if not flight_id.startswith(('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z')):
            continue
            
        # Get the most recent data for this flight
        try:
            flight_data_json = r.lindex(flight_id, 0)
            if flight_data_json:
                flight_data = json.loads(flight_data_json)
                flight_data['id'] = flight_id
                flights[flight_id] = flight_data
        except Exception as e:
            print(f"Error processing flight {flight_id}: {e}")
    
    return flights

def detect_anomalies_in_redis_data():
    """
    Retrieve flight data from Redis, process it, and detect anomalies.
    """
    # Get all flights from Redis
    flights = get_all_flights_from_redis()
    
    if not flights:
        print("No flight data found in Redis")
        return
    
    print(f"Retrieved {len(flights)} flights from Redis")
    
    # Convert to DataFrame
    flights_df = pd.DataFrame(list(flights.values()))
    
    # Apply minimal preprocessing
    processed_df = minimal_preprocess_redis_data(flights_df)
    
    # Detect anomalies
    anomalies = flight_prediction(processed_df, model, scaler, detector)
    
    # Process and report results
    process_anomaly_results(anomalies, flights_df)

def minimal_preprocess_redis_data(df):
    """
    Adapt the minimal_preprocess function for Redis data.
    Handle any special formatting needed for Redis-sourced data.
    """
    # Make sure required columns exist
    required_columns = ['alt', 'gs', 'heading', 'vertRate', 'altChange']
    
    # Add missing columns with default values
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Handle altChange encoding
    if 'altChange' in df.columns:
        df['altChange'] = df['altChange'].fillna(' ')
        df['altChange_encoded'] = df['altChange'].map({' ': 0, 'C': 1, 'D': -1}).fillna(0)
    else:
        df['altChange_encoded'] = 0
    
    # Calculate rate of change if we have multiple records for the same flight
    if len(df) > 1 and 'pitr' in df.columns:
        # Group by flight ID and calculate time differences
        df = df.sort_values(['id', 'pitr']).reset_index(drop=True)
        
        # Compute time difference in seconds
        df['time_diff'] = df.groupby('id')['pitr'].diff().fillna(1)
        
        # Compute rates of change
        df['gs_change_rate'] = df.groupby('id')['gs'].diff() / df['time_diff']
        df['heading_change_rate'] = df.groupby('id')['heading'].diff().apply(
            lambda x: (x + 180) % 360 - 180) / df['time_diff']
        
        # Replace infinity or extreme values
        df['gs_change_rate'] = df['gs_change_rate'].replace([np.inf, -np.inf], 0)
        df['heading_change_rate'] = df['heading_change_rate'].replace([np.inf, -np.inf], 0)
        
        # Handle NaN values
        df['gs_change_rate'] = df['gs_change_rate'].fillna(0)
        df['heading_change_rate'] = df['heading_change_rate'].fillna(0)
    else:
        # Add columns with default values if we don't have enough data
        df['gs_change_rate'] = 0
        df['heading_change_rate'] = 0
    
    return df

def process_anomaly_results(anomalies, original_df):
    """
    Process anomaly detection results and take appropriate actions.
    """
    # Merge anomaly results with original data
    result_df = pd.DataFrame(anomalies)
    
    # Filter to show only anomalies
    anomalies_only = result_df[result_df['anomaly'] == True]
    
    if len(anomalies_only) > 0:
        print(f"\n🚨 DETECTED {len(anomalies_only)} ANOMALIES 🚨")
        
        # Display each anomaly with relevant details
        for idx, row in anomalies_only.iterrows():
            flight_id = original_df.iloc[idx]['id']
            
            print(f"\nAnomalous Flight: {flight_id}")
            print(f"Position: Lat {original_df.iloc[idx]['lat']:.4f}, Lon {original_df.iloc[idx]['lon']:.4f}")
            print(f"Altitude: {original_df.iloc[idx]['alt']} ft")
            print(f"Ground Speed: {original_df.iloc[idx]['gs']} knots")
            print(f"Heading: {original_df.iloc[idx]['heading']}°")
            
            # Optionally store back to Redis with anomaly flag
            anomaly_key = f"anomaly:{flight_id}"
            anomaly_data = original_df.iloc[idx].to_dict()
            anomaly_data['anomaly_detected'] = True
            anomaly_data['detection_time'] = datetime.now().isoformat()
            
            r.set(anomaly_key, json.dumps(anomaly_data))
            r.expire(anomaly_key, 3600)  # Keep for 1 hour
    else:
        print("No anomalies detected in current data")

def main():
    """
    Main function to run periodic anomaly detection on Redis data.
    """
    print("Starting flight anomaly detection service...")
    
    try:
        while True:
            print(f"\n[{datetime.now().isoformat()}] Checking for anomalies...")
            detect_anomalies_in_redis_data()
            
            # Wait before next check - adjust as needed
            time.sleep(15)  # Check every 15 seconds
            
    except KeyboardInterrupt:
        print("\nShutting down anomaly detection service...")
    except Exception as e:
        print(f"Error in anomaly detection service: {e}")

if __name__ == "__main__":
    main()