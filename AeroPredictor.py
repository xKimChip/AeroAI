import pandas as pd
from AeroEncoder import minimal_preprocess, load_model, AnomalyDetector
import json, redis
import numpy as np
model, scaler, detector = load_model()

r = redis.Redis(host='localhost', port=6379, db=0)

def min_preprocess(filepath: str):
    with open(filepath, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    features = ['alt', 'gs', 'heading', 'vertRate']
        
    # Convert to numeric, keeping all values
    for col in features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                # print(f"Found {nan_count} missing/invalid values in {col}")
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                
    df['altChange'] = df['altChange'].fillna(0)

    if 'vertRate' not in df.columns:
        df['vertRate'] = 0

    df['altChange_encoded'] = df['altChange'].map({' ': 0, 'C': 1, 'D': -1}).fillna(0)
    
    key_size = r.llen(df['id'])
        
    ## Time Difference, 
    
    
    
    # if 'pitr' in df.columns and 'gs' in df.columns and key_size > 1:
        
    #     r.lset("THY6526-1745891097-ed-1227p", 0, json.dumps(temp))
    #     for i in range(key_size):
    #         r.lindex(df['id'], i)

    
    return df



        

filename = "data/data_MVP.json"            # file to be read
#min_preprocess(filename)
#print(type(r.lindex("THY6526-1745891097-ed-1227p", 0)))
bytes_data = r.lindex("THY6526-1745891097-ed-1227p", 0)
temp = json.loads(bytes_data)
temp['time_diff'] = 15
#print (temp)
print(r.lset("THY6526-1745891097-ed-1227p", 0, json.dumps(temp)))