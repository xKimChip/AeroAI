import pandas as pd
from AeroEncoder import load_model, flight_prediction
import json, redis
import numpy as np
model, scaler, detector = load_model()

# Holds a short-term history of a flight
rMem = redis.Redis(host='localhost', port=6379, db=0)

# Holds long-term history of all flights over the RunTime
REDIS_HOST = 'redis-14815.c289.us-west-1-2.ec2.redns.redis-cloud.com'
REDIS_PORT = 14815
REDIS_USER = 'default'
REDIS_PASSWORD = 'J9rkVUXSCnwbadDNPaicB2YnFe4EZjxo'

rDisk = redis.Redis(
   host=REDIS_HOST, port=REDIS_PORT,
   username=REDIS_USER,
   password=REDIS_PASSWORD,
   socket_connect_timeout=5
   )

EXPIRE_TIME = 300 # 5 minutes
EXPIRE_TIME_LONG = 86400 # 24 hours

def process_for_redis(data, eTime=EXPIRE_TIME):
   # use a list to get the recent history of a data point.
   if 'altChange' not in data:
      data['altChange'] = ' '
   
   data_vals = {
      'pitr': data['pitr'],
      'lat': data['lat'],
      'lon': data['lon'],
      'alt': data['alt'],
      'vertRate': data.get('vertRate', 0),
      'gs': data['gs'],
      'heading': data['heading'],
      'altChange_encoded': {' ': 0, 'C': 1, 'D': -1}.get(data['altChange'], ' '),
   }
   key = data['id']
   
   # Processing the data
   if rMem.exists(key) > 0:
      temp = json.loads(rMem.lindex(key, 0))                   # get the first item in the list
      data_vals['time_diff'] = int(data['pitr']) - int(temp['pitr']) if int(data['pitr']) - int(temp['pitr']) > 0 else 1 # compute the time difference
      data_vals['gs_change_rate'] = (float(data['gs']) - float(temp['gs'])) / data_vals['time_diff']                     # comute the ground speed change rate
      heading_diff = ((float(data['heading']) - float(temp['heading'])) + 180) % 360 - 180
      data_vals['heading_change_rate'] = heading_diff / data_vals['time_diff']                                           # compute the heading change rate                          
   else:
      data_vals['time_diff'] = 1                            # if the list does not exist, set the first time_diff to 1
      data_vals['gs_change_rate'] = 0                       # if the list does not exist, set the first gs_change_rate to 0   
      data_vals['heading_change_rate'] = 0                  # if the list does not exist, set the first heading_change_rate to 0
      
   
   rMem.lpush(key, json.dumps(data_vals)) # push the columns of data to the list
   if rMem.ttl(key) == -1: # set the expirate time to ETIME
      rMem.expire(key, eTime)
   else:
      rMem.expire(key, eTime, xx=True)
    
   rMem.ltrim(key, 0, 10) # keep only the last 10 items
   
   return key, data_vals

def move_to_predict(data):
   # Process the data and send it to Redis in Memory
   key, data_vals = process_for_redis(data)
   
   df = pd.DataFrame(data_vals, index=[0])


   results_df = flight_prediction(df, model, scaler, detector)
   
   data_vals['anomaly_score'] = (results_df['anomaly']).to_dict()
   
   rDisk.lpush(key, json.dumps(data_vals)) # push the columns of data to the list
   if rDisk.ttl(key) == -1: # set the expirate time to ETIME
      rDisk.expire(key, EXPIRE_TIME_LONG)
   else:
      rMem.expire(key, EXPIRE_TIME_LONG, xx=True)
    
    
if __name__ == "__main__":
    filename = "data/data_MVP.json"            # file to be read
    #min_preprocess(filename)
    #print(type(r.lindex("THY6526-1745891097-ed-1227p", 0)))
    bytes_data = rMem.lindex("THY6526-1745891097-ed-1227p", 0)
    temp = json.loads(bytes_data)
    temp['time_diff'] = 15
    #print (temp)
    print(rMem.lset("THY6526-1745891097-ed-1227p", 0, json.dumps(temp)))
    




        

