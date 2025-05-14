import pandas as pd
from AeroEncoder import load_model, flight_prediction, SaliencyAnalyzer
import json, redis
import numpy as np
model, scaler, detector = load_model()
features = ['alt', 'gs', 'heading', 'vertRate', 'altChange_encoded', 'gs_change_rate', 'heading_change_rate']

# add saliency to the model
model.add_saliency_analyzer(features)

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
EXPIRE_TIME_LONG = 18000 # 5 hours
REENTRY_TIME = 17980  # 4 hours 59 minutes 40 seconds





def check_reetry(key):
   if rMem.ttl(key) >= REENTRY_TIME:
      return True
   return False

def process_for_redis(data, eTime=EXPIRE_TIME):
   # use a list to get the recent history of a data point.
   if 'altChange' not in data:
      data['altChange'] = ' '
   
   data_vals = {
      'ident': data['ident'],
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

      #data_vals['anomaly_score'] = max(0, temp['anomaly_score'] - data_vals['time_diff'] // 150)                   # Degrade anomaly score by 0.2 every 30 seconds                    
   else:
      data_vals['time_diff'] = 1                            # if the list does not exist, set the first time_diff to 1
      data_vals['gs_change_rate'] = 0                       # if the list does not exist, set the first gs_change_rate to 0   
      data_vals['heading_change_rate'] = 0                  # if the list does not exist, set the first heading_change_rate to 0
      
   # Predict in offline use as well
   # rMem.lpush(key, json.dumps(data_vals)) # push the columns of data to the list
   # if rMem.ttl(key) == -1: # set the expirate time to ETIME
   #    rMem.expire(key, eTime)
   # else:
   #    rMem.expire(key, eTime, xx=True)
    
   # rMem.ltrim(key, 0, 10) # keep only the last 10 items
   
   return key, data_vals

def move_to_predict(data):
   # Process the data and send it to Redis in Memory
   key, data_vals = process_for_redis(data)
   
   df = pd.DataFrame(data_vals, index=[0])


   results_df = flight_prediction(df, model, scaler, detector, explain=True)
   data_vals['anomaly'] = results_df.to_json() # Uncomment to see the whole dataframe

   data_vals['anomaly_score'] = float(results_df['anomaly'].values[0]) # get the anomaly score from the results

   if data_vals['anomaly_score'] > 0:   # uncomment to see anomalies easier
      data_vals['saliency'] = results_df['saliency'].values[0] # get the saliency from the results
      print(key)
   
   rMem.lpush(key, json.dumps(data_vals)) # push the columns of data to the list
   if rMem.ttl(key) == -1: # set the expirate time to ETIME
      rMem.expire(key, EXPIRE_TIME_LONG)
   else:
      rMem.expire(key, EXPIRE_TIME_LONG, xx=True)
   rMem.publish('lpush_channel', key) # publish the key to the channel
   
   
   # For Cloud usage
#    rDisk.lpush(key, json.dumps(data_vals)) # push the columns of data to the list
#    if rDisk.ttl(key) == -1: # set the expirate time to ETIME
#       rDisk.expire(key, EXPIRE_TIME_LONG)
#    else:
#       rDisk.expire(key, EXPIRE_TIME_LONG, xx=True)
#    rDisk.publish('lpush_channel', key) # publish the key to the channel
    
# if __name__ == "__main__":
#    rDisk.publish('test_channel2', 'Hello, Redis!') # test the redis connection
    




        

