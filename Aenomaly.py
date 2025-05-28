from AeroPredictor import move_to_predict
import json, time, sys


def anomalous_parse_json( json_string):
   #try:
      # parse all data into dictionary decoded:
         
      # Send data to redis first 
      
      move_to_predict(json_string)

      return 0
#    except (ValueError, KeyError, TypeError):
#        print("JSON format error: ", sys.exc_info()[0])
#        #print(str)
#        #print(traceback.format_exc())
#        return -1
   
#def anomaly_fly_high(json_string):
    
    

base_flight = { 
            "id": "Ranomaly_123",
            "ident": "Ranomaly",
            "pitr": str(int(time.time())), 
            "lat": "40.50000",
            "lon": "48.00000",
            "alt": "35000",
            "vertRate": "0",
            "gs": "400",
            "heading": "270",
            "altChange_encoded": 0,
}

anomalous_parse_json(base_flight)