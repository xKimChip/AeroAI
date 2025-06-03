from AeroPredictor import move_to_predict
import json, time, sys, math, argparse


def anomalous_parse_json( json_string):
   try:
      # parse all data into dictionary decoded:
         
      # Send data to redis first 
      
      move_to_predict(json_string)

      return 0
   except (ValueError, KeyError, TypeError):
       print("JSON format error: ", sys.exc_info()[0])
       #print(str)
       #print(traceback.format_exc())
       return -1
def calculate_dist_since_idle(json_string):
   R = 3958.756 # Radius of the Earth in miles
   
   lat = math.radians(float(json_string["lat"]))
   lon = math.radians(float(json_string["lon"]))
   heading = math.radians(float(json_string["heading"]))
   
   dist_traveled = int(json_string["gs"]) * ((int(time.time()) - int(json_string["pitr"])) / 3600)
   angular_distance = dist_traveled / R
   
   new_lat = math.asin(
      math.sin(lat) * math.cos(angular_distance) +
      math.cos(lat) * math.sin(angular_distance) * math.cos(heading)
   )
   
   new_lon = lon + math.atan2(
      math.sin(heading) * math.sin(angular_distance) * math.cos(lat),
      math.cos(angular_distance) - math.sin(lat) * math.sin(new_lat)
   )
   
   return math.degrees(new_lat), math.degrees(new_lon)
   
   
def calculate_translation(json_string):
   R = 3958.756 # Radius of the Earth in miles
   
   lat = math.radians(float(json_string["lat"]))
   lon = math.radians(float(json_string["lon"]))
   heading = math.radians(float(json_string["heading"]))
   
   dist_traveled = int(json_string["gs"]) * (20 / 3600)
   angular_distance = dist_traveled / R
   
   new_lat = math.asin(
      math.sin(lat) * math.cos(angular_distance) +
      math.cos(lat) * math.sin(angular_distance) * math.cos(heading)
   )
   
   new_lon = lon + math.atan2(
      math.sin(heading) * math.sin(angular_distance) * math.cos(lat),
      math.cos(angular_distance) - math.sin(lat) * math.sin(new_lat)
   )
   
   return math.degrees(new_lat), math.degrees(new_lon)
   
def anomaly_fly_up(json_string, height = 6000):

   for i in range(0,height,height//4):
      json_string["alt"] = str(int(json_string["alt"]) + height//4)
      json_string["vertRate"] = str(6 *(int(json_string["vertRate"]) // 3))
      json_string["pitr"] = str(int(json_string["pitr"]) + 10)
      json_string["lat"], json_string["lon"] = calculate_translation(json_string)
      anomalous_parse_json(json_string)
      time.sleep(2)
    
   json_string["vertRate"] = "0"
   json_string["pitr"] = str(int(json_string["pitr"]) + 10)
   json_string["lat"], json_string["lon"] = calculate_translation(json_string)
   anomalous_parse_json(json_string)

def anomaly_fly_down(json_string, height = 6000):
   anomaly_fly_up(json_string, height = -height)

def circle_fly_cw(json_string,degrees=360,delay=10):
   for i in range(0, degrees, degrees//6):
      new_head = int(json_string["heading"]) + degrees//6
      if new_head < 0:
         new_head += 360
      elif new_head >= 360:
         new_head -= 360
      json_string["heading"] = new_head
      json_string["pitr"] = str(int(json_string["pitr"]) + delay)
      json_string["lat"], json_string["lon"] = calculate_translation(json_string)
      anomalous_parse_json(json_string)
      time.sleep(2)
   
   json_string["pitr"] = str(int(json_string["pitr"]) + delay)
   json_string["lat"], json_string["lon"] = calculate_translation(json_string)
   anomalous_parse_json(json_string)
   time.sleep(2)

def circle_fly_ccw(json_string,degrees=360):
   circle_fly_cw(json_string, degrees=-degrees)

def fly_straight(json_string, gs=400, delay=10, slow=True):
   for i in range(0, 6):
      json_string["gs"] = str(gs)
      json_string["pitr"] = str(int(json_string["pitr"]) + 10)
      json_string["lat"], json_string["lon"] = calculate_translation(json_string)
      anomalous_parse_json(json_string)
      time.sleep(2)
   if slow:
      json_string["gs"] = "400"
   json_string["pitr"] = str(int(json_string["pitr"]) + delay)
   json_string["lat"], json_string["lon"] = calculate_translation(json_string)
   anomalous_parse_json(json_string)
   time.sleep(2)
   
def RTX(json_string):
   # makes an X
   new_head = int(json_string["heading"]) + 45
   if new_head < 0:
      new_head += 360
   elif new_head >= 360:
      new_head -= 360
   json_string["heading"] = str(new_head)
   for i in range(0, 2):
      json_string["pitr"] = str(int(json_string["pitr"]) + 10)
      json_string["lat"], json_string["lon"] = calculate_translation(json_string)
      anomalous_parse_json(json_string)
      time.sleep(2)
   new_head = int(json_string["heading"]) + 180
   if new_head < 0:
      new_head += 360
   elif new_head >= 360:
      new_head -= 360
   json_string["heading"] = str(new_head)
   for i in range(0, 1):
      json_string["pitr"] = str(int(json_string["pitr"]) + 10)
      json_string["lat"], json_string["lon"] = calculate_translation(json_string)
      anomalous_parse_json(json_string)
      time.sleep(2)
   new_head = int(json_string["heading"]) - 90
   if new_head < 0:
      new_head += 360
   elif new_head >= 360:
      new_head -= 360
   json_string["heading"] = str(new_head)
   for i in range(0, 1):
      json_string["pitr"] = str(int(json_string["pitr"]) + 10)
      json_string["lat"], json_string["lon"] = calculate_translation(json_string)
      anomalous_parse_json(json_string)
      time.sleep(2)
   new_head = int(json_string["heading"]) + 180
   if new_head < 0:
      new_head += 360
   elif new_head >= 360:
      new_head -= 360
   json_string["heading"] = str(new_head)
   for i in range(0, 2):
      json_string["pitr"] = str(int(json_string["pitr"]) + 10)
      json_string["lat"], json_string["lon"] = calculate_translation(json_string)
      anomalous_parse_json(json_string)
      time.sleep(2)
   new_head = int(json_string["heading"]) + 180
   if new_head < 0:
      new_head += 360
   elif new_head >= 360:
      new_head -= 360
   json_string["heading"] = str(new_head)
   for i in range(0, 1):
      json_string["pitr"] = str(int(json_string["pitr"]) + 10)
      json_string["lat"], json_string["lon"] = calculate_translation(json_string)
      anomalous_parse_json(json_string)
      time.sleep(2)
   new_head = int(json_string["heading"]) - 90
   if new_head < 0:
      new_head += 360
   elif new_head >= 360:
      new_head -= 360
   json_string["heading"] = str(new_head)
   for i in range(0, 1):
      json_string["pitr"] = str(int(json_string["pitr"]) + 10)
      json_string["lat"], json_string["lon"] = calculate_translation(json_string)
      anomalous_parse_json(json_string)
      time.sleep(2)
   # makes a T
   new_head = int(json_string["heading"]) - 45
   if new_head < 0:
      new_head += 360
   elif new_head >= 360:
      new_head -= 360
   json_string["heading"] = str(new_head)
   for i in range(0, 2):
      json_string["pitr"] = str(int(json_string["pitr"]) + 10)
      json_string["lat"], json_string["lon"] = calculate_translation(json_string)
      anomalous_parse_json(json_string)
      time.sleep(2)
   new_head = int(json_string["heading"]) - 180
   if new_head < 0:
      new_head += 360
   elif new_head >= 360:
      new_head -= 360
   json_string["heading"] = str(new_head)
   for i in range(0, 1):
      json_string["pitr"] = str(int(json_string["pitr"]) + 10)
      json_string["lat"], json_string["lon"] = calculate_translation(json_string)
      anomalous_parse_json(json_string)
      time.sleep(2)
   new_head = int(json_string["heading"]) + 90
   if new_head < 0:
      new_head += 360
   elif new_head >= 360:
      new_head -= 360
   json_string["heading"] = str(new_head)
   json_string['gs'] = str(int(json_string['gs']) // 2)
   for i in range(0, 2):
      json_string["pitr"] = str(int(json_string["pitr"]) + 10)
      #json_string['gs'] = str(int(json_string['gs']) // 2)
      json_string["lat"], json_string["lon"] = calculate_translation(json_string)
      anomalous_parse_json(json_string)
      time.sleep(2)
   for i in range(0, 1):
      json_string["pitr"] = str(int(json_string["pitr"]) + 10)
      json_string["lat"], json_string["lon"] = calculate_translation(json_string)
      anomalous_parse_json(json_string)
      time.sleep(2)
   json_string['gs'] = str(int(json_string['gs']) * 2)
   new_head = int(json_string["heading"]) + 90
   if new_head < 0:
      new_head += 360
   elif new_head >= 360:
      new_head -= 360
   json_string["heading"] = str(new_head)
   for i in range(0, 1):
      json_string["pitr"] = str(int(json_string["pitr"]) + 10)
      json_string["lat"], json_string["lon"] = calculate_translation(json_string)
      anomalous_parse_json(json_string)
      time.sleep(2)
   # makes an R
   new_head = int(json_string["heading"]) + 90
   if new_head < 0:
      new_head += 360
   elif new_head >= 360:
      new_head -= 360
   json_string['gs'] = str(int(json_string['gs']) // 2)
   json_string["heading"] = str(new_head)
   json_string["pitr"] = str(int(json_string["pitr"]) + 10)
   json_string["lat"], json_string["lon"] = calculate_translation(json_string)
   anomalous_parse_json(json_string)

   time.sleep(2)
   new_head = int(json_string["heading"]) - 45
   if new_head < 0:
      new_head += 360
   elif new_head >= 360:
      new_head -= 360
   json_string["heading"] = str(new_head)
   json_string["pitr"] = str(int(json_string["pitr"]) + 10)
   json_string["lat"], json_string["lon"] = calculate_translation(json_string)
   anomalous_parse_json(json_string)
   time.sleep(2)
   new_head = int(json_string["heading"]) - 45
   if new_head < 0:
      new_head += 360
   elif new_head >= 360:
      new_head -= 360
   json_string['gs'] = str(int(json_string['gs']) * 2)
   json_string["heading"] = str(new_head)
   json_string["pitr"] = str(int(json_string["pitr"]) + 10)
   json_string["lat"], json_string["lon"] = calculate_translation(json_string)
   anomalous_parse_json(json_string)
   time.sleep(2)
   new_head = int(json_string["heading"]) + 180
   if new_head < 0:
      new_head += 360
   elif new_head >= 360:
      new_head -= 360
   json_string["heading"] = str(new_head)
   json_string["pitr"] = str(int(json_string["pitr"]) + 10)
   json_string["lat"], json_string["lon"] = calculate_translation(json_string)
   anomalous_parse_json(json_string)
   time.sleep(2)
   new_head = int(json_string["heading"]) -45
   if new_head < 0:
      new_head += 360
   elif new_head >= 360:
      new_head -= 360
   json_string['gs'] = str(int(json_string['gs']) // 2)
   json_string["heading"] = str(new_head)
   json_string["pitr"] = str(int(json_string["pitr"]) + 5)
   json_string["lat"], json_string["lon"] = calculate_translation(json_string)
   anomalous_parse_json(json_string)
   time.sleep(2)
   new_head = int(json_string["heading"]) - 90
   if new_head < 0:
      new_head += 360
   elif new_head >= 360:
      new_head -= 360
   json_string["heading"] = str(new_head)
   json_string["pitr"] = str(int(json_string["pitr"]) + 5)
   json_string["lat"], json_string["lon"] = calculate_translation(json_string)
   anomalous_parse_json(json_string)
   time.sleep(2)
   new_head = int(json_string["heading"]) -45
   if new_head < 0:
      new_head += 360
   elif new_head >= 360:
      new_head -= 360
   json_string["heading"] = str(new_head)
   json_string['gs'] = str(int(json_string['gs']) * 2)
   json_string["pitr"] = str(int(json_string["pitr"]) + 10)
   json_string["lat"], json_string["lon"] = calculate_translation(json_string)
   anomalous_parse_json(json_string)
   time.sleep(2)
   new_head = int(json_string["heading"]) - 90
   if new_head < 0:
      new_head += 360
   elif new_head >= 360:
      new_head -= 360
   json_string["heading"] = str(new_head)
   json_string['gs'] = str(int(json_string['gs']) // 2)
   for i in range(0, 3):
      json_string["pitr"] = str(int(json_string["pitr"]) + 10)
      json_string["lat"], json_string["lon"] = calculate_translation(json_string)
      anomalous_parse_json(json_string)
      time.sleep(2)
   
   new_head = int(json_string["heading"]) + 90
   if new_head < 0:
      new_head += 360
   elif new_head >= 360:
      new_head -= 360
   json_string["heading"] = str(new_head)
   json_string['gs'] = str(int(json_string['gs']) * 2)
   for i in range(0, 1):
      json_string["pitr"] = str(int(json_string["pitr"]) + 10)
      json_string["lat"], json_string["lon"] = calculate_translation(json_string)
      anomalous_parse_json(json_string)
      time.sleep(2)
   


def create_base_flight(ident="Ranomaly"):
   return {
      "id": f"{ident}_id",
      "ident": ident,
      "pitr": str(int(time.time())),
      "lat": "40.50000",
      "lon": "48.00000",
      "alt": "40000",
      "vertRate": "0",
      "gs": "400",
      "heading": "270",
      "altChange_encoded": 0,
   }
   
def get_user_choice():
    """Get and validate user input"""
    while True:
        try:
            choice = input("\nEnter your choice (1-9, 0 to quit, or 'help'): ").strip().lower()
            return choice
        except KeyboardInterrupt:
            print("\n\nExiting simulation...")
            return "0"
        except EOFError:
            return "0"

def show_menu():
    """Display available simulation options"""
    print("\n" + "="*50)
    print("FLIGHT SIMULATION MENU")
    print("="*50)
    print("1. cw       - Fly clockwise circle")
    print("2. ccw      - Fly counter-clockwise circle")
    print("3. up       - Climb altitude")
    print("4. down     - Descend altitude")
    print("5. speed    - Change speed of flight (400 knots by default speed)")
    print("6. noslow   - Change speed of flight without slow down (400 knots by default speed)")

    
    print("9. help     - Show this menu")
    print("0. quit     - Exit simulation")
    print("="*50)


def run_interactive_simulation(base_flight):
   """Run interactive simulation with user input"""
   print(f"\nStarting interactive flight simulation with ident: {base_flight['ident']}")
   
   # Initialize flight position
   anomalous_parse_json(base_flight)
   
   while True:
      choice = get_user_choice()
      if (base_flight["pitr"]) < str(int(time.time())):
         calculate_dist_since_idle(base_flight)
         print("Flight catching up to real time...")
      
      if choice in ['0', 'quit', 'exit', 'q']:
         print("Ending simulation. Safe travels!")
         break
      
      elif choice in ['1', 'cw']:
         degree = input("\nEnter degree change (default = 360): ").strip().lower()
         print("Executing clockwise circle...")
         circle_fly_cw(base_flight, degrees=int(degree))
         print("Clockwise circle completed.")
      
      elif choice in ['2', 'ccw']:
         degree = input("\nEnter degree change (default = 360): ").strip().lower()
         print("Executing counter-clockwise circle...")
         circle_fly_ccw(base_flight, degrees=int(degree))
         print("Counter-clockwise circle completed.")
      
      elif choice in ['3', 'up']:
         altitude = input("\nEnter altitude change: ").strip().lower()
         print("Climbing altitude...")
         anomaly_fly_up(base_flight, height=int(altitude))
         print("Climb completed.")
      
      elif choice in ['4', 'down']:
         altitude = input("\nEnter altitude change: ").strip().lower()
         print("Descending altitude...")
         anomaly_fly_down(base_flight, height=int(altitude))
         print("Descent completed.")
      
      elif choice in ['5', 'speed']:
         speed = input("\nEnter speed: ").strip().lower()
         if int(speed) > 400:
            print("Flying straight at high speed...")
            fly_straight(base_flight, gs=int(speed))
            print("High speed flight completed.")
         elif int(speed) < 400 and int(speed) > 0:
            print("Flying straight at low speed...")
            fly_straight(base_flight, gs=int(speed))
            print("Low speed flight completed.")
         elif int(speed) == 400:
            print("Flying straight at normal speed...")
            fly_straight(base_flight)
            print("Straight flight completed.")
         else:
            print(f"Invalid speed: {speed}. Please enter a positive integer.")
      elif choice in ['6', 'noslow']:
         speed = input("\nEnter speed: ").strip().lower()
         if int(speed) > 400:
            print("Flying straight at high speed...")
            fly_straight(base_flight, gs=int(speed),slow=False)
            print("High speed flight completed.")
         elif int(speed) < 400 and int(speed) > 0:
            print("Flying straight at low speed...")
            fly_straight(base_flight, gs=int(speed),slow=False)
            print("Low speed flight completed.")
         elif int(speed) == 400:
            print("Flying straight at normal speed...")
            fly_straight(base_flight,slow=False)
            print("Straight flight completed.")
         else:
            print(f"Invalid speed: {speed}. Please enter a positive integer.")
            
      elif choice in ['rtx']:
         print("Thank you Sponsors...")
         RTX(base_flight)
         print("Thank you completed.")
      elif choice in ['9', 'help', 'menu']:
         show_menu()
      else:
         print(f"Invalid choice: '{choice}'. Type 'help' to see available options.")
      
      # Brief pause between operations
      time.sleep(1) 
   

def main():
   # Set up argument parser
   parser = argparse.ArgumentParser(description='Flight Anomaly Simulator')
   parser.add_argument('--ident', '-i', 
                     type=str, 
                     default='Ranomaly',
                     help='Flight identifier (default: Ranomaly)')
   parser.add_argument('--callsign', '-c',
                     type=str,
                     help='Alternative to --ident (same functionality)')
   
   # Parse arguments
   args = parser.parse_args()
   
   # Use callsign if provided, otherwise use ident
   flight_ident = args.callsign if args.callsign else args.ident
   
   # Create flight with specified ident
   base_flight = create_base_flight(flight_ident)
   
   print(f"Starting flight simulation with ident: {flight_ident}")
   
   # Run the simulation
   run_interactive_simulation(base_flight)

if __name__ == "__main__":
    main()