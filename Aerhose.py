#!/usr/bin/env python

import json, socket, ssl, sys, time, zlib, math, signal, time, redis
import tkinter as tk
from AeroPredictor import move_to_predict, check_reetry
from datetime import datetime



username = "RTXDC"
apikey = ""
latitude = 37.7749
longitude = 46.53798
range = 300

compression = None                           # set to "deflate", "decompress", or "gzip" to enable compression
servername = "firehose.flightaware.com"
filename = "data/hose_data.json"             # file to save the data
TO_FILE = True                               # set to True to save the data to a file
LIVE = True                                  # set to True for live demo

SIGINT_FLAG = False
#rMem = redis.Redis(host='localhost', port=6379, db=0)



def sigkill_handler(signum, frame):
   global SIGINT_FLAG
   print(f"Received signal {signum}")
   SIGINT_FLAG = True
   return 0


def get_user_input():
   global username, apikey, latitude, longitude

   root = tk.Tk()
   root.geometry("300x500")
   root.title("AeroHose")
   
   username_var = tk.StringVar(root, value=username)
   apikey_var = tk.StringVar(root, value=apikey)
   latitude_var = tk.DoubleVar(root, value=latitude)
   longitude_var = tk.DoubleVar(root, value=longitude)
   range_var = tk.IntVar(root, value=range)
   count_var = tk.IntVar(root, value=0)
   endless_var = tk.BooleanVar(root, value=False)
   append_var = tk.BooleanVar(root, value=False)
   duration_var = tk.IntVar(root, value=0)
   
   
   values = {}
   
   def submit_values():
      # Get values from entry fields
      values['username'] = username_var.get()
      values['apikey'] = apikey_var.get()
      values['latitude'] = latitude_var.get()
      values['longitude']= longitude_var.get()
      values['range'] = range_var.get()
      values['count'] = count_var.get()
      values['endless'] = endless_var.get()
      values['append'] = append_var.get()
      values['duration'] = duration_var.get()

      # Print values for debugging
      #print(f"Username: {values['username']}")
      #print(f"API Key: {values['apikey']}")
      print(f"Latitude: {values['latitude']}")
      print(f"Longitude: {values['longitude']}")
      print(f"Range: {values['range']}")
      print(f"Count: {values['count']}")
      print(f"Endless: {values['endless']}")
      print(f"Append: {values['append']}")
      print(f"Time: {values['duration']}")
      
      root.destroy()  # Close the GUI after submission
   

   label = tk.Label(root, text="AeroHose Location Tool", font=("Helvetica", 10))
   label.pack(padx=20, pady=20)

   # Username field
   user_label = tk.Label(root, text="Username", font=("Helvetica", 8))
   user_label.pack(padx=10, pady=0)
   user = tk.Entry(root, width=20, textvariable=username_var)
   user.pack(padx=10, pady=5)

   # API Key field
   passkey_label = tk.Label(root, text="API Key", font=("Helvetica", 8))
   passkey_label.pack(padx=10, pady=0)
   passkey = tk.Entry(root, width=20, textvariable=apikey_var)
   passkey.pack(padx=10, pady=5)

   # Latitude field
   latitude_label = tk.Label(root, text="Latitude", font=("Helvetica", 8))
   latitude_label.pack(padx=10, pady=0)
   lat = tk.Entry(root, textvariable=latitude_var)
   lat.pack(padx=10, pady=5)

   # Longitude field
   longitude_label = tk.Label(root, text="Longitude", font=("Helvetica", 8))
   longitude_label.pack(padx=10, pady=0)
   lon = tk.Entry(root, textvariable=longitude_var)
   lon.pack(padx=10, pady=5)
   
   range_label = tk.Label(root, text="Range", font=("Helvetica", 8))
   range_label.pack(padx=10, pady=5)
   ran = tk.Entry(root, textvariable=range_var)
   ran.pack(padx=10, pady=5)
   
   # Label Frame
   label_frame = tk.Frame(root)
   label_frame.pack(padx=10, pady=5, fill=tk.X)

   
   count_label = tk.Label(label_frame, text="Count", font=("Helvetica", 8))
   count_label.pack(side=tk.RIGHT, padx=(0,100))
      
   time_label = tk.Label(label_frame, text ="Time (s)", font=("Helvetica", 8))
   time_label.pack(side=tk.RIGHT)


   
   count_frame = tk.Frame(root)
   count_frame.pack(padx=10, pady=5, fill=tk.X)

   # Endless checkbox
   endless_checkbox = tk.Checkbutton(count_frame, text="Endless", variable=endless_var)
   endless_checkbox.pack(side=tk.RIGHT)

   # Count Field
   count_entry = tk.Entry(count_frame, width=10, textvariable=count_var)
   count_entry.pack(side=tk.RIGHT, padx=0, pady=0)
      
   
   time_entry = tk.Entry(count_frame, width=10, textvariable=duration_var)
   time_entry.pack(side=tk.RIGHT, padx=0, pady=0)
   
   
   sub_app_frame = tk.Frame(root)
   sub_app_frame.pack(padx=10, pady=0, fill=tk.X)
   
   append_checkbox = tk.Checkbutton(sub_app_frame, text="Append", variable=append_var)
   append_checkbox.pack(side=tk.RIGHT)

   
   
   # Submit button
   submit_button = tk.Button(sub_app_frame, text="Submit", command=submit_values)
   submit_button.pack(side=tk.RIGHT, padx=0, pady=10)
   

   
   root.mainloop()
   
   return values['username'], values['apikey'], values['latitude'], values['longitude'], values['range'], values['count'], values['endless'], values['append'],values['duration']



class InflateStream:
   "A wrapper for a socket carrying compressed data that does streaming decompression"

   def __init__(self, sock, mode):
      self.sock = sock
      self._buf = bytearray()
      self._eof = False
      if mode == 'deflate':     # no header, raw deflate stream
         self._z = zlib.decompressobj(-zlib.MAX_WBITS)
      elif mode == 'compress':  # zlib header
         self._z = zlib.decompressobj(zlib.MAX_WBITS)
      elif mode == 'gzip':      # gzip header
         self._z = zlib.decompressobj(16 | zlib.MAX_WBITS)
      else:
         raise ValueError('unrecognized compression mode')

   def _fill(self):
      rawdata = self.sock.recv(8192)
      if len(rawdata) == 0:
         self._buf += self._z.flush()
         self._eof = True
      else:
         self._buf += self._z.decompress(rawdata)

   def readline(self):
      newline = self._buf.find(b'\n')
      while newline < 0 and not self._eof:
         self._fill()
         newline = self._buf.find(b'\n')

      if newline >= 0:
         rawline = self._buf[:newline+1]
         del self._buf[:newline+1]
         return rawline.decode('ascii')

      # EOF
      return ''

def haversine(lat1, lon1, lat2, lon2):
    
   #R = 6371.0  # Radius of Earth in kilometers
   R = 3958.756  # Radius of Earth in miles
   lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
   dlon = lon2 - lon1
   dlat = lat2 - lat1
   
   a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
   c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

   distance = R * c
   #print(f"Distance: {distance:.2f} miles")    debugging 
   return distance

# function to parse JSON data:
def parse_json( str , output, latitude, longitude, range, append):
   try:
      # parse all data into dictionary decoded:
      decoded = json.loads(str)

      # Only looking for positional updates, other types seen "arrival", "flinfo", "cancellation", "surface_offblock", "power_on", "departure", ""
      if decoded["type"] != "position":
         #print(f"Skipped type: {decoded["type"]}")
         return -1
      elif haversine(float(decoded["lat"]), float(decoded['lon']), latitude, longitude) > range:
         #print(f"Skipped position: {decoded['lat']}, {decoded['lon']}")
         return -1
      elif check_reetry(decoded['id']):
         print(f"Skipped item: {decoded['id']}")
         return -1
      elif not decoded.get('alt') or not decoded.get('gs') :
         #print(f"Skipped item: {decoded['id']}")
         return -1
         
      # Send data to redis first 
      move_to_predict(decoded)
      
      if append and output is not None:
         json_str = json.dumps(decoded, indent="\t\t")
         output.write('\t' + json_str.replace('}', '\t}'))
      elif not append and not LIVE:
         with open(filename, 'w') as output:
            json.dump([decoded], output, indent="\t")


      # compute the latency of this message:
      clocknow = time.time()
      diff = clocknow - int(decoded['pitr'])
      #print("diff = {0:.2f} s\n".format(diff))
      return 0
   except (ValueError, KeyError, TypeError):
       print("JSON format error: ", sys.exc_info()[0])
       #print(str)
       #print(traceback.format_exc())
       return -1

def initiate_hose():
   # get popup for user input
   username, apikey, latitude, longitude, range, count, endless, append, duration = get_user_input()
   global SIGINT_FLAG

   # Create socket
   sock = socket.socket(socket.AF_INET)
   # Create a SSL context with the recommended security settings for client sockets, including automatic certificate verification
   context = ssl.create_default_context()
   # the folowing line requires Python 3.7+ and OpenSSL 1.1.0g+ to specify minimum_version
   context.minimum_version = ssl.TLSVersion.TLSv1_2

   ssl_sock = context.wrap_socket(sock, server_hostname = servername)
   print("Connecting...")
   ssl_sock.connect((servername, 1501))
   print("Connection succeeded")


   # build the initiation command:
   initiation_command = "live username {} password {}".format(username, apikey)
   if compression is not None:
      initiation_command += " compression " + compression

   # send initialization command to server:
   initiation_command += "\n"
   if sys.version_info[0] >= 3:
      ssl_sock.write(bytes(initiation_command, 'UTF-8'))
   else:
      ssl_sock.write(initiation_command)

   # return a file object associated with the socket
   if compression is not None:
      file = InflateStream(sock = ssl_sock, mode = compression)
   else:
      file = ssl_sock.makefile('r')


   # Catch SIGKILL


   # use "while True" for no limit in messages received
   output = None
   
   if TO_FILE and append:
      output = open(filename, 'w')
      output.write("[\n")
      
   if signal.signal(signal.SIGINT, sigkill_handler) == 0:
      print("SIGINT received")
   if time == 0:
      end_time = time.time() * 2
   else:
      start_time = time.time()
      end_time = start_time + duration
      print(f"Start time: {datetime.fromtimestamp(start_time)}")
   
   while count > 0 or endless or time.time() < end_time:
      try :
         # read line from file:
         
         inline = file.readline()
         if inline == '':
            # EOF
            break

         # parse the line
         if parse_json(inline, output, latitude, longitude, range, append) == 0:
            if SIGINT_FLAG:
               print("SIGINT received, exiting...")
               break
            elif append:
               time_bool = time.time() < end_time
               if (endless or count > 1 or time_bool):
                  output.write(",\n")   
            if not endless and count != 0:
               count -= 1
         
      except socket.error as e:
         print('Connection fail', e)
         print(traceback.format_exc())
   if append:
      output.write('\n]')
      output.close()
   print(f"Approximate End_time: {datetime.fromtimestamp(end_time)}")
   print(f"Actual End_time: {datetime.fromtimestamp(time.time())}")
   
   # wait for user input to end
   # input("\n Press Enter to exit...");
   # close the SSLSocket, will also close the underlying socket
   ssl_sock.close()

initiate_hose()
