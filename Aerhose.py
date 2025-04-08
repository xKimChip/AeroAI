#!/usr/bin/env python

import json, socket, ssl, sys, time, zlib, math
from haversine import haversine, Unit
import tkinter as tk


username = ""
apikey = ""
latitude = ""
longitude = ""
range = ""

compression = None        # set to "deflate", "decompress", or "gzip" to enable compression
servername = "firehose.flightaware.com"
filename = "data/hose_data.json"            # file to save the data
TO_FILE = True                    # set to True to save the data to a file

def get_user_input():
   global username, apikey, latitude, longitude

   root = tk.Tk()
   root.geometry("200x400")
   root.title("AeroHose")
   
   username_var = tk.StringVar(root, value="RTXDC")
   apikey_var = tk.StringVar(root, value="ab2cbcfcbee5c5a3263d3caf2021eafe286f0caf")
   latitude_var = tk.StringVar(root, value="37.7749")
   longitude_var = tk.StringVar(root, value="46.53798")
   range_var = tk.StringVar(root, value="300")
   
   values = {}
   
   def submit_values():
      # Get values from entry fields
      values['username'] = username_var.get()
      values['apikey'] = apikey_var.get()
      values['latitude'] = latitude_var.get()
      values['longitude']= longitude_var.get()
      values['range'] = range_var.get()

      # Print values for debugging
      print(f"Username: {values['username']}")
      print(f"API Key: {values['apikey']}")
      print(f"Latitude: {values['latitude']}")
      print(f"Longitude: {values['longitude']}")
      print(f"Range: {values['range']}")
      
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


   # Submit button
   submit_button = tk.Button(root, text="Submit", command=submit_values,)
   submit_button.pack(pady=10)
   
   root.mainloop()
   
   return values['username'], values['apikey'], values['latitude'], values['longitude'], values['range']

   

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
def parse_json( str ):
   try:
       # parse all data into dictionary decoded:
       decoded = json.loads(str)
       
       # Only looking for positional updates, other types seen "arrival", "flinfo"
       if decoded["type"] != "position":
          print(f"Skipped type: {decoded[type]}")
          #return -1

       elif haversine(float(decoded["lat"]), float(decoded['lon']), float(latitude), float(longitude)) > float(range):
          print(f"Skipped position: {decoded['lat']}, {decoded['lon']}")
          #return -1
         
      
       #print(decoded)
       elif TO_FILE:
         with open(filename, 'w') as f:
            #print("writing to file")
            json.dump(decoded, f, indent=4)
       

       # compute the latency of this message:
       clocknow = time.time()
       diff = clocknow - int(decoded['pitr'])
       #print("diff = {0:.2f} s\n".format(diff))
   except (ValueError, KeyError, TypeError):
       print("JSON format error: ", sys.exc_info()[0])
       #print(str)
       #print(traceback.format_exc())
   return 0

# get popup for user input
username, apikey, latitude, longitude, range = get_user_input()

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




# use "while True" for no limit in messages received
count = 5
while count > 0:
   try :
      # read line from file:
      inline = file.readline()
      if inline == '':
         # EOF
         break

      # parse the line
      if parse_json(inline) == 0:
         count -= 1
   except socket.error as e:
      print('Connection fail', e)
      print(traceback.format_exc())

# wait for user input to end
# input("\n Press Enter to exit...");
# close the SSLSocket, will also close the underlying socket
ssl_sock.close()

