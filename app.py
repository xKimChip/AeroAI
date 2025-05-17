from flask import Flask, render_template
from flask_socketio import SocketIO
import redis
import threading
import json

# Flask app setup
app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

# Redis config
REDIS_HOST = 'redis-14815.c289.us-west-1-2.ec2.redns.redis-cloud.com'
REDIS_PORT = 14815
REDIS_USER = 'default'
REDIS_PASSWORD = 'J9rkVUXSCnwbadDNPaicB2YnFe4EZjxo'

# Redis client
rDisk = redis.Redis(
   host=REDIS_HOST, port=REDIS_PORT,
   username=REDIS_USER,
   password=REDIS_PASSWORD,
   socket_connect_timeout=5
   )
#rDisk = redis.Redis(host='localhost', port=6379, db=0) 

@app.route("/")
def index():
    return render_template("index.html")

def message_handler(message):
    if message["type"] != "message":
        return

    try:
        key = message["data"].decode()
        print(type(message["data"].decode()))
        raw_data = rDisk.lindex(key, 0)

        if raw_data is not None:
            parsed = json.loads(raw_data.decode())
            print(f"[INFO] Emitting data from key '{key}': {parsed}")
            socketio.emit('new_data', parsed)
        else:
            print(f"[WARN] No data at key: {key}")

    except Exception as e:
        print(f"[ERROR] Failed handling message: {e}")


def initiate_redis_sub():
    print('Initiating Redis Subscriber')
    
    pubsub = rDisk.pubsub()
    pubsub.subscribe('lpush_channel')

    print('Redis Subscriber Initiated')

    for message in pubsub.listen():
        message_handler(message)


def start_listener():
    listener_thread = threading.Thread(target=initiate_redis_sub)
    listener_thread.daemon = True
    listener_thread.start()

if __name__ == "__main__":
    start_listener()
    socketio.run(app, debug=True, port=5001)
