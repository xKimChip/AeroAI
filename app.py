# app.py
from gevent import monkey; monkey.patch_all()
from flask import Flask, render_template
from flask_socketio import SocketIO
import redis, json

app = Flask(__name__)
socketio = SocketIO(
    app,
    async_mode='gevent',
    cors_allowed_origins="*",
    serve_client=True    
)

rDisk = redis.Redis(host='localhost', port=6379, db=0)

@app.route("/")
def index():
    return render_template("index.html")

def redis_listener():
    p = rDisk.pubsub()
    p.subscribe('lpush_channel')
    for msg in p.listen():
        if msg['type']!='message': continue
        key = msg['data'].decode()
        raw = rDisk.lindex(key, 0)
        if not raw: continue
        data = json.loads(raw.decode())
        socketio.emit('new_data', data)

if __name__ == "__main__":
    socketio.start_background_task(redis_listener)
    socketio.run(app, host="0.0.0.0", port=5001)
