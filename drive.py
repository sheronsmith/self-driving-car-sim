"""
Talk to the Udacity simulator in autonomous mode.

    python drive.py model.keras

Opens a socket on port 4567. The sim sends a camera frame,
we preprocess it, run it through the model, and send back
the steering angle + throttle.
"""

import argparse, base64
from io import BytesIO

import eventlet
import eventlet.wsgi
import numpy as np
import socketio
from flask import Flask
from PIL import Image
from keras.models import load_model

from utils import preprocess

sio = socketio.Server()
app = Flask(__name__)

model = None
MAX_SPEED = 25
MIN_SPEED = 10


@sio.on('telemetry')
def telemetry(sid, data):
    if not data:
        sio.emit('manual', {})
        return

    speed = float(data['speed'])

    # decode the base64 image from the sim
    raw = Image.open(BytesIO(base64.b64decode(data['image'])))
    img = np.asarray(raw)
    img = preprocess(img)
    img = np.expand_dims(img, axis=0)

    steering = float(model.predict(img, verbose=0)[0][0])

    # basic speed controller — slow down on sharp turns
    if speed > MAX_SPEED:
        throttle = 0.0
    elif speed < MIN_SPEED:
        throttle = 0.5
    else:
        throttle = 0.3 - 0.006 * abs(steering) * speed

    print(f'steer {steering:+.4f}  throttle {throttle:.3f}  speed {speed:.1f}')
    sio.emit('steer', {
        'steering_angle': str(steering),
        'throttle':       str(throttle)
    })


@sio.on('connect')
def connect(sid, environ):
    print('simulator connected')
    sio.emit('steer', {'steering_angle': '0', 'throttle': '0'})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('model_path', help='path to .keras model')
    args = ap.parse_args()

    model = load_model(args.model_path)
    print(f'loaded {args.model_path}')

    wsgi_app = socketio.WSGIApp(sio, app)
    print('waiting for simulator on port 4567 ...')
    eventlet.wsgi.server(eventlet.listen(('', 4567)), wsgi_app)
