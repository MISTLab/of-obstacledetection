#!/usr/bin/env python3
import cv2
import math
import numpy as np
from keras import backend as k
from keras.models import model_from_json

# init variables
json_model_path = "./model/model_struct.json"
weights_path = "./model/model_weights.h5"
target_size = (200, 200)
alpha = 0.7
beta = 0.5
v_max = 10
v_old = 0
sa_old = 0

cap = cv2.VideoCapture(0)

# Set keras to test phase
k.set_learning_phase(0)

# Load json and weights, then compile model
with open(json_model_path, 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights(weights_path)
model.compile(loss='mse', optimizer='sgd')

while True:
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, target_size)
    img = np.asarray(img, dtype=np.float32) * np.float32(1.0/255.0)

    carry = np.array(img)[np.newaxis, :, :, np.newaxis]

    outs = model.predict(carry, batch_size=None, verbose=0, steps=None)
    theta, p_t = outs[0][0], outs[1][0]

    velocity = (1 - alpha) * v_old + alpha * (1 - p_t) * v_max
    steering_angle = (1 - beta) * sa_old + beta * math.pi / 2 * theta
    sa_deg = steering_angle / math.pi * 180

    v_old = velocity
    sa_old = steering_angle

    out = {'Velocity\t ==> \t': velocity, 'Steering Angle\t ==> \t': sa_deg}
    for name, val in out.items():
        # prnt = "{0} ==> {1:3.4f}"
        # print(print.format(name, float(val)))
        print(name + "%3.4f" % float(val))

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
