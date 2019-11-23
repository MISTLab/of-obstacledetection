#!/usr/bin/env python3
"""
dronetello.py

DroNet implementation for the DJI/Ryze Tello

Written by Moritz Sperling
Based on the work of A. Loquercio et al., 2018 (https://github.com/uzh-rpg/rpg_public_dronet)
and D. Fuentes (https://github.com/damiafuentes/DJITelloPy)

Licensed under the MIT License (see LICENSE for details)
"""

import os
import sys
import cv2
import time
import math
import pygame
import numpy as np
from pygame.locals import *
from djitellopy import Tello
from keras import backend as k
from keras.models import model_from_json

sys.path.insert(0, '../workflow/util/')
from img_utils import pred_as_bar, pred_as_indicator


#
#        8888888          d8b 888    d8b          888 d8b                   888    d8b
#          888            Y8P 888    Y8P          888 Y8P                   888    Y8P
#          888                888                 888                       888
#          888   88888b.  888 888888 888  8888b.  888 888 .d8888b   8888b.  888888 888  .d88b.  88888b.
#          888   888 "88b 888 888    888     "88b 888 888 88K          "88b 888    888 d88""88b 888 "88b
#          888   888  888 888 888    888 .d888888 888 888 "Y8888b. .d888888 888    888 888  888 888  888
#          888   888  888 888 Y88b.  888 888  888 888 888      X88 888  888 Y88b.  888 Y88..88P 888  888
#        8888888 888  888 888  "Y888 888 "Y888888 888 888  88888P' "Y888888  "Y888 888  "Y88P"  888  888
#

class FrontEnd(object):
    """ Maintains the Tello display and moves it through the keyboard keys.
        Press escape key to quit.
        The controls are:
            - Tab: Takeoff
            - Shift: Land
            - Space: Emergency shutdown
            - Backspace: Shutdown
            - WASW: Forward, backward, left and right.
            - Q and E: Counter clockwise and clockwise rotations
            - R and F: Up and down.
            - P: Switch through controllable parameters
            - + and -: Raise or lower parameter
            - #: Enable / Disable DroNet
            - C: Toggle recording of frames
    """

    def __init__(self):

        # FlowDroNet Configuration
        self.json_model_path = "./models/DroNet/model_struct.json"
        self.weights_path = "./models/DroNet/model_weights_new_best.h5"
        self.output_folder = './recorded/'
        self.target_size = (200, 200)
        self.FPS = 10

        # config of controllable parameters
        # initial values
        self.controll_params = {
            'speed': 100,
            'alpha': 0.7,
            'beta': 0.5,
        }
        # stepsize
        self.controll_params_d = {
            'speed': 10,
            'alpha': 0.1,
            'beta': 0.1,
        }
        # max (min is 0)
        self.controll_params_m = {
            'speed': 100,
            'alpha': 1,
            'beta': 1,
        }

        # Init internal variables
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.internalSpeed = 100
        self.send_rc_control = False
        self.battery_percentage = 0
        self.v_old = 0
        self.sa_old = 0
        self.wasDroNet = True
        self.last_pred_col = 0
        self.last_pred_ang = 0
        self.lastTime = time.time()
        self.should_stop = False
        self.record_frames = False
        self.isArmed = False
        self.current_parameter = 0
        self.param_keys = list(self.controll_params.keys())

        # set keras to test phase
        k.set_learning_phase(0)

        # Tensorflow: load json and weights, then compile model
        with open(self.json_model_path, 'r') as json_file:
            loaded_model_json = json_file.read()
        self.net = model_from_json(loaded_model_json)
        self.net.load_weights(self.weights_path)
        self.net.compile(loss='mse', optimizer='sgd')

        # Create output folder if not exists
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # Init pygame
        pygame.init()
        pygame.display.set_caption("DroNeTello")
        self.screen = pygame.display.set_mode([985, 754])
        pygame.time.set_timer(USEREVENT + 1, int(1. / self.FPS * 1000))

        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

    #
    #      888b     d888          d8b               888
    #      8888b   d8888          Y8P               888
    #      88888b.d88888                            888
    #      888Y88888P888  8888b.  888 88888b.       888      .d88b.   .d88b.  88888b.
    #      888 Y888P 888     "88b 888 888 "88b      888     d88""88b d88""88b 888 "88b
    #      888  Y8P  888 .d888888 888 888  888      888     888  888 888  888 888  888
    #      888   "   888 888  888 888 888  888      888     Y88..88P Y88..88P 888 d88P
    #      888       888 "Y888888 888 888  888      88888888 "Y88P"   "Y88P"  88888P"
    #                                                                         888
    #                                                                         888
    #                                                                         888
    #

    def run(self):

        if not self.tello.connect():
            print("Tello not connected")
            return

        if not self.tello.set_speed(self.internalSpeed):
            print("Not set speed to lowest possible")
            return

        # In case streaming is on. This happens when we quit this program without the escape key.
        if not self.tello.streamoff():
            print("Could not stop video stream")
            return

        if not self.tello.streamon():
            print("Could not start video stream")
            return

        frame_read = self.tello.get_frame_read()

        self.should_stop = False
        while not self.should_stop:
            # sometimes read battery status
            if np.random.random() < 0.05:
                self.battery_percentage = self.tello.get_battery()

            # read frame
            img = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)

            # get output from dronet
            self.get_dronet_command(img)

            # update hud
            self.update_hud(img.copy())

            # save frame if recording
            if self.record_frames:
                fname = os.path.join(self.output_folder, time.strftime("record_%Y%m%d_%H%M%S_0.jpg", time.gmtime()))
                i = 0
                while os.path.exists(fname):
                    fname = fname.replace('_' + str(i) + '.', '_' + str(i + 1) + '.',)
                    i = i + 1
                cv2.imwrite(fname, frame_read.frame)

            # handle input from dronet or user
            for event in pygame.event.get():
                if event.type == USEREVENT + 1:
                    self.send_input()
                elif event.type == QUIT:
                    self.should_stop = True
                elif event.type == KEYDOWN:
                    if (event.key == K_ESCAPE) or (event.key == K_BACKSPACE):
                        self.should_stop = True
                    else:
                        self.keydown(event.key)
                elif event.type == KEYUP:
                    self.keyup(event.key)

                # shutdown stream
                if frame_read.stopped:
                    frame_read.stop()
                    break

            # wait a little
            time.sleep(1 / self.FPS)

        # always call before finishing to deallocate resources
        self.tello.end()
        time.sleep(1)
        exit(0)

    def get_dronet_command(self, img):
        # prep image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, self.target_size)
        img = np.asarray(img, dtype=np.float32) * np.float32(1.0 / 255.0)
        carry = np.array(img)[np.newaxis, :, :, np.newaxis]

        # inference
        outs = self.net.predict(carry, batch_size=None, verbose=0, steps=None)
        theta, p_t = outs[0][0], outs[1][0]
        self.last_pred_col = p_t
        self.last_pred_ang = theta

        # calculate real velocity and steering angle (see reference implementation)
        velocity = (1 - self.controll_params['alpha']) * self.v_old \
                   + self.controll_params['alpha'] * (1 - p_t) * self.controll_params['speed']
        steering_angle = (1 - self.controll_params['beta']) * self.sa_old \
                         + self.controll_params['beta'] * math.pi / 2 * theta
        sa_deg = -steering_angle / math.pi * 180

        # save current velocity and steering angle for next step
        self.v_old = velocity
        self.sa_old = steering_angle

        # set forward and yaw velocity if dronet is active
        if self.isArmed:
            self.for_back_velocity = int(velocity)
            self.yaw_velocity = int(sa_deg)

    #
    #      8888888                            888         888b     d888          888    888                    888
    #        888                              888         8888b   d8888          888    888                    888
    #        888                              888         88888b.d88888          888    888                    888
    #        888   88888b.  88888b.  888  888 888888      888Y88888P888  .d88b.  888888 88888b.   .d88b.   .d88888 .d8888b
    #        888   888 "88b 888 "88b 888  888 888         888 Y888P 888 d8P  Y8b 888    888 "88b d88""88b d88" 888 88K
    #        888   888  888 888  888 888  888 888         888  Y8P  888 88888888 888    888  888 888  888 888  888 "Y8888b.
    #        888   888  888 888 d88P Y88b 888 Y88b.       888   "   888 Y8b.     Y88b.  888  888 Y88..88P Y88b 888      X88
    #      8888888 888  888 88888P"   "Y88888  "Y888      888       888  "Y8888   "Y888 888  888  "Y88P"   "Y88888  88888P'
    #                       888
    #                       888
    #                       888
    #

    def keydown(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key
        """
        if key == pygame.K_w:  # set forward velocity
            self.isArmed = False
            self.for_back_velocity = self.controll_params['speed']
        elif key == pygame.K_s:  # set backward velocity
            self.isArmed = False
            self.for_back_velocity = -self.controll_params['speed']
        elif key == pygame.K_a:  # set left velocity
            self.isArmed = False
            self.left_right_velocity = -self.controll_params['speed']
        elif key == pygame.K_d:  # set right velocity
            self.isArmed = False
            self.left_right_velocity = self.controll_params['speed']
        elif key == pygame.K_r:  # set up velocity
            self.isArmed = False
            self.up_down_velocity = self.controll_params['speed']
        elif key == pygame.K_f:  # set down velocity
            self.isArmed = False
            self.up_down_velocity = -self.controll_params['speed']
        elif key == pygame.K_e:  # set yaw clockwise velocity
            self.isArmed = False
            self.yaw_velocity = self.controll_params['speed']
        elif key == pygame.K_q:  # set yaw counter clockwise velocity
            self.isArmed = False
            self.yaw_velocity = -self.controll_params['speed']
        elif key == pygame.K_TAB:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_LSHIFT:  # land
            self.isArmed = False
            self.tello.land()
            self.send_rc_control = False
        elif key == pygame.K_SPACE:  # emergency shutdown
            self.isArmed = False
            self.tello.emergency()
            self.send_rc_control = False
            self.should_stop = True
        elif key == pygame.K_HASH:  # arm/disarm dronet
            self.isArmed = not self.isArmed
            self.for_back_velocity = 0
            self.yaw_velocity = 0
        elif key == pygame.K_p:  # switch through parameters
            if self.current_parameter < len(self.controll_params) - 1:
                self.current_parameter = self.current_parameter + 1
            else:
                self.current_parameter = 0
        elif key == pygame.K_PLUS:  # raise current parameter
            what = self.param_keys[self.current_parameter]
            if self.controll_params[what] < self.controll_params_m[what] - 0.01:
                self.controll_params[what] = self.controll_params[what] + self.controll_params_d[what]
        elif key == pygame.K_MINUS:  # lower current parameter
            what = self.param_keys[self.current_parameter]
            if self.controll_params[what] > 0.01:
                self.controll_params[what] = self.controll_params[what] - self.controll_params_d[what]
        elif key == pygame.K_c:      # toggle recording of frames
            self.record_frames = not self.record_frames

    def keyup(self, key):
        """ Update velocities based on key released
        Arguments:
            key: pygame key
        """
        if key == pygame.K_w or key == pygame.K_s:  # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_r or key == pygame.K_f:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_q or key == pygame.K_e:  # set zero yaw velocity
            self.yaw_velocity = 0

    def send_input(self):
        """ Update routine. Send velocities to Tello."""
        print("V: " + str(self.for_back_velocity) + "; Y: " + str(self.yaw_velocity))
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
                                       self.yaw_velocity)

    #
    #        888    888          888                                888b     d888          888    888                    888
    #        888    888          888                                8888b   d8888          888    888                    888
    #        888    888          888                                88888b.d88888          888    888                    888
    #        8888888888  .d88b.  888 88888b.   .d88b.  888d888      888Y88888P888  .d88b.  888888 88888b.   .d88b.   .d88888 .d8888b
    #        888    888 d8P  Y8b 888 888 "88b d8P  Y8b 888P"        888 Y888P 888 d8P  Y8b 888    888 "88b d88""88b d88" 888 88K
    #        888    888 88888888 888 888  888 88888888 888          888  Y8P  888 88888888 888    888  888 888  888 888  888 "Y8888b.
    #        888    888 Y8b.     888 888 d88P Y8b.     888          888   "   888 Y8b.     Y88b.  888  888 Y88..88P Y88b 888      X88
    #        888    888  "Y8888  888 88888P"   "Y8888  888          888       888  "Y8888   "Y888 888  888  "Y88P"   "Y88888  88888P'
    #                                888
    #                                888
    #                                888
    #

    def update_hud(self, frame):

        """Draw drone info and record on frame"""
        if self.isArmed:
            stats = ["DroNet active."]
            if self.wasDroNet:
                stats.append("Predictions:")
                stats.append("[C: {:4.3f}] [SA: {:4.3f}]".format(float(self.last_pred_col), float(self.last_pred_ang)))
                stats.append("Commands:")
                stats.append("[V: {:03d}] [SA: {:03d}]".format(int(self.v_old), int(self.sa_old / math.pi * 180)))
            else:
                stats.append("Command overwritten ...")
        else:
            stats = ["DroNet disarmed.", "Predictions:",
                     "[C: {:4.3f}] [SA: {:4.3f}]".format(float(self.last_pred_col), float(self.last_pred_ang))]

        stats.append(self.param_keys[self.current_parameter]
                     + ": {:4.1f}".format(self.controll_params[self.param_keys[self.current_parameter]]))
        for idx, stat in enumerate(stats):
            text = stat.lstrip()
            cv2.putText(frame, text, (0, 30 + (idx * 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 0, 0), lineType=30)

        # show blinking red dot when recording
        if self.record_frames:
            cv2.putText(frame, "Recording", (frame.shape[1] - 187, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), lineType=30)
            if round(time.time()) % 2 == 0:
                cv2.circle(frame, (frame.shape[1] - 30,  30), 15, (255, 0, 0), -1)

        # show battery percentage
        cv2.putText(frame, "Battery: {:d} %".format(int(self.battery_percentage)),
                    (frame.shape[1] - 187, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), lineType=30)

        # display steering angle prediction as indicator in the middle
        hspacer = np.ones((12, frame.shape[1], 3), dtype=np.uint8) * 255
        steerbar = pred_as_indicator(self.last_pred_ang, (20, frame.shape[1]), "Steering")
        steerbar = cv2.cvtColor(steerbar, cv2.COLOR_BGR2RGB)
        frame_out = np.vstack((frame, hspacer, steerbar))

        # display collision prediction as bar on the side
        vspacer = np.zeros((frame_out.shape[0], 5, 3), dtype=np.uint8)
        collbar = pred_as_bar(self.last_pred_col, (frame_out.shape[0], 20), "Collision")
        collbar = cv2.cvtColor(collbar, cv2.COLOR_BGR2RGB)
        frame_out = np.hstack((frame_out, vspacer, collbar))

        frame_out = np.fliplr(frame_out)
        frame_out = np.rot90(frame_out)
        frame_out = pygame.surfarray.make_surface(frame_out)
        self.screen.fill([0, 0, 0])
        self.screen.blit(frame_out, (0, 0))
        pygame.display.update()


def main():
    frontend = FrontEnd()

    # run frontend
    frontend.run()


if __name__ == '__main__':
    main()
