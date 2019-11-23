#!/usr/bin/env python3
"""
flowdronetello.py

FlowDroNet implementation for the DJI/Ryze Tello

Written by Moritz Sperling
Based on the work of A. Loquercio et al., 2018 (https://github.com/uzh-rpg/rpg_public_dronet)
and P. Ferriere (https://github.com/philferriere/tfoptflow)
and D. Fuentes (https://github.com/damiafuentes/DJITelloPy)

Licensed under the MIT License (see LICENSE for details)
"""

import os
import sys
import cv2
import json
import time
import pygame
import djitellopy
import numpy as np
from copy import deepcopy
from pygame.locals import *
localpath = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, localpath + '/../workflow/util/')
from img_utils import flow_to_img, pred_as_bar, pred_as_indicator
sys.path.insert(0, './tfoptflow/tfoptflow/')
from model_flowdronet import ModelFlowDroNet, _DEFAULT_FLOWDRONET_OPTS


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
            - WSAD: Forward, backward, left and right.
            - Q and E: Counter clockwise and clockwise rotations
            - R and F: Up and down.
            - P: Switch through controllable parameters
            - + and -: Raise or lower parameter
            - #: Enable / Disable FlowDroNet
            - O: Show / Hide optical flow
            - C: Toggle recording of frames
    """

    def __init__(self):

        # FlowDroNet Configuration
        self.pwcnet_ckpt_path = localpath + '/models/pwc-net/pwcnet.ckpt-11000'
        self.dronet_model_path = localpath + '/models/FlowDroNet/model_graph_final.pb'
        self.output_folder = '/recordings/test_0'
        self.logfile = "log.json"
        self.FPS = 10
        self.hud_scale = 1.5
        self.starting_speed = 100

        # Configuration of controllable parameters
        # initial values
        self.params = {
            'v_max': 50,
            'r_max': 100,
            'r_scale': 0,
            'prev_len': 3,
            'prev_coll': 0.6,
            'prev_steer': 0.3,
            'reverse_thresh': 0.6,
            'reverse_offset': 0.7,
        }
        # stepsize
        self.params_d = {
            'v_max': 5,
            'r_max': 5,
            'r_scale': 50,
            'prev_len': 1,
            'prev_coll': 0.1,
            'prev_steer': 0.1,
            'reverse_thresh': 0.1,
            'reverse_offset': 0.1,
        }
        # max (min is 0)
        self.params_m = {
            'v_max': 100,
            'r_max': 100,
            'r_scale': 1000,
            'prev_len': 10,
            'prev_coll': 1,
            'prev_steer': 1,
            'reverse_thresh': 1,
            'reverse_offset': 1,
        }

        # Logging
        self.log_info = {
            'cur_v': [],
            'cur_s': [],
            'prd_c': [],
            'prd_s': []
        }

        # Init internal variables (do not change anything below here)
        self.for_back_velocity = 0  # Drone velocities between -100~100
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.internalSpeed = 100
        self.send_rc_control = False
        self.v_old = [0, ]
        self.s_old = [0, ]
        self.was_dronet = True
        self.last_pred_col = 0
        self.last_pred_ang = 0
        self.last_time = time.time()
        self.is_armed = False
        self.battery_percentage = 0
        self.should_stop = False
        self.record_data = False
        self.show_raw_data = False
        self.frame = None
        self.current_parameter = 0
        self.param_keys = list(self.params.keys())
        self.log_count = 0

        # Configure the pwc-net model for inference, starting with the default options
        self.nn_opts = deepcopy(_DEFAULT_FLOWDRONET_OPTS)
        self.nn_opts['verbose'] = False
        self.nn_opts['batch_size'] = 1
        self.nn_opts['ckpt_path'] = self.pwcnet_ckpt_path
        self.nn_opts['dronet_mode'] = 'rgb'
        self.nn_opts['dronet_model_path'] = self.dronet_model_path
        self.target_size = (int(self.nn_opts['y_shape'][1]), int(self.nn_opts['y_shape'][0]))

        # Create output folder if not exists
        if not os.path.exists(os.path.join(self.output_folder, str(self.log_count).zfill(2))):
            os.makedirs(os.path.join(self.output_folder, str(self.log_count).zfill(2)))

        # Init pygame with display
        pygame.init()
        pygame.display.init()
        pygame.display.set_caption("FlowDroNeTello")
        self.screen = pygame.display.set_mode([int(self.hud_scale * self.target_size[0]) + 44,
                                               int(self.hud_scale * self.target_size[1]) + 36])

        # Show bootscreen
        loadimg = cv2.imread(localpath + "/../misc/images/startuplogo.png")
        loadimg = np.fliplr(loadimg)
        loadimg = np.rot90(loadimg)
        loadimg = pygame.surfarray.make_surface(loadimg)
        self.screen.fill([0, 0, 0])
        self.screen.blit(loadimg, (0, 0))
        pygame.display.update()
        pygame.time.set_timer(USEREVENT + 1, int(1. / self.FPS * 1000))

        # Instantiate the model in inference mode and display the model configuration
        self.nn = ModelFlowDroNet(mode='test', options=self.nn_opts)
        self.nn.print_config()

        # Init Tello object that interacts with the Tello drone
        self.tello = djitellopy.Tello()

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
            print("Speed as low as possible")
            return

        # In case streaming is on. This happens when we quit this program without the escape key.
        if not self.tello.streamoff():
            print("Could not stop video stream")
            return

        if not self.tello.streamon():
            print("Could not start video stream")
            return

        # start framereader and read initial frame
        frame_read = self.tello.get_frame_read()
        img = cv2.resize(frame_read.frame, self.target_size)
        prv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Main Loop
        while not self.should_stop:
            # sometimes read battery status
            if np.random.random() < 0.05:
                self.battery_percentage = self.tello.get_battery()

            # read frame
            img = cv2.resize(frame_read.frame, self.target_size)
            cur = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # create img pair
            if self.for_back_velocity > 0:
                img_pair = [prv, cur]
            else:
                img_pair = [cur, prv]

            # get output from flowdronet
            if self.show_raw_data or self.is_armed:
                flow = self.get_dronet_command(img_pair)
            else:
                flow = None
            prv = cur

            # produce hud
            hud = self.update_hud(cur, flow)

            # save frame and hud if recording
            if self.record_data:
                self.save_img(frame_read.frame, 'rec_img')
                self.save_img(hud, 'rec_hud')

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

        # save log
        if self.record_data:
            with open(os.path.join(self.output_folder, str(self.log_count).zfill(2), self.logfile) , "w") as f:
                json.dump(self.log_info, f)

        # always call before finishing to deallocate resources
        self.tello.end()
        pygame.quit()
        exit(0)

    def get_dronet_command(self, img_pair):

        # inference
        flow, steer_coll = self.nn.flowdronet_predict((img_pair,))
        theta, p_t = steer_coll[0], steer_coll[1]
        self.last_pred_col = p_t
        self.last_pred_ang = theta

        # calculate resulting velocity:
        if p_t < self.params['reverse_thresh']:
            # if the probability to crash is low, then go forward with v_max * (1 - prob) but also
            # take a percentage of the previous average speed into account:
            # v_new = (1 - a) * v_old +  a * (1 - p_t) * v_max
            mean_velocity = np.mean(self.v_old)
            prev_velocity = (1 - self.params['prev_coll']) * mean_velocity if not np.isnan(mean_velocity) else 0
            velocity = prev_velocity + self.params['prev_coll'] * (1 - p_t) * self.params['v_max']
        else:
            # if the probability to crash is high, instantly move backwards to keep flow alive.
            # v_new = (1 - d - p_t) * v_max
            velocity = (1 - self.params['reverse_offset'] - p_t) * self.params['v_max']
            self.v_old.clear()

        # theta_new = (1 - b) * s_old + b * theta * scale with range[-r_max .. r_max]
        steering = np.clip((1 - self.params['prev_steer']) * np.mean(self.s_old)
                           + self.params['prev_steer'] * theta * self.params['r_scale'],
                       -self.params['r_max'], self.params['r_max']) * np.clip(p_t * 2, 0.2, 1)

        # set forward and yaw velocity if dronet is armed
        if self.is_armed:
            # append current velocity and steering angle for averaging
            self.v_old.append(velocity)
            self.s_old.append(steering)

            # set actual velocities
            self.for_back_velocity = int(velocity)
            self.yaw_velocity = int(steering)

            # delete oldest values
            if len(self.v_old) > self.params['prev_len']:
                self.v_old.pop(0)
                self.s_old.pop(0)

        if self.record_data:
            self.log_info['cur_v'].append(int(velocity))
            self.log_info['cur_s'].append(int(steering))
            self.log_info['prd_c'].append(float(p_t))
            self.log_info['prd_s'].append(float(theta))

        return flow

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
            self.is_armed = False
            self.for_back_velocity = self.params['v_max']
        elif key == pygame.K_s:  # set backward velocity
            self.is_armed = False
            self.for_back_velocity = -self.params['v_max']
        elif key == pygame.K_a:  # set left velocity
            self.is_armed = False
            self.left_right_velocity = -self.params['v_max']
        elif key == pygame.K_d:  # set right velocity
            self.is_armed = False
            self.left_right_velocity = self.params['v_max']
        elif key == pygame.K_r:  # set up velocity
            self.is_armed = False
            self.up_down_velocity = self.params['v_max']
        elif key == pygame.K_f:  # set down velocity
            self.is_armed = False
            self.up_down_velocity = -self.params['v_max']
        elif key == pygame.K_e:  # set yaw clockwise velocity
            self.is_armed = False
            self.yaw_velocity = self.params['r_max']
        elif key == pygame.K_q:  # set yaw counter clockwise velocity
            self.is_armed = False
            self.yaw_velocity = -self.params['r_max']
        elif key == pygame.K_TAB:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_LSHIFT:  # land
            self.is_armed = False
            self.tello.land()
            self.send_rc_control = False
        elif key == pygame.K_SPACE:  # emergency shutdown
            self.is_armed = False
            self.tello.emergency()
            self.send_rc_control = False
            self.should_stop = True
        elif key == pygame.K_HASH:  # arm/disarm dronet
            self.is_armed = not self.is_armed

            # instantly start moving to actually have optical flow and not just noise
            self.for_back_velocity = self.starting_speed if self.is_armed else 0
            self.yaw_velocity = 0
        elif key == pygame.K_p:  # switch through parameters
            if self.current_parameter < len(self.params) - 1:
                self.current_parameter = self.current_parameter + 1
            else:
                self.current_parameter = 0
        elif key == pygame.K_PLUS:  # raise current parameter
            what = self.param_keys[self.current_parameter]
            if self.params[what] < self.params_m[what] - 0.01:
                self.params[what] = self.params[what] + self.params_d[what]
        elif key == pygame.K_MINUS:  # lower current parameter
            what = self.param_keys[self.current_parameter]
            if self.params[what] > 0.01:
                self.params[what] = self.params[what] - self.params_d[what]
        elif key == pygame.K_o:  # show/hide optical flow and predictbars
            self.show_raw_data = not self.show_raw_data
            if self.show_raw_data:
                self.screen = pygame.display.set_mode([int(self.hud_scale * self.target_size[0] + 76),
                                                       int(self.hud_scale * 2 * self.target_size[1] + 76)])
            else:
                self.screen = pygame.display.set_mode([int(self.hud_scale * self.target_size[0] + 44),
                                                       int(self.hud_scale * self.target_size[1] + 36)])
        elif key == pygame.K_c:  # toggle recording of data
            self.record_data = not self.record_data

            if self.record_data:
                if not os.path.exists(os.path.join(self.output_folder, str(self.log_count).zfill(2))):
                    os.makedirs(os.path.join(self.output_folder, str(self.log_count).zfill(2)))
                    print(os.path.join(self.output_folder, str(self.log_count).zfill(2)))
                else:
                    with open(os.path.join(self.output_folder, str(self.log_count).zfill(2), self.logfile), "w") as f:
                        json.dump(self.log_info, f)
                    self.log_count = self.log_count + 1
                    self.log_info['cur_v'].clear()
                    self.log_info['cur_s'].clear()
                    self.log_info['prd_c'].clear()
                    self.log_info['prd_s'].clear()

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

    def update_hud(self, frame, flow):
        """Draw drone info and print on frame"""

        # resize to desired scale
        new_size = (int(self.target_size[0] * self.hud_scale), int(self.target_size[1] * self.hud_scale))
        img = cv2.resize(frame, new_size)

        # show blinking red dot when recording
        if self.record_data and round(time.time()) % 2 == 0:
            cv2.putText(img, "Recording", (img.shape[1] - 187, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), lineType=30)
            if round(time.time()) % 2 == 0:
                cv2.circle(img, (new_size[0] - 30, 30), 15, (255, 0, 0), -1)

        # show battery percentage
        #cv2.putText(img, "Battery: {:d} %".format(int(self.battery_percentage)), (img.shape[1] - 187, img.shape[0] - 12),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), lineType=30)

        # display info on screen
        if self.is_armed:
            stats = ["FlowDroNet active."]
            if self.was_dronet:
                stats.append("Predictions:")
                stats.append("[C: {:4.3f}] [SA: {:4.3f}]".format(float(self.last_pred_col), float(self.last_pred_ang)))
                stats.append("Commands:")
                stats.append("[V: {:03d}] [SA: {:03d}]".format(int(self.for_back_velocity), int(self.yaw_velocity)))
            else:
                stats.append("Command overwritten ...")
        elif self.show_raw_data:
            stats = ["FlowDroNet disarmed.", "Predictions:",
                     "[C: {:4.3f}] [SA: {:4.3f}]".format(float(self.last_pred_col), float(self.last_pred_ang))]
        else:
            stats = ["FlowDroNet disabled."]

        stats.append(self.param_keys[self.current_parameter]
                     + ": {:4.1f}".format(self.params[self.param_keys[self.current_parameter]]))
        for idx, stat in enumerate(stats):
            text = stat.lstrip()
            cv2.putText(img, text, (10, 30 + (idx * 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 0, 0), lineType=30)

        # display yaw velocity as indicator in the middle
        hspacer = np.ones((12, img.shape[1], 3), dtype=np.uint8) * 255
        yawbar = pred_as_indicator(self.yaw_velocity / 100., (20, img.shape[1]), "Yaw Velocity")
        yawbar = cv2.cvtColor(yawbar, cv2.COLOR_BGR2RGB)

        # show flow and predictions if desired
        if (self.show_raw_data or self.is_armed) and flow is not None:
            steerbar = pred_as_indicator(self.last_pred_ang, (20, img.shape[1]), "Steering Pred.")
            steerbar = cv2.cvtColor(steerbar, cv2.COLOR_BGR2RGB)
            flow_img = cv2.resize(flow_to_img(flow), new_size) if flow.shape[2] == 2 else cv2.resize(flow, new_size)
            self.frame = np.vstack((img, hspacer, yawbar, hspacer, steerbar, hspacer, flow_img))
        else:
            self.frame = np.vstack((img, hspacer, yawbar, hspacer))

        # display velocity as bar on the side
        vspacer = np.ones((self.frame.shape[0], 12, 3), dtype=np.uint8) * 255
        velobar = pred_as_indicator(self.for_back_velocity / 100, (20, self.frame.shape[0]), "Velocity",
                                    mode="vertical")
        velobar = cv2.cvtColor(velobar, cv2.COLOR_BGR2RGB)

        # show predictions if desired
        if (self.show_raw_data or self.is_armed) and flow is not None:
            vspacer = np.ones((self.frame.shape[0], 12, 3), dtype=np.uint8) * 255
            collbar = pred_as_bar(self.last_pred_col, (self.frame.shape[0], 20), "Collision Prob.")
            collbar = cv2.cvtColor(collbar, cv2.COLOR_BGR2RGB)
            output = np.hstack((vspacer, self.frame, vspacer, velobar, vspacer, collbar))
        else:
            output = np.hstack((vspacer, self.frame, vspacer, velobar))

        # update output
        self.frame = np.fliplr(output)
        self.frame = np.rot90(self.frame)
        self.frame = pygame.surfarray.make_surface(self.frame)
        self.screen.fill([0, 0, 0])
        self.screen.blit(self.frame, (0, 0))
        pygame.display.update()

        return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    def save_img(self, img, string):
        """ Save frame to disk with timestamp. """
        folder = os.path.join(self.output_folder, str(self.log_count).zfill(2))
        fname = os.path.join(folder, time.strftime(string + "_%Y%m%d_%H%M%S_0.png", time.gmtime()))
        i = 0
        while os.path.exists(fname):
            fname = fname.replace('_' + str(i) + '.', '_' + str(i + 1) + '.', )
            i = i + 1
        cv2.imwrite(fname, img)


def main():
    frontend = FrontEnd()

    # run frontend
    frontend.run()


if __name__ == '__main__':
    main()
