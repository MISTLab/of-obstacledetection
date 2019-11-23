#!/usr/bin/env python3
"""
create_eval_video.py

This script creates a video from two advanced evaluation results (step_05_evaluate_set.py).
Useful for comparing the results.

Written by Moritz Sperling

Licensed under the MIT License (see LICENSE for details)
"""
import os
import cv2
import glob
import numpy as np
from tqdm import tqdm


input_real = "/data/dronetoriginal/eval/eval_adv"
input_flow = "/data/flowdronet/eval/eval_adv"
output_name = "/data/comparison.mp4"
ext_real = "png"
ext_flow = "png"

fps = 20
vidsize = (980, 720)
imgsize = (900, 300)

# check if single set or multiple sets to test
is_single_run = os.path.isdir(os.path.join(input_real, 'images'))
if is_single_run:
	both_folders = [[input_real, input_flow]]
else:
	tmp = next(os.walk(os.path.join(input_real, '.')))[1]
	both_folders = []
	for f in tmp:
		if os.path.isdir(os.path.join(input_flow, f)):
			subfolders = [os.path.join(input_real, f), os.path.join(input_flow, f)]
			both_folders.append(subfolders)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(output_name, fourcc, fps, vidsize)

# iterate through folders
for l, folders in enumerate(tqdm(sorted(both_folders))):

	ipth_real = os.path.join(folders[0], 'images/*.' + ext_real)
	imgs_real = sorted(glob.glob(ipth_real))

	ipth_flow = os.path.join(folders[1], 'images/*.' + ext_flow)
	imgs_flow = sorted(glob.glob(ipth_flow))

	if len(imgs_real) != len(imgs_flow):
		imgs_real = imgs_real[1:]

	for i in range(0, len(imgs_real)):
		img_real = cv2.imread(imgs_real[i], cv2.IMREAD_COLOR)
		img_flow = cv2.imread(imgs_flow[i], cv2.IMREAD_COLOR)

		img_real = cv2.resize(img_real, imgsize)
		img_flow = cv2.resize(img_flow, imgsize)

		shape = np.shape(img_real)
		img_out = np.ones((vidsize[1] , vidsize[0], shape[2]), dtype=img_real.dtype) * 255

		img_out[40:shape[0] + 40, 10:shape[1] + 10, 0:3] = img_real
		img_out[shape[0] + 110:2 * shape[0] + 110, 10:shape[1] + 10, :] = img_flow

		cv2.putText(img_out, 'DroNet - Predictions:', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
		cv2.putText(img_out, 'DroNet - Activation Maps:', (510, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
		cv2.putText(img_out, 'FlowDroNet - Predictions:', (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
		cv2.putText(img_out, 'FlowDroNet - Activation Maps:', (510, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2,cv2.LINE_AA)
		cv2.putText(img_out, 'Experiment: ' + os.path.split(folders[0])[1], (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,cv2.LINE_AA)

		video.write(img_out)

video.release()
print("written to: " + output_name)