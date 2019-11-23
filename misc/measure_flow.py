#!/usr/bin/env python3
"""
measure_flow.py

Tool for manual measurement of optical flow and comparison to data in flow file.

Written by Moritz Sperling

Licensed under the MIT License (see LICENSE for details)
"""
import os
import cv2
import sys
import numpy as np
localpath = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, localpath + '/../workflow/util/')
import img_utils as iu

path_in_flow0 = '/Users/nfinite/data/tfoptflow/KITTI12_resized/training/flow_noc/000004_10.flo'
path_in_img_1 = '/Users/nfinite/data/tfoptflow/KITTI12_resized/training/colored_0/000004_10.png'
path_in_img_2 = '/Users/nfinite/data/tfoptflow/KITTI12_resized/training/colored_0/000004_11.png'

img_1 = iu.load_img(path_in_img_1)
img_2 = iu.load_img(path_in_img_2)
flowr = iu.load_img(path_in_flow0, img_mode="flow")
go = True
point = (0, 0)


def store_xy(event, x, y):
    # grab references to the global variables
    global point, go

    # if the left mouse button was clicked, record the (x, y) coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        go = False


# setup display window and the mouse callback function
cv2.namedWindow("image")
cv2.setMouseCallback("image", store_xy)

# display the first image and wait for input
while go:
    cv2.imshow("image", img_1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit(0)

# save coordinates
coords_1 = point

# display the second image and wait for input
go = True
while go:
    cv2.imshow("image", img_2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit(0)

# save coordinates and calc differences
coords_2 = point
dx = float(coords_2[0] - coords_1[0])
dy = float(coords_2[1] - coords_1[1])

# print differences of coords
print("Resulting differences of coordinates:")
print("DX: {:5.1f}".format(dx))
print("DY: {:5.1f}".format(dy))

# prep roi for averaging in flow image
roi = flowr[coords_1[1]-5:coords_1[1]+5, coords_1[0]-5:coords_1[0]+5, :]
rx = list(filter(lambda a: a != 0, roi[..., 0].reshape(-1)))
ry = list(filter(lambda a: a != 0, roi[..., 1].reshape(-1)))
fx = np.mean(rx)
fy = np.mean(ry)

# print mean of flow roi
print("Stats from flow image:")
print("FU: {:5.1f}".format(fx))
print("FV: {:5.1f}".format(fy))

# print ratios of magnitude
md = np.sqrt(dx * dx + dy * dy)
mf = np.sqrt(fx * fx + fy * fy)
print("Ratio: {:5.1f}".format((md / mf)))

# draw flow image for verification
flow_out = iu.flow_to_img(flowr)
cv2.drawMarker(flow_out, coords_1, (0, 255, 0), cv2.MARKER_TILTED_CROSS)
cv2.drawMarker(flow_out, coords_2, (0, 0, 255), cv2.MARKER_TILTED_CROSS)
cv2.imshow("image", flow_out)
cv2.waitKey(5000)
cv2.destroyAllWindows()
