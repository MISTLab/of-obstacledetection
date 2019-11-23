#!/usr/bin/env python3
"""
show_flow_dataset.py

Plots all files possibly containing flow data in a directory and its subdirectories.

Written by Moritz Sperling

Licensed under the MIT License (see LICENSE for details)
"""
import os
import sys
import cv2
import glob
import numpy as np
import absl.flags as gflags
localpath = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, localpath + '/../workflow/util/')
import img_utils as iu
from common_flags import FLAGS


def _main():
    folder_in = FLAGS.input_dir
    exts = ['*.npy', '*.pfm', '*.flo', '*.bin', '*.png']
    flow_multi = 20
    n_im = 0

    # get filenames
    ipth = []
    fnames = []
    try:
        for ext in exts:
            ipth = os.path.join(folder_in, "**/", ext)
            fnames = sorted(glob.glob(ipth, recursive=True))
            n_im = len(fnames)
            if n_im != 0:
                break

        if n_im == 0:
            raise IOError("No input files found.")
    except IOError as e:
        print(e)
        exit(0)

    print("Showing: " + ipth + " (" + str(n_im) + " images)")

    # iterate all imgs
    for fname in fnames:
        flow = iu.load_img(fname, img_mode="flow")
        img = iu.flow_to_img(flow)
        imx = cv2.applyColorMap(np.array(flow[..., 0] * flow_multi + 128, dtype=np.uint8), cv2.COLORMAP_RAINBOW)
        imy = cv2.applyColorMap(np.array(flow[..., 1] * flow_multi + 128, dtype=np.uint8), cv2.COLORMAP_RAINBOW)
        out = np.hstack((img, imx, imy))

        # show output
        cv2.imshow('frame', out)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break


def main(argv):
    # Utility main to load flags
    try:
        FLAGS(argv)  # parse flags
    except gflags.Error:
        print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
        sys.exit(1)
    _main()


if __name__ == "__main__":
    main(sys.argv)
