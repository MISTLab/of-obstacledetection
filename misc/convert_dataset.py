#!/usr/bin/env python3
"""
convert_dataset.py

Convert a DroNet dataset to desired style (size, flow mag. etc.).

Written by Moritz Sperling

Licensed under the MIT License (see LICENSE for details)
"""
import os
import sys
import cv2
import glob
import shutil
import numpy as np
from tqdm import tqdm
sys.path.insert(0, '../workflow/util/')
import img_utils as iu

datdir = '/data/test/new_raw'
outdir = '/data/test/new_rgb'
outres = (400, 300)
fexts = ['*.npy', '*.pfm', '*.flo', '*.bin', '*.png']
rexts = ['*.jpg', '*.png']


dirs = sorted(glob.glob(datdir + '/*'))

for subdir in tqdm(dirs):
    subdir = os.path.split(subdir)[-1]

    m = 0
    folder_in = os.path.join(datdir, subdir, 'images/')

    # get filenames
    n_im = 0
    mode = "rgb"
    try:
        # look for regular images
        for ext in rexts:
            fnames = sorted(glob.glob(folder_in + ext))
            n_im = len(fnames)
            if n_im != 0:
                break

        # look for flow files
        if n_im == 0:
            for ext in fexts:
                fnames = sorted(glob.glob(folder_in + ext))
                n_im = len(fnames)
                if n_im != 0:
                    mode = "flow"
                    break

        if n_im == 0:
            raise IOError("No input files found.")
    except IOError as e:
        print(e)
        exit(0)

    # delete and recreate output folder
    outfolder = os.path.join(outdir, subdir, 'images')
    if os.path.exists(outfolder):
        shutil.rmtree(outfolder)
    os.makedirs(outfolder)

    # copy label file
    for fn in ['labels.txt', 'sync_steering.txt']:
        labelfilein = os.path.join(datdir, subdir, fn)
        if os.path.exists(labelfilein):
            labelfileou = os.path.join(outdir, subdir, fn)
            shutil.copyfile(labelfilein, labelfileou)

    # iterate all
    for item in fnames:
        imgname = os.path.split(item)[1]
        outfile = os.path.join(outfolder, imgname).replace(".flo", ".png")
        data = iu.load_img(item, img_mode=mode)

        # center optical flow
        #m = round(data.shape[1] / 2)
        #h = data.shape[0]
        #data[..., 0] = data[..., 0] - np.mean(data[h - 50:h - 10, m - 200:m + 200, 0])

        # apply modifications to image here
        #data[ ..., 0] = data[..., 0] - np.mean(data[..., 0])
        #data[ ..., 1] = data[..., 1] - np.mean(data[..., 1])

        img = iu.flow_to_img(data)
        cv2.imwrite(outfile, img)

print('Output stored in: ' + outdir)