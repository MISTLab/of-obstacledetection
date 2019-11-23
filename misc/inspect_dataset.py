#!/usr/bin/env python3
"""
inspect_dataset.py

Inspect a DroNet dataset and delete unwanted experiments.

Written by Moritz Sperling

Licensed under the MIT License (see LICENSE for details)
"""
import os
import csv
import sys
import cv2
import glob
import shutil
import numpy as np
sys.path.insert(0, '../workflow/util/')
import img_utils as iu

datdir = '/data/train'
outdir = '/data/train_inspected'

fexts = ['*.npy', '*.pfm', '*.flo', '*.bin', '*.png']
rexts = ['*.jpg', '*.png']


dirs = sorted(glob.glob(datdir + '/*'))

counter = 0
while counter < len(dirs):
    subdir = os.path.split(dirs[counter])[-1]

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

    # load labels
    lbl = []
    for fn in ['labels.txt', 'sync_steering.txt']:
        labelfilein = os.path.join(datdir, subdir, fn)
        if os.path.isfile(labelfilein):
            with open(labelfilein, newline='') as f:
                for row in csv.reader(f, delimiter=';'):
                    lbl.append(float(row[0]))

    # show all for making a decisison
    for i, item in enumerate(fnames):
        # load data and convert to rgb
        data = iu.load_img(item, img_mode=mode)
        img = iu.flow_to_img(data) if mode == "flow" else data

        labelbar = iu.pred_as_bar(lbl[i], (img.shape[0], 20), "Ground Truth")
        img_out = np.hstack((img, np.ones((img.shape[0], 10, 3), dtype=np.uint8) * 255, labelbar))

        cv2.imshow('frame', img_out)
        key = cv2.waitKey(50 + int(lbl[i]*200)) & 0xFF
        if key == ord(' '):
            break

    # make a decision
    copy_experiment = False
    cv2.imshow('frame', img_out)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('y'):
        copy_experiment = True
        print("Copy: " + subdir)
        counter = counter + 1
    elif key == ord('n'):
        copy_experiment = False
        counter = counter + 1

    # copy experiment if it was ok
    if copy_experiment:
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

        # copy images
        for item in fnames:
            img_name = os.path.split(item)[1]
            outfile = os.path.join(outfolder, img_name)
            shutil.copyfile(item, outfile)

print('Output stored in: ' + outdir)