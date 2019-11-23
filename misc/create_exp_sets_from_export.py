#!/usr/bin/env python3
"""
create_exp_sets_from_export.py

Script for splitting a folder with exported frames into single experiments.
Used to create new datasets from videos.

Written by Moritz Sperling

Licensed under the MIT License (see LICENSE for details)
"""
import os
import sys
import cv2
import glob
import shutil
import absl.flags as gflags
localpath = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, localpath + '/../workflow/util/')
from common_flags import FLAGS
from misc_utils import del_and_recreate_folder


def _main():
    # init variables
    folder = FLAGS.test_dir

    # get filenames
    ipth = os.path.join(folder, 'images/*.jpg')
    imgs = sorted(glob.glob(ipth))

    create_new_set = True
    idx = 0
    outfolder = ''

    # iterate through imgs
    for fname in imgs:
        # load image
        img = cv2.imread(fname, cv2.IMREAD_COLOR)

        # show img create next set when space is pressed
        cv2.imshow('frame', img)
        if cv2.waitKey(100) & 0xFF == ord(' '):
            create_new_set = True

        # new set
        if create_new_set:
            if FLAGS.test_dir[-1] == "/":
                outfolder = folder[:-1] + "_" + str(idx).zfill(5) + "/images"
            else:
                outfolder = folder + "_" + str(idx).zfill(2) + "/images"

            # prep output folder
            del_and_recreate_folder(outfolder)

            create_new_set = False
            idx = idx + 1

        # copy file
        file = os.path.split(fname)[1]
        fname_out = os.path.join(outfolder, file)
        shutil.copy(fname, fname_out)
        print(outfolder)

    cv2.destroyAllWindows()


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
