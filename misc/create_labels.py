#!/usr/bin/env python3
"""
create_labels.py

Script for creating labels for a new collision dataset.

Written by Moritz Sperling

Licensed under the MIT License (see LICENSE for details)
"""
import cv2
import glob
import os, sys
import absl.flags as gflags
localpath = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, localpath + '/../workflow/util/')
from common_flags import FLAGS
from misc_utils import get_experiment_folders

def _main():
    # init variables
    filename = "labels.txt"
    folder = FLAGS.test_dir

    # iterate subdirectories
    dirs = get_experiment_folders(folder)
    for subdir in dirs:

        # ignore hidden folders
        if subdir[0] != '.':

            # get filenames
            path = os.path.join(folder, subdir, 'images/*.jpg')
            imgs = sorted(glob.glob(path))

            labels = []
            collision = False

            # iterate through imgs
            for i, fname in enumerate(imgs):

                # load image and prep for dronet
                img = cv2.imread(fname, cv2.IMREAD_COLOR)

                cv2.putText(img, "[#{:03d}]".format(i), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 0, 255), 2, lineType=30)

                # show img and label as collision when space is pressed
                cv2.imshow('frame',img)
                if cv2.waitKey(100) & 0xFF == ord(' '):
                    # de/activate collision
                    collision = not collision

                # append current label
                if collision:
                    labels.append(1)
                    print('collision')
                else:
                    labels.append(0)
                    print('path clear')

            cv2.destroyAllWindows()

            # produce labels file
            if len(labels) > 2:
                outfile = os.path.join(folder, subdir, filename)
                with open(outfile, 'w') as f:
                    for item in labels:
                        f.write("%s\n" % str(item))
                f.close()


def main(argv):
    # Utility main to load flags
    try:
        FLAGS(argv)  # parse flags
    except gflags.Error:
        print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
        sys.exit(1)
    _main()


if __name__ == "__main__":
    main(sys.argv)