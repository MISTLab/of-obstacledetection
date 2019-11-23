#!/usr/bin/env python3
"""
get_flow_stats.py

Gathers the statistics of all flow files in a direactory.
To be used with the matlab script eval_flow_stats.m

Written by Moritz Sperling

Licensed under the MIT License (see LICENSE for details)
"""
import glob
import os
import sys
import absl.flags as gflags
from tqdm import tqdm
localpath = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, localpath + '/../workflow/util/')
import img_utils as iu
from common_flags import FLAGS


def _main():
    name = FLAGS.name or "flow_stats"
    folder_in = FLAGS.input_dir
    img_mode = "flow"
    n = 200
    lim = 200
    scale = 1/100
    exts = ['*.npy', '*.pfm', '*.flo', '*.bin', '*.png']
    n_im = 0
    clip = False

    # place results in input dir if no output dir is set
    if FLAGS.output_dir is None:
        folder_out = folder_in
    else:
        folder_out = FLAGS.output_dir

    # init
    flow_collector = iu.FlowStatsCollector(folder_out, n=n, lim=lim, name=name, clip=clip)

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

    print("Evaluating: " + ipth + " (" + str(n_im) + " images)")

    # iterate all imgs
    for i, fname in tqdm(enumerate(fnames)):
        flow = iu.load_img(fname, img_mode=img_mode) * scale
        flow_collector.collect_flow_stats(flow)

    # save data
    flow_collector.print_flow_stats()
    flow_collector.write_flow_stats()


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
