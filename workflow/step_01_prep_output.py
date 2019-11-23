#!/usr/bin/env python3
"""
step_01_prep_output.py

First step of workflow: prepare output from Unity and store as DroNet dataset.

Written by Moritz Sperling

Licensed under the MIT License (see LICENSE for details)
"""
import os
import csv
import cv2
import sys
import random
import shutil
import numpy as np
from tqdm import tqdm
import absl.flags as gflags
import util.img_utils as iu
import util.misc_utils as misc
from util.common_flags import FLAGS
from sklearn.externals import joblib
from sklearn.preprocessing import MaxAbsScaler


# path and filename options
localpath = os.path.dirname(os.path.realpath(__file__))
filenames = {'collision': "labels.txt",
             'steering': "sync_steering.txt"}

# special options
verbose_val = False
create_subfolders = True

# init counter
c = dict(all=0, exp_in=0, exp_out=0, bright=0, dark=0, zero_steering=0, zero_collision=0, dismissed=0, cloud=0,
         good=0, steering=0, collision=0, train=0, val=0, tmp=0)


#
#        888b     d888          d8b               8888888888                         888    d8b
#        8888b   d8888          Y8P               888                                888    Y8P
#        88888b.d88888                            888                                888
#        888Y88888P888  8888b.  888 88888b.       8888888 888  888 88888b.   .d8888b 888888 888  .d88b.  88888b.
#        888 Y888P 888     "88b 888 888 "88b      888     888  888 888 "88b d88P"    888    888 d88""88b 888 "88b
#        888  Y8P  888 .d888888 888 888  888      888     888  888 888  888 888      888    888 888  888 888  888
#        888   "   888 888  888 888 888  888      888     Y88b 888 888  888 Y88b.    Y88b.  888 Y88..88P 888  888
#        888       888 "Y888888 888 888  888      888      "Y88888 888  888  "Y8888P  "Y888 888  "Y88P"  888  888
#


def _main():
    global c, s

    # load settings or use args or defaults
    s = misc.load_settings_or_use_args(FLAGS)

    # custom settings
    #s['center_flow'] = True

    # init misc settings
    s['input_dir'] = FLAGS.input_dir

    # init folders
    outfolder = create_outfolder()
    misc.del_and_recreate_folder(os.path.join(outfolder, s['train_dir']))
    misc.del_and_recreate_folder(os.path.join(outfolder, s['val_dir']))
    misc.del_and_recreate_folder(os.path.join(outfolder, s['model_dir']))

    # read labels
    labels_fname = os.path.join(s['input_dir'], "labels.txt")
    c['all'] = file_len(labels_fname) - 2
    data = read_labels(labels_fname)

    # clamp and normalize steering angles
    scaler = MaxAbsScaler()
    data['steering'] = np.clip(data['steering'], -s['max_steer_angle'], s['max_steer_angle'])
    data['steering'] = np.squeeze(scaler.fit_transform(data['steering'].reshape(-1, 1)))

    # save scaler
    joblib.dump(scaler, os.path.join(outfolder, s['scaler_filename']))

    # split into sets and filter out unwanted elements of the data
    candidates = prep_data(data)

    # load cloud image for preprocessing
    cloud = cv2.imread(localpath + s['cloud_file'], 0).astype(np.float)

    # init flow stat collector
    flowdataname = "flowstats_" + s['name'] if s['name'] != "same" else "flow_statistics"
    flow_stat_collector = iu.FlowStatsCollector(outfolder, lim=200, name=flowdataname, n=s['flow_stat_bin_count'], clip=False)

    # iterate through all experiments
    for exp in tqdm(candidates, desc="Preparing Dataset: ", maxinterval=1, unit="Exp."):
        # global counter
        c['tmp'] = c['tmp'] + len(exp)

        # load and validate images (check if too dark/bright)
        imgs = load_set_images(exp['filename'], s['input_dir'])

        # check for too bright or dark images and remove them
        del_idxs = validate_set_images(imgs)
        imgs = np.delete(imgs, del_idxs, 0)
        exp = np.delete(exp, del_idxs, 0)

        # apply quote of items that are allowed to have zero data
        if len(exp) > s['min_data_per_exp']:
            del_idxs = apply_zero_quote(exp)
            imgs = np.delete(imgs, del_idxs, 0)
            exp = np.delete(exp, del_idxs, 0)

        # process and write set if still enough data available
        if len(exp) > s['min_data_per_exp']:
            new_imgs = apply_preprocessing(imgs, cloud)

            # get stats of flow images and rescale so that the mean magnitude = 1
            if s['img_mode'] in ["flow", "flow_as_rgb", "flow_as_mag"]:
                flow_stat_collector.collect_exp_flow_stats(new_imgs)

            # write experiment set to train or val
            train_or_val = "train" if random.random() > s['train_val_qouta'] else "val"
            folder = create_subfolder(exp, os.path.join(outfolder, s[train_or_val + "_dir"]))
            write_set(exp, new_imgs, folder)
            c[train_or_val] = c[train_or_val] + len(new_imgs)

            # logging
            key = get_type(exp)
            c['good'] = c['good'] + len(exp)
            c[key] = c[key] + len(exp)
        else:
            if verbose_val:
                print("ID: {:5d}; \t Info: Not enough data left after image validation".format(exp['id'][0]))
            c['dismissed'] = c['dismissed'] + len(exp)

    # Save flow statistics
    flow_stats_available = False
    if s['img_mode'] in ["flow", "flow_as_rgb", "flow_as_mag"]:
        flow_stat_collector.write_flow_stats()
        flow_stats_available = True

    # Store settings in output folders
    s['img_channels'] = 3 if s['img_mode'] == "flow_as_rgb" else s['img_channels']  # set to 3 channels for rgb
    s['img_channels'] = 1 if s['img_mode'] == "flow_as_mag" else s['img_channels']  # set to 1 channel for magnitude
    s['img_mode'] = "rgb" if s['img_mode'] == "flow_as_rgb" else s['img_mode']  # set img mode to rgb after conversion
    s['img_mode'] = "grayscale" if s['img_mode'] in ["depth", "flow_as_mag"] else s['img_mode']  # set img mode to grayscale if depth or magnitude
    settings_out_filename = os.path.join(outfolder, s['settings_fname'])
    misc.write_to_file(s, settings_out_filename, beautify=True)

    # create eval folder
    if create_subfolders:
        try:
            os.mkdir(os.path.join(outfolder, s['eval_dir']))
        except:
            print("Warning: Unable to create eval folder.")

    # Output of statistics
    print("\n\n## " + misc.colorize("[RESULTS]", "magenta") + " ############################################"
          + "\n\n- " + misc.colorize("INPUT", "cyan") + " ----------------------------------------------"
          + "\nFolder:     " + FLAGS.input_dir
          + "\nLabels:     " + labels_fname
          + "\nImages:     {:d} ({:d} Experiments)".format(c['all'], c['exp_in'])
          + "\n\n- " + misc.colorize("VALIDATION", "cyan") + " -----------------------------------------"
          + "\nToo bright: {:d}".format(c['bright'])
          + "\nToo dark:   {:d}".format(c['dark'])
          + "\nDismissed:  {:d} ({:3.1f} %)".format(c['dismissed'], c['dismissed'] / c['all'] * 100.))

    if flow_stats_available:
        print("\n- " + misc.colorize("FLOW STATS", "cyan") + " -----------------------------------------")
        flow_stat_collector.print_flow_stats()

    print("\n- " + misc.colorize("OUTPUT", "cyan") + " ---------------------------------------------"
          + "\nFolder:     " + outfolder
          + "\nGOOD:       {:d} ({:3.1f} %; {:d} Experiments)".format(c['good'],
                                                                      c['good'] / c['all'] * 100.,
                                                                      c['exp_out'])
          + "\nClouded:    {:d} ({:3.1f} %)".format(c['cloud'], c['cloud'] / c['good'] * 100.)
          + "\nSteering:   {:d} (zero: {:d})".format(c['steering'], c['zero_steering'])
          + "\nCollision:  {:d} (zero: {:d})".format(c['collision'], c['zero_collision'])
          + "\nTrain/Val:  {:d}/{:d} "
            "(Val. Quota: {:3.1f} %)".format(c['train'], c['val'], c['val'] / c['train'] * 100.)
          + "\n\n#########################################################\n")

#
#        8888888                                               8888888b.                                                     d8b
#          888                                                 888   Y88b                                                    Y8P
#          888                                                 888    888
#          888   88888b.d88b.   8888b.   .d88b.   .d88b.       888   d88P 888d888 .d88b.   .d8888b .d88b.  .d8888b  .d8888b  888 88888b.   .d88b.
#          888   888 "888 "88b     "88b d88P"88b d8P  Y8b      8888888P"  888P"  d88""88b d88P"   d8P  Y8b 88K      88K      888 888 "88b d88P"88b
#          888   888  888  888 .d888888 888  888 88888888      888        888    888  888 888     88888888 "Y8888b. "Y8888b. 888 888  888 888  888
#          888   888  888  888 888  888 Y88b 888 Y8b.          888        888    Y88..88P Y88b.   Y8b.          X88      X88 888 888  888 Y88b 888
#        8888888 888  888  888 "Y888888  "Y88888  "Y8888       888        888     "Y88P"   "Y8888P "Y8888   88888P'  88888P' 888 888  888  "Y88888
#                                            888                                                                                               888
#                                       Y8b d88P                                                                                          Y8b d88P
#                                        "Y88P"                                                                                            "Y88P"
#


def validate_set_images(imgs):
    """
    Makes sure that the input images are ok.
    :param imgs:    List of images to check
    :return:        List of indices to delete
    """

    del_indices = []

    # iterate all images
    for i, img in enumerate(imgs):
        # convert to regular image if necessary
        if s['img_mode'] in ["flow", "flow_as_rgb", "flow_as_mag"]:
            img_s = cv2.resize(img, (64, 48), interpolation=cv2.INTER_NEAREST)
            img_s = np.clip(img_s, -s['flow_limit'], s['flow_limit'])
            img_s = iu.flow_to_img(img_s, normalize=True, return_mag=True)
        else:
            img_s = cv2.resize(img, (64, 48), interpolation=cv2.INTER_AREA)

        # add channel if grayscale
        if s['img_mode'] in ["grayscale", "depth", "flow", "flow_as_rgb", "flow_as_mag"]:
            img_s = img_s.reshape((img_s.shape[0], img_s.shape[1], 1))

        # detect images that are too bright
        limpixels = 0
        for k in range(img_s.shape[2]):
            img_c = img_s[:, :, k]
            limpixels = limpixels + np.sum(np.bincount(img_c.ravel(), minlength=256)[240:])

        if limpixels > s['bright_limit'] * img_s.shape[2]:
            del_indices.append(i)
            c['bright'] = c['bright'] + 1
            continue

        # detect images that are too dark
        if (np.mean(img_s)) < s['dark_limit']:
            del_indices.append(i)
            c['dark'] = c['dark'] + 1
            continue

    return del_indices


def apply_preprocessing(imgs, cloud):
    """
    Apply preprocessing to images of a set.
    :param imgs:    List of images
    :param cloud:   Cloud image to apply on input
    :return:        List of prepped images
    """

    out = []

    # iterate all images
    for img in imgs:

        # do preprocessing
        s['gaussian_blur_size'] = random.randrange(s['gaussian_blur_min'], s['gaussian_blur_max'], 2)
        img_out = cv2.GaussianBlur(img, (s['gaussian_blur_size'], s['gaussian_blur_size']), 0)

        # center optical flow
        if s['img_mode'] in ["flow", "flow_as_rgb", "flow_as_mag"] and s['center_flow']:
            m = round(img_out.shape[1] / 2)
            h = img_out.shape[0]
            img_out[..., 0] = img_out[..., 0] - np.mean(img_out[h-50:h-10, m-200:m+200, 0])

        img_out = iu.sp_generator(img_out, amount=s['saltnpepper_amount'], img_mode=s['img_mode'])
        img_out = cv2.GaussianBlur(img_out, (s['gaussian_blur_size'], s['gaussian_blur_size']), 0)

        if s['img_mode'] in ["flow", "flow_as_rgb", "flow_as_mag"]:
            for _ in range(0, random.randrange(0, round(s['median_blur_max'] / 3))):
                img_out[..., 0] = cv2.medianBlur(img_out[..., 0], 3)
                img_out[..., 1] = cv2.medianBlur(img_out[..., 1], 3)

            # clip flow data to desired range
            img_out = np.clip(img_out, -s['flow_limit'], s['flow_limit'])

            # slightly randomize flow scaling per channel
            img_out[..., 0] = img_out[..., 0] + img_out[..., 0] * s['channel_randomness'] * (np.random.random() - 0.5)
            img_out[..., 1] = img_out[..., 1] + img_out[..., 1] * s['channel_randomness'] * (np.random.random() - 0.5)

            # scale flow to have magnitude of roughly 1 plus a bit of randomization
            if s['scale_flow_to_mag1']:
                flow_magnitude = np.linalg.norm(img_out, axis=2)
                img_out = img_out / np.mean(flow_magnitude) * (1 + (np.random.random() - 0.3) * 0.3)
        else:
            median_blur_size = random.randrange(s['median_blur_min'], s['median_blur_max'], 2)
            img_out = cv2.medianBlur(img_out.astype(np.uint8), median_blur_size)

        # cloud overlay: apply cloud
        clouded = False

        # add channel if grayscale
        if s['img_mode'] in ["grayscale", "depth"]:
            img_out = img_out.reshape((img_out.shape[0], img_out.shape[1], 1))

        # apply cloud to each channel
        for i in range(0, s["img_channels"]):

            if random.random() <= s['cloud_quota']:
                # cloud overlay: get roi from cloud
                h, w = cloud.shape
                y = int(random.randrange(0, w - s['img_width'] - 1))
                x = int(random.randrange(0, h - s['img_height'] - 1))
                cloud_roi = cloud[x:x + s['img_height'], y:y + s['img_width']]
                cloud_roi = cloud_roi / np.amax(cloud_roi) * np.amax(img_out[:, :, i]) * 0.5

                # apply to image
                cloud_strength = random.random() * s['cloud_max']
                img_out[:, :, i] = cv2.addWeighted(img_out[:, :, i].astype(cloud_roi.dtype), 1 - cloud_strength, cloud_roi, cloud_strength, 0)
                clouded = True

        if clouded:
            c['cloud'] = c['cloud'] + 1

        # finishing blur
        out.append(cv2.GaussianBlur(img_out, (s['gaussian_blur_size'], s['gaussian_blur_size']), 0))
    return out


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


def read_labels(filename):
    """
    Reads the labels from text file into suitable datastructure.
    :param filename:    Input Filename
    :return:            All the data in structured np.array
    """

    tmp = []
    with open(filename, newline='') as i:
        # ignore first 2 lines
        for _ in range(2):
            next(i)

        # iterate through labels file and save in temporary list of tuples
        for k, row in enumerate(csv.reader(i, delimiter=';')):
            tmp.append((int(row[0]),
                        row[1],
                        float(row[2]),
                        float(row[3]),
                        float(row[4]),
                        int(row[5]),
                        float(row[6]),
                        int(row[7]),
                        int(row[8]),
                        int(row[9])))

    i.close()

    # convert tuples list to a nice datastructure
    data = np.array(tmp, dtype=([('id', 'i4'),
                                 ('filename', '<U16'),
                                 ('pos_x', 'f4'),
                                 ('pos_y', 'f4'),
                                 ('pos_z', 'f4'),
                                 ('exp_type', 'i4'),
                                 ('steering', 'f4'),
                                 ('collision', 'i4'),
                                 ('obst_id', 'i4'),
                                 ('epoch', 'i4')]))
    return data


def prep_data(data):
    """
    Prepares the input data: group into category and sets and dismiss unneccessary data.
    :param data:    Input Data from labels
    :return:        Reduced dataset, split into experiments (sets)
    """

    n = 0
    candidates = []

    # iterate trough all possible IDs
    last_idx = data['id'][-1]
    for i in range(last_idx + 1):

        # select items with same set id
        temp_set = data[data['id'] == i]

        if len(temp_set) > 0:
            # redefine type of data if necessary

            # get start of collision/steering and count
            nc, idxc = get_exp_stats(temp_set, 'collision')
            ns, idxs = get_exp_stats(temp_set, 'steering')

            # validate set and redefine experiment type if appropriate
            is_valid, new_exp_type = validate_exp_data(nc, ns, idxc, idxs, temp_set['exp_type'])
            if is_valid:

                # set new data according to validation
                temp_set['exp_type'] = new_exp_type

                # select range for input
                if temp_set['exp_type'][0] == 1:
                    # steering
                    start = 1
                    end = len(temp_set) - s['ignore_last_images']
                else:
                    # collision
                    # dismiss items after collision
                    temp_set = temp_set[:nc+idxc]

                    # dismiss items before collision
                    temp_set = np.delete(temp_set, range(idxc, idxc + s['ignore_before_coll']), 0)

                    start = s['ignore_first_images']
                    end = len(temp_set) - 1

                # put set into categories when enough data available
                quote = 2 - (s['zero_quota']['steering'] + s['zero_quota']['collision']) / 2
                if end - start > s['min_data_per_exp'] * quote:
                    # dismiss items at begining and end
                    temp_set = temp_set[start:end]

                    # append to candidates
                    candidates.append(temp_set)
                    n = n + len(temp_set)
            else:
                # output if validation failed
                if verbose_val:
                    print("ID: {:5d}; \t Info: ".format(i) + new_exp_type)

    # log stats
    c['dismissed'] = c['dismissed'] + len(data) - n
    c['exp_in'] = last_idx + 1
    c['exp_out'] = len(candidates)

    return candidates


def validate_exp_data(nc, ns, idxc, idxs, data):
    """
    Makes sure that the experiment type is correctly assigned and enough data is avaiable.
    :param nc:      Number of collisions
    :param ns:      Number of steering angles
    :param idxc:    Begin of collision
    :param idxs:    Begin of steering
    :param data:    List of experiment type from data
    :return:        True and list of appropriate data types if experiment is ok, otherwise False and 0
    """

    # when steering/collision occures too early the experiment is most likely invalid
    if ((idxs < len(data) / 3) and (idxs != -1)) or ((idxc < len(data) / 3) and (idxc != -1)):
        return False, 'Steering/Collision: too early'

    if data[0] == 1:
        # steering experiment
        if ns > s['min_real_data']:
            # when enough data available
            if nc < s['max_coll_in_steer_data']:
                # data good if no major collision occured
                return True, data
            else:
                # experiment most likely chaotic, so invalid
                return False, 'Steering: too many collisions'
        else:
            # when low on steering data
            if nc > s['min_real_data']:
                # switch experiment type if collision data available
                return True, np.zeros_like(data)
            else:
                # otherwise invalid
                return False, 'Steering: not enough data'
    else:
        # collision experiment
        if nc > s['min_real_data']:
            # data is good when enough data available
            return True, data
        else:
            # when low on collision data
            if ns > s['min_real_data']:
                # switch experiment type if collision data available
                return True, np.ones_like(data)
            else:
                # otherwise invalid
                return False, 'Collision: not enough data'


def get_exp_stats(data, key):
    """
    Returns the amount of nonzero data and the index of first nonzero element.
    :param data:    Data
    :param key:     Key for collision or steering
    :return:
    """
    n = 0
    idx = -1
    found = False
    for k, d in enumerate(data):
        if d[key] != 0:
            n = n + 1
            if not found:
                idx = k
                found = True
    return n, idx


def apply_zero_quote(data):
    """
    Filters out a percentage of data with label 0
    :param data:    Data
    :return:        Data without a percentage of 0-data
    """
    del_idxs = []
    key = get_type(data)
    for i, _ in enumerate(data):
        if data[key][i] == 0:
            if random.random() > s['zero_quota'][key]:
                del_idxs.append(i)
            else:
                c['zero_' + key] = c['zero_' + key] + 1

    return del_idxs


def load_set_images(fnames, folder):
    """
    Loads a set of images.
    :param fnames:      Filenames of images to load
    :param folder:      Folder where the images are located
    :return:            List of images
    """
    imgs = []

    # for real filename
    if s['img_mode'] in ["flow", "flow_as_rgb", "flow_as_mag"]:
        s['img_channels'] = 2
        imgprefix = "flow_raw_"
        imgext = ".bin"
    elif s['img_mode'] == "grayscale":
        s['img_channels'] = 1
        imgprefix = "img_"
        imgext = ".png"
    elif s['img_mode'] == "depth":
        s['img_channels'] = 1
        imgprefix = "depth_"
        imgext = ".png"
    else:
        s['img_channels'] = 3
        imgprefix = "flow_"
        imgext = ".png"

    # load all images
    for fname in fnames:
        fname = os.path.splitext(fname)[0]
        filename = os.path.join(folder, (imgprefix + fname + imgext))
        img = iu.load_img(filename, img_mode=s['img_mode'], target_size=(s['img_width'], s['img_height']),
                          crop_size=(s['crop_img_width'], s['crop_img_height']))

        if img is None:
            print("Image not found: " + filename)
            sys.exit(1)

        imgs.append(img)
    return imgs


def write_set(data, imgs, folder):
    """
    Writes experiment set to disk.
    :param data:    Data to write
    :param imgs:    Images to write
    :param folder:  Output folder
    """

    # write images
    for idx, img in enumerate(imgs):

        # write output
        if s['img_mode'] == "flow":
            outfile = os.path.join(folder, "images", str(idx).zfill(5) + ".png")
            iu.flow_write(img, outfile, compress=True)
        elif s['img_mode'] == "flow_as_rgb":
            img = iu.flow_to_img(img)
            outfile = os.path.join(folder, "images", str(idx).zfill(5) + ".png")
            cv2.imwrite(outfile, img)
        elif s['img_mode'] == "flow_as_mag":
            img = iu.flow_to_img(img, return_mag=True)
            outfile = os.path.join(folder, "images", str(idx).zfill(5) + ".png")
            cv2.imwrite(outfile, img)
        else:
            outfile = os.path.join(folder, "images", str(idx).zfill(5) + ".png")
            cv2.imwrite(outfile, img)

    # select either steering or collision data
    key = get_type(data)

    # write labels file
    with open(os.path.join(folder, filenames[key]), 'w') as f:
        for item in data[key]:
            f.write("%s\n" % str(item))
    f.close()


def create_outfolder():
    """
    Create output directory
    :return:
    """

    # check for output dir flag
    if FLAGS.output_dir is None:
        if s['name'] == "same":
            indir = FLAGS.input_dir
            if indir[-1] == "/":
                indir = indir[:-1]
            folder = indir + "_prepped"
        else:
            folder = FLAGS.input_dir + '_' + s['name']

        # delete and recreate output folder
        misc.del_and_recreate_folder(folder)
    else:
        folder = FLAGS.output_dir if s['name'] == "same" else FLAGS.output_dir

    return folder


def create_subfolder(data, folder):
    """
    Prepares folder for output.
    :param data:    Data sample
    :param folder:  Data sample
    :return:        Folder path
    """

    # folderpath
    outfolder = os.path.join(folder,
                             str('ep_' + str(int(data['epoch'][0]))
                                 + '_' + str(int(data['id'][0])).zfill(5)
                                 + '_' + get_type(data)
                                 + '_obs' + str(data['obst_id'][0])))

    # delete and recreate output folder
    if os.path.exists(outfolder):
        shutil.rmtree(outfolder)
    os.makedirs(os.path.join(outfolder, "images"))

    return outfolder


def get_type(data):
    # Gets type key of data
    return 'steering' if data['exp_type'][0] == 1 else 'collision'


def file_len(fname):
    # Gets file length.
    i = 0
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def main(argv):
    # Utility main to load flags
    try:
        FLAGS(argv)  # parse flags
        #assert (FLAGS.input_dir is not None), "No input directory given"
    except gflags.Error:
        print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
        sys.exit(1)
    _main()


if __name__ == "__main__":
    main(sys.argv)
