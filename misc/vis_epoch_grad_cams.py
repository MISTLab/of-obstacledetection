#!/usr/bin/env python3
"""
vis_epoch_grad_cams.py

Script for creating a visualisation of the activations over different epochs of training.

Written by Moritz Sperling

Licensed under the MIT License (see LICENSE for details)
"""
import os
import sys
import cv2
import glob
import time
import numpy as np
import absl.flags as gflags
from keras import activations
from keras import backend as k
from keras.models import model_from_json
from vis.utils import utils as vutils
from vis.visualization import visualize_cam, overlay
localpath = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, localpath + '/../workflow/util/')
import img_utils as util
from common_flags import FLAGS
from misc_utils import del_and_recreate_folder, load_settings_or_use_args

img_name = '/data/test.jpg'
out_size = (640, 480)


# main function
def _main():
    # Load settings or use args or defaults
    s = load_settings_or_use_args(FLAGS)
    img_mode = s['img_mode']

    all_imgs = []

    # load image
    img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    img = cv2.resize(img, out_size, interpolation=cv2.INTER_AREA)
    img_s = cv2.resize(img.copy(), (s['img_width'], s['img_height']), interpolation=cv2.INTER_AREA)
    #img_s = util.central_image_crop(img_s, s['crop_img_height'], s['crop_img_width'])

    # prep array for input
    img_out = np.asarray(img_s, dtype=np.float32) * np.float32(1.0 / 255.0)
    if img_mode == "rgb":
        carry = np.array(img_out)[np.newaxis, ...].astype(np.float32)
    else:
        img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
        carry = np.array(img_out)[np.newaxis, :, :, np.newaxis]

    print(carry.shape)

    # Load json and create model
    json_model_path = os.path.join(FLAGS.experiment_rootdir, s['model_dir'], s['json_model_fname'])

    # Get weights paths
    weights_2_load = sorted(glob.glob(os.path.join(FLAGS.experiment_rootdir, s['model_dir'], 'model_weights*')), key=os.path.getmtime)

    # prep output directory
    outfolder = os.path.join(FLAGS.experiment_rootdir, 'vis_epochs')
    del_and_recreate_folder(outfolder)

    # iterate through trained models
    for i, weights_load_path in enumerate(weights_2_load):
        k.clear_session()
        model = get_model(json_model_path, weights_load_path)
        penultimate_layer = vutils.find_layer_idx(model, 'conv2d_9')

        img_out, dur = get_grad_cam(img, carry, model, penultimate_layer)

        # make and store predictions
        theta, p_t, img_out = make_prediction(img_out, carry, model)

        # ouput of results
        stats = ["Epoch: {:d}".format(i), "Predictions:", "[C: {:4.3f}] [SA: {:4.3f}]".format(float(p_t), float(theta))]

        # place output on image
        for idx, stat in enumerate(stats):
            text = stat.lstrip()
            cv2.putText(img_out, text, (0, 30 + (idx * 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2, lineType=30)

        # store img
        all_imgs.append(img_out.astype(np.uint8))

        # show output
        cv2.imshow('frame', img_out)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # prep output video
    out_name = os.path.splitext(os.path.split(img_name)[1])[0] + ".mp4"
    output_name = os.path.join(outfolder, out_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_name, fourcc, 1, (out_size[0] + 20, out_size[1]))

    # write video
    for img in all_imgs:
        video.write(img)
    video.release()


def get_model(json_model_path, weights_path):
    # Load json and weights, then compile model
    with open(json_model_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)
    model.compile(loss='mse', optimizer='adadelta')

    # prep for grad-cam
    layer_idx = vutils.find_layer_idx(model, 'dense_1')

    # grad-cam: swap softmax with linear
    model.layers[layer_idx].activation = activations.linear
    model = vutils.apply_modifications(model)

    return model


def make_prediction(img, carry, model):
    # inference
    outs = model.predict(carry, batch_size=1, verbose=0, steps=None)

    # store predictions
    theta, p_t = outs[0][0], outs[1][0]

    # display predictions as bars on the sides
    predictbar = np.zeros((out_size[1], 20, 3), dtype=np.uint8)
    index = int(float(p_t) * out_size[1])
    predictbar[0:index, :, 2] = np.ones_like(predictbar[0:index, :, 2], dtype=np.uint8) * 255
    predictbar[index:, :, 1] = np.ones_like(predictbar[index:, :, 1], dtype=np.uint8) * 255

    return theta, p_t, np.hstack((img.copy(), predictbar))


def get_grad_cam(img, carry, model, penultimate_layer):
    ts = time.time()

    # get activation map
    grads = visualize_cam(model, penultimate_layer,
                          filter_indices=1,
                          seed_input=carry,
                          backprop_modifier=None,
                          grad_modifier=None)

    # time visualisation to prevent slowdown
    duration = time.time() - ts

    # Lets overlay the heatmap onto original image.
    jhm = cv2.cvtColor(grads, cv2.COLOR_BGR2RGB)
    jhm = cv2.resize(jhm, out_size, interpolation=cv2.INTER_AREA)

    return overlay(img.copy(), jhm), duration


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
