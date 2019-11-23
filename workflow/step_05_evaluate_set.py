#!/usr/bin/env python3
"""
step_05_evaluate_set.py

Fifth step of workflow: advanced evaluation of a trained model with activation maps and output of results.

Written by Moritz Sperling
Based on the work of A. Loquercio, et al., 2018 (https://github.com/uzh-rpg/rpg_public_dronet)

Licensed under the MIT License (see LICENSE for details)
"""
import os
import cv2
import csv
import sys
import glob
import json
import time
import numpy as np
from tqdm import tqdm
import absl.flags as gflags
import util.img_utils as iu
import util.misc_utils as misc
from keras import activations
from keras import backend as k
from keras.models import model_from_json
from util.common_flags import FLAGS
from vis.utils import utils
from vis.visualization import visualize_cam, overlay


show_output = True
txtcolor = (0, 0, 255)
label_files = {'collision': "labels.txt",
               'steering': "sync_steering.txt"}


# main function
def _main():
    global s

    # Load settings or use args or defaults
    s = misc.load_settings_or_use_args(FLAGS)

    # Init variables (no need to change anything here)
    last_duration = 1
    json_model_path = os.path.join(FLAGS.experiment_rootdir, s['model_dir'], s['json_model_fname'])
    weights_path = os.path.join(FLAGS.experiment_rootdir, s['model_dir'], s['weights_fname'])
    s['target_size'] = s.get('target_size', (s['img_width'], s['img_height']))
    s['crop_size'] = s.get('crop_size', (s['crop_img_width'], s['crop_img_height']))
    s['output_size'] = s.get('output_size', (400, 300))

    extensions = ['jpg', 'png']

    # Special settings for flow mode
    if s['img_mode'] == "flow":
        extensions = ['npy', 'bin', 'flo', 'pfm', 'png']

    # Load model
    model = get_model(json_model_path, weights_path)

    # Penultimate layer index for grad cams
    penultimate_layer = utils.find_layer_idx(model, s['penultimate_layer'])

    # Check if single or multiple experiments to test
    folders = misc.get_experiment_folders(FLAGS.test_dir)

    # Prepare output directory
    if s['name'] == "same":
        if FLAGS.test_dir[-1] == "/":
            outdir = "eval_" + os.path.split(s['test_dir'][:-1])[1] + "_adv"
        else:
            outdir = "eval_" + os.path.split(s['test_dir'])[1] + "_adv"
    else:
        outdir = "eval_" + s['name'] + "_" + os.path.split(s['test_dir'])[1] + "_adv"

    if FLAGS.output_dir is None:
        outfolder = os.path.join(FLAGS.experiment_rootdir, s['eval_dir'], outdir)
    else:
        outfolder = os.path.join(FLAGS.output_dir, outdir)
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    # Store settings as backup
    settings_out_filename = os.path.join(outfolder, s['settings_fname'])
    misc.write_to_file(s, settings_out_filename, beautify=True)

    # Iterate through folders
    for folder in tqdm(sorted(folders), desc="Evaluating Dataset: ", maxinterval=1, unit="Exp."):

        # Get filenames
        imgs = []
        for extension in extensions:
            ipth = os.path.join(folder, 'images/*.' + extension)
            imgs = sorted(glob.glob(ipth))
            if len(imgs) > 0:
                break

        # Check if labels are available, then load them
        exp_type = None
        lbl = np.zeros(len(imgs)) - 1337
        for key in label_files:
            fullname = os.path.join(folder, label_files[key])

            if os.path.isfile(fullname):
                lbl = []
                exp_type = key
                with open(fullname, newline='') as i:
                    for row in csv.reader(i, delimiter=';'):
                        lbl.append(float(row[0]))
                break

        # Prepare output subfolder
        foldersub = os.path.join(outfolder, os.path.split(folder)[1])
        misc.del_and_recreate_folder(os.path.join(foldersub, "images"))

        # # add first label on flow data (otherwise it will have one label less than the real data)
        # if "flow" in folder:
        #     lbl.insert(0, 0)

        # Check if labels and images fit
        if len(lbl) >= len(imgs):
            col = []
            ste = []

            # Iterate through imgs with n stepsize
            for x in range(len(imgs)):

                # Load image and prep for dronet
                data = iu.load_img(imgs[x], img_mode=s['img_mode'],
                                   crop_size=s['crop_size'],
                                   target_size=s['target_size'])

                # Load image again for displaying
                if s['img_mode'] == "flow":
                    # Convert for output
                    img = iu.flow_to_img(data)
                else:
                    # Read img and rescale data
                    img = cv2.imread(imgs[x], cv2.IMREAD_COLOR)
                    data = np.asarray(data, dtype=np.float32) * np.float32(1.0 / 255.0)

                carry = data[np.newaxis, ...]

                # Make and store predictions
                img = cv2.resize(img, s['output_size'], interpolation=cv2.INTER_AREA)
                theta, p_t, img_out = make_prediction(img, carry, model, lbl[x])
                img_out = display_predictions_as_bars(img_out, theta, p_t, lbl[x], exp_type)

                col.append(float(p_t))
                ste.append(float(theta))

                # Grad-CAMs
                if s['show_activations']:
                    # Cleanup for keras-vis bug (slowdown + leaky memory)
                    if last_duration > 0.4:
                        k.clear_session()
                        model = get_model(json_model_path, weights_path)

                    # Produce activation maps for collision or steering
                    img_h, last_duration = get_grad_cam(img, carry, model, penultimate_layer, s['filter_indices'])

                    # Put the pieces together
                    img_out = np.hstack((img_out, np.ones((s['output_size'][1], 50, 3), dtype=np.uint8) * 255, img_h))

                # Write output
                img_name = os.path.split(imgs[x])[1]
                img_name = os.path.splitext(img_name)[0]
                cv2.imwrite(os.path.join(foldersub, "images", img_name + ".png"), img_out)

                # Show output
                if show_output:
                    cv2.imshow('frame', img_out)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cv2.destroyAllWindows()

            # Write predicted probabilities and real labels
            dict_data = {'pred_collisions': col,
                         'pred_steerings': ste,
                         'real_labels': lbl}
            jsonfilen = os.path.join(foldersub, 'predicted_and_real_labels_'
                                     + os.path.basename(s['weights_fname']) + '.json')

            with open(jsonfilen, "w") as f:
                json.dump(dict_data, f)
                print("Written file {}".format(jsonfilen))
            f.close()

        else:
            print("Something's wrong ...")
            exit(1)


def get_model(json_model_path, weights_path):
    # Load json and weights, then compile model
    with open(json_model_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)
    model.compile(loss='mse', optimizer='adadelta')

    # Prep for grad-cam
    layer_idx = utils.find_layer_idx(model, s['softmax_layer'])

    # Grad-CAM: swap softmax with linear
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

    return model


def make_prediction(img, x, model, lbl):
    # Inference
    outs = model.predict(x, batch_size=1, verbose=0, steps=None)

    # Store predictions
    theta, p_t = outs[0][0], outs[1][0]

    # Output of results
    stats = ["Predictions:", "[C: {:4.3f}] [SA: {:4.3f}]".format(float(p_t), float(theta))]
    if lbl > -1337:
        stats.append("Label: [{:4.3f}]".format(lbl))

    # Place output on image
    img_out = img.copy()
    for idx, stat in enumerate(stats):
        text = stat.lstrip()
        cv2.putText(img_out, text, (5, 30 + (idx * 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, txtcolor, 1, lineType=30)
    return theta, p_t, img_out


def display_predictions_as_bars(img, theta, p_t, lbl, exp_type):
    # Display label and prediction on the side
    if exp_type == "steering":
        labelbar = iu.pred_as_indicator(lbl, (20, s['output_size'][1]), "Ground Truth", mode="vertical")
        predictbar = iu.pred_as_indicator(theta, (20, s['output_size'][1]), "Prediction", mode="vertical")
    elif exp_type == "collision":
        labelbar = iu.pred_as_bar(lbl, (s['output_size'][1], 20), "Ground Truth")
        predictbar = iu.pred_as_bar(p_t, (s['output_size'][1], 20), "Prediction")
    else:
        labelbar = np.ones((s['output_size'][1], 20, 3), dtype=np.uint8) * 255
        predictbar = np.ones((s['output_size'][1], 20, 3), dtype=np.uint8) * 255

    spacer = np.ones((s['output_size'][1], 5, 3), dtype=np.uint8) * 255

    return np.hstack((img, spacer, labelbar, spacer, predictbar))


def get_grad_cam(img, carry, model, penultimate_layer, indices):
    ts = time.time()

    # Get activation map
    grads = visualize_cam(model, penultimate_layer,
                          filter_indices=indices,
                          seed_input=carry,
                          backprop_modifier=None,
                          grad_modifier=None)

    # Time visualisation to prevent slowdown
    duration = time.time() - ts

    # Lets overlay the heatmap onto original image.
    jhm = cv2.cvtColor(grads, cv2.COLOR_BGR2RGB)
    jhm = cv2.resize(jhm, s['output_size'], interpolation=cv2.INTER_AREA)

    return overlay(img.copy(), jhm), duration


def main(argv):
    # Utility main to load flags
    try:
        FLAGS(argv)  # parse flags
        assert (FLAGS.experiment_rootdir is not None), "Please provide experiment root directory."
        assert (FLAGS.test_dir is not None), "Please provide directory for testing."
        assert (FLAGS.weights_fname is not None), "Please provide the model weights filename to be used."
    except gflags.Error:
        print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
        sys.exit(1)
    _main()


if __name__ == "__main__":
    main(sys.argv)
