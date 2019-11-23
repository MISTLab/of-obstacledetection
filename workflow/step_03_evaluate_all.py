#!/usr/bin/env python3
"""
step_03_evaluate_all.py

Third step of workflow: Evaluate a trained model on a FlowDroNet-compatible dataset.

Written by Moritz Sperling
Based on the work of A. Loquercio et al., 2018 (https://github.com/uzh-rpg/rpg_public_dronet)

Licensed under the MIT License (see LICENSE for details)
"""
import os
import sys
import glob
import itertools
import numpy as np
from random import randint
import absl.flags as gflags
from sklearn import metrics
from keras import backend as k
import util.misc_utils as misc
import util.dronet_utils as utils
from util.common_flags import FLAGS


# main function
def _main():

    # Load settings or use args or defaults
    s = misc.load_settings_or_use_args(FLAGS)

    # Misc settings
    s['rescale_factor'] = s.get('rescale_factor', 1. if s['img_mode'] == "flow" else 1. / 255)
    if FLAGS.test_dir is None:
        s['test_dir'] = os.path.join(FLAGS.experiment_rootdir, s['test_dir'])
    model_dir = os.path.join(FLAGS.experiment_rootdir, s['model_dir'])

    # Set testing mode (dropout/batchnormalization)
    k.set_learning_phase(0)

    # Generate testing data
    test_datagen = utils.DroneDataGenerator(rescale=s['rescale_factor'])
    test_generator = test_datagen.flow_from_directory(s['test_dir'],
                                                      shuffle=False,
                                                      color_mode=s['img_mode'],
                                                      target_size=(s['img_width'], s['img_height']),
                                                      crop_size=(s['crop_img_width'], s['crop_img_height']),
                                                      batch_size=1,
                                                      is_training=False)

    # Load json and create model
    json_model_path = os.path.join(model_dir, s['json_model_fname'])
    model = utils.json_to_model(json_model_path)

    # Get weights paths
    weights_2_load = sorted(glob.glob(os.path.join(model_dir, 'model_weights*')), key=os.path.getmtime)

    # Prepare output directory
    if s['name'] == "same":
        if FLAGS.test_dir[-1] == "/":
            outdir = "eval_" + os.path.split(s['test_dir'][:-1])[1]
        else:
            outdir = "eval_" + os.path.split(s['test_dir'])[1]
    else:
        outdir = "eval_" + s['name'] + "_" + os.path.split(s['test_dir'])[1]

    if FLAGS.output_dir is None:
        outfolder = os.path.join(FLAGS.experiment_rootdir, s['eval_dir'], outdir)
    else:
        outfolder = os.path.join(FLAGS.output_dir, outdir)
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    # Store settings as backup
    settings_out_filename = os.path.join(outfolder, s['settings_fname'])
    misc.write_to_file(s, settings_out_filename, beautify=True)

    # Iterate through trained models
    for weights_load_path in weights_2_load:
        model.load_weights(weights_load_path)

        print("\nEvaluating: " + weights_load_path)
        print("On: " +  s['test_dir'] + "\n")
        index = weights_load_path.split("_")[-1][:-3]

        # Compile model
        model.compile(loss='mse', optimizer='adadelta')

        # Get predictions and ground truth
        n_samples = test_generator.samples
        nb_batches = int(np.ceil(n_samples / 1))

        predictions, ground_truth, t, fnames = utils.compute_predictions_and_gt(
            model, test_generator, nb_batches, verbose=1)

        # Param t. t=1 steering, t=0 collision
        t_mask = t == 1

        # ************************* Steering evaluation ***************************

        # Predicted and real steerings
        pred_steerings = predictions[t_mask, 0]
        real_steerings = ground_truth[t_mask, 0]

        # Compute random and constant baselines for steerings
        random_steerings = random_regression_baseline(real_steerings)
        constant_steerings = constant_baseline(real_steerings)

        # Create dictionary with filenames
        dict_fname = {'test_regression_' + str(index.zfill(4)) + '.json': pred_steerings,
                      'random_regression_' + str(index.zfill(4)) + '.json': random_steerings,
                      'constant_regression_' + str(index.zfill(4)) + '.json': constant_steerings}

        # Evaluate predictions: EVA, residuals, and highest errors
        for fname, pred in dict_fname.items():
            abs_fname = os.path.join(outfolder, fname)
            evaluate_regression(pred, real_steerings, abs_fname)

        # Write predicted and real steerings
        dict_test = {'pred_steerings': pred_steerings.tolist(),
                     'real_steerings': real_steerings.tolist()}
        pred_fname = os.path.join(outfolder, 'predicted_and_real_steerings_' + str(index.zfill(4)) + '.json')
        misc.write_to_file(dict_test, pred_fname)

        # *********************** Collision evaluation ****************************

        # Predicted probabilities and real labels
        pred_prob = predictions[~t_mask, 1]
        pred_labels = np.zeros_like(pred_prob)
        pred_labels[pred_prob >= 0.5] = 1

        real_labels = ground_truth[~t_mask, 1]

        # Compute random, weighted and majorirty-class baselines for collision
        random_labels = random_classification_baseline(real_labels)

        # Create dictionary with filenames
        dict_fname = {'test_classification_' + str(index.zfill(4)) + '.json': pred_labels,
                      'random_classification_' + str(index.zfill(4)) + '.json': random_labels}

        # Evaluate predictions: accuracy, precision, recall, F1-score, and highest errors
        for fname, pred in dict_fname.items():
            abs_fname = os.path.join(outfolder, fname)
            evaluate_classification(pred_prob, pred, real_labels, abs_fname)

        chain = itertools.chain(*fnames)
        ffnames = list(chain)
        # Write predicted probabilities and real labels
        dict_test = {'pred_probabilities': pred_prob.tolist(),
                     'real_labels': real_labels.tolist(),
                     'filenames': ffnames}
        pred_fname = os.path.join(outfolder, 'predicted_and_real_labels_' + str(index.zfill(4)) + '.json')
        misc.write_to_file(dict_test, pred_fname)


# Functions to evaluate steering prediction
def explained_variance_1d(ypred, y):
    """
    Var[ypred - y] / var[y].
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


def compute_explained_variance(predictions, real_values):
    """
    Computes the explained variance of prediction for each
    steering and the average of them
    """
    assert np.all(predictions.shape == real_values.shape)
    ex_variance = explained_variance_1d(predictions, real_values)
    print("EVA = {}".format(ex_variance))
    return ex_variance


def compute_sq_residuals(predictions, real_values):
    assert np.all(predictions.shape == real_values.shape)
    sq_res = np.square(predictions - real_values)
    sr = np.mean(sq_res, axis=-1)
    print("MSE = {}".format(sr))
    return sq_res


def compute_rmse(predictions, real_values):
    assert np.all(predictions.shape == real_values.shape)
    mse = np.mean(np.square(predictions - real_values))
    rmse = np.sqrt(mse)
    print("RMSE = {}".format(rmse))
    return rmse


def compute_highest_regression_errors(predictions, real_values, n_errors=20):
    """
    Compute the indexes with highest error
    """
    assert np.all(predictions.shape == real_values.shape)
    sq_res = np.square(predictions - real_values)
    highest_errors = sq_res.argsort()[-n_errors:][::-1]
    return highest_errors


def random_regression_baseline(real_values):
    mean = np.mean(real_values)
    std = np.std(real_values)
    return np.random.normal(loc=mean, scale=abs(std), size=real_values.shape)


def constant_baseline(real_values):
    mean = np.mean(real_values)
    return mean * np.ones_like(real_values)


def evaluate_regression(predictions, real_values, fname):
    evas = compute_explained_variance(predictions, real_values)
    rmse = compute_rmse(predictions, real_values)
    highest_errors = compute_highest_regression_errors(predictions, real_values,
                                                       n_errors=20)
    dictionary = {"evas": evas.tolist(), "rmse": rmse.tolist(),
                  "highest_errors": highest_errors.tolist()}
    misc.write_to_file(dictionary, fname)


# Functions to evaluate collision

def read_training_labels(file_name):
    labels = []
    try:
        labels = np.loadtxt(file_name, usecols=0)
        labels = np.array(labels)
    except IOError:
        print("File {} failed loading labels".format(file_name))
    return labels


def count_samples_per_class(train_dir):
    experiments = glob.glob(train_dir + "/*")
    num_class0 = 0
    num_class1 = 0
    for exp in experiments:
        file_name = os.path.join(exp, "labels.txt")
        try:
            labels = np.loadtxt(file_name, usecols=0)
            num_class1 += np.sum(labels == 1)
            num_class0 += np.sum(labels == 0)
        except IOError:
            print("File {} failed loading labels".format(file_name))
            continue
    return np.array([num_class0, num_class1])


def random_classification_baseline(real_values):
    """
    Randomly assigns half of the labels to class 0, and the other half to class 1
    """
    return [randint(0, 1) for _ in range(real_values.shape[0])]


def weighted_baseline(real_values, samples_per_class):
    """
    Let x be the fraction of instances labeled as 0, and (1-x) the fraction of
    instances labeled as 1, a weighted classifier randomly assigns x% of the
    labels to class 0, and the remaining (1-x)% to class 1.
    """
    weights = samples_per_class / np.sum(samples_per_class)
    return np.random.choice(2, real_values.shape[0], p=weights)


def majority_class_baseline(real_values, samples_per_class):
    """
    Classify all test data as the most common label
    """
    major_class = np.argmax(samples_per_class)
    return [major_class for _ in real_values]


def compute_highest_classification_errors(predictions, real_values, n_errors=20):
    """
    Compute the indexes with highest error
    """
    assert np.all(predictions.shape == real_values.shape)
    dist = abs(predictions - real_values)
    highest_errors = dist.argsort()[-n_errors:][::-1]
    return highest_errors


def evaluate_classification(pred_prob, pred_labels, real_labels, fname):
    ave_accuracy = metrics.accuracy_score(real_labels, pred_labels)
    print('Average accuracy = ', ave_accuracy)
    precision = metrics.precision_score(real_labels, pred_labels)
    print('Precision = ', precision)
    recall = metrics.recall_score(real_labels, pred_labels)
    print('Recall = ', recall)
    f_score = metrics.f1_score(real_labels, pred_labels)
    print('F1-score = ', f_score)
    highest_errors = compute_highest_classification_errors(pred_prob, real_labels,
                                                           n_errors=20)
    dictionary = {"ave_accuracy": ave_accuracy, "precision": precision,
                  "recall": recall, "f_score": f_score,
                  "highest_errors": highest_errors.tolist()}
    misc.write_to_file(dictionary, fname)


def main(argv):
    # Utility main to load flags
    try:
        FLAGS(argv)  # parse flags
        assert (FLAGS.experiment_rootdir is not None), "No experiment root directory given."
    except gflags.Error:
        print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
        sys.exit(1)
    _main()


if __name__ == "__main__":
    main(sys.argv)
