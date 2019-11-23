#!/usr/bin/env python3
"""
step_02_train.py

Second step of workflow: Train FlowDroNet from scratch.

Written by Moritz Sperling
Based on the work of A. Loquercio et al., 2018 (https://github.com/uzh-rpg/rpg_public_dronet)

Licensed under the MIT License (see LICENSE for details)
"""
import os
import sys
import numpy as np
import tensorflow as tf
import absl.flags as gflags
from keras import optimizers
import util.dronet_utils as utils
from util.common_flags import FLAGS
import util.dronet_model as cnn_model
import util.misc_utils as misc


def _main():

    # Load settings or use args or defaults
    s = misc.load_settings_or_use_args(FLAGS)

    # Create the experiment rootdir if not already there
    model_dir = os.path.join(FLAGS.experiment_rootdir, s['model_dir'])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Image mode
    if s['img_mode'] == 'rgb':
        s['img_channels'] = s.get('img_channels', 3)
    elif s['img_mode'] == 'flow':
        s['img_channels'] = s.get('img_channels', 2)
        s['rescale_factor'] = None
        s['width_shift_range'] = None
        s['height_shift_range'] = None
    elif s['img_mode'] == 'grayscale':
        s['img_channels'] = s.get('img_channels', 1)
    else:
        raise IOError("Unidentified image mode: use 'grayscale', 'flow' or 'rgb'")

    # Generate training data with real-time augmentation
    train_datagen = utils.DroneDataGenerator(rotation_range=s['rotation_range'],
                                             rescale=s['rescale_factor'],
                                             width_shift_range=s['width_shift_range'],
                                             height_shift_range=s['height_shift_range'],
                                             vertical_flip=False)

    train_generator = train_datagen.flow_from_directory(os.path.join(FLAGS.experiment_rootdir, s['train_dir']),
                                                        shuffle=True,
                                                        color_mode=s['img_mode'],
                                                        target_size=(s['img_width'], s['img_height']),
                                                        crop_size=(s['crop_img_width'], s['crop_img_height']),
                                                        batch_size=s['batch_size'])

    # Generate validation data with real-time augmentation
    val_datagen = utils.DroneDataGenerator(rescale=s['rescale_factor'])

    val_generator = val_datagen.flow_from_directory(os.path.join(FLAGS.experiment_rootdir, s['val_dir']),
                                                    shuffle=False,
                                                    color_mode=s['img_mode'],
                                                    target_size=(s['img_width'], s['img_height']),
                                                    crop_size=(s['crop_img_width'], s['crop_img_height']),
                                                    batch_size=s['batch_size'])

    # Weights to restore
    if not s['restore_model']:
        # In this case weights will start from random
        s['initial_epoch'] = 0
        s['restore_path'] = None
    else:
        # In this case weigths will start from the specified mode
        s['restore_path'] = os.path.join(model_dir, s['weights_fname'])

    # Define model
    model = get_model(s['crop_img_width'], s['crop_img_height'], s['img_channels'], 1, s['restore_path'])

    # Serialize model into json
    json_model_path = os.path.join(model_dir, s['json_model_fname'])
    utils.model_to_json(model, json_model_path)

    # Store settings
    settings_out_filename = os.path.join(FLAGS.experiment_rootdir, s['settings_fname'])
    misc.write_to_file(s, settings_out_filename, beautify=True)

    # Train model
    train_model(train_generator, val_generator, model, s['initial_epoch'], s)


def get_model(img_width, img_height, img_channels, output_dim, weights_path):
    """
    Initialize model.

    # Arguments
       img_width: Target image width.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.
       weights_path: Path to pre-trained model.

    # Returns
       model: A Model instance.
    """
    model = cnn_model.resnet8(img_width, img_height, img_channels, output_dim)

    if weights_path:
        try:
            model.load_weights(weights_path)
            print("Loaded model from {}".format(weights_path))
        except IOError:
            print("Impossible to find weight path. Returning untrained model")

    return model


def train_model(train_data_generator, val_data_generator, model, initial_epoch, s):
    """
    Model training.

    # Arguments
       train_data_generator: Training data generated batch by batch.
       val_data_generator: Validation data generated batch by batch.
       model: Target image channels.
       initial_epoch: Dimension of model output.
    """

    # Initialize loss weights
    model.alpha = tf.Variable(s['alpha_loss_weight'], trainable=False, name='alpha', dtype=tf.float32)
    model.beta = tf.Variable(s['beta_loss_weight'], trainable=False, name='beta', dtype=tf.float32)

    # Initialize number of samples for hard-mining
    model.k_mse = tf.Variable(s['batch_size'], trainable=False, name='k_mse', dtype=tf.int32)
    model.k_entropy = tf.Variable(s['batch_size'], trainable=False, name='k_entropy', dtype=tf.int32)

    # Set optimizer to adadelta (worked better over different dataset sizes than adam)
    optimizer = optimizers.Adadelta()

    # Configure training process
    model.compile(loss=[utils.hard_mining_mse(model.k_mse), utils.hard_mining_entropy(model.k_entropy)],
                  optimizer=optimizer,
                  loss_weights=[model.alpha, model.beta])

    # Save training and validation losses.
    save_model_and_loss = utils.MyCallback(filepath=os.path.join(FLAGS.experiment_rootdir, s['model_dir']),
                                           logpath=FLAGS.experiment_rootdir,
                                           period=s['log_rate'],
                                           batch_size=s['batch_size'])

    # Train model
    steps_per_epoch = int(np.ceil(train_data_generator.samples / s['batch_size']))
    validation_steps = int(np.ceil(val_data_generator.samples / s['batch_size']))

    model.fit_generator(train_data_generator,
                        epochs=s['epochs'], steps_per_epoch=steps_per_epoch,
                        callbacks=[save_model_and_loss],
                        validation_data=val_data_generator,
                        validation_steps=validation_steps,
                        initial_epoch=initial_epoch)


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
