#!/usr/bin/env python3
"""
common_flags.py

Handler for the common arguments of the workflow.

Based on the work of A. Loquercio et al., 2018 (https://github.com/uzh-rpg/rpg_public_dronet)

Licensed under the MIT License (see LICENSE for details)
"""

import absl.flags as gflags

FLAGS = gflags.FLAGS

# Input
gflags.DEFINE_string('settings_fname', 	    None,   'Settings filename')
gflags.DEFINE_integer('img_width', 			None,   'Target Image Width')
gflags.DEFINE_integer('img_height', 		None,   'Target Image Height')
gflags.DEFINE_integer('crop_img_width', 	None,   'Cropped image widht')
gflags.DEFINE_integer('crop_img_height', 	None,   'Cropped image height')
gflags.DEFINE_string('img_mode', 			None,   'Load mode for images, either rgb, flow, flow_as_rgb, flow_as_mag or grayscale')
gflags.DEFINE_string('name', 				None,   'Name your experiment')

# Training
gflags.DEFINE_integer('batch_size', 		None,   'Batch size in training and evaluation')
gflags.DEFINE_integer('epochs', 			None,   'Number of epochs for training')
gflags.DEFINE_integer('log_rate', 			None,   'Logging rate for full model (epochs)')
gflags.DEFINE_integer('initial_epoch', 		None,   'Initial epoch to start training')
gflags.DEFINE_bool('restore_model', 		None, 	'Whether to restore a trained model for training')

# Testing
gflags.DEFINE_bool('show_activations',      None,   'Enable grad-cams via keras-vis (advanced evaluation only)')

# Folders
gflags.DEFINE_string('experiment_rootdir', 	None,   'Main Folder containing all the subfolders, logs and settings')
gflags.DEFINE_string('model_dir', 	        None,   'Sub-Folder containing the models weights and definition')
gflags.DEFINE_string('train_dir', 			None,   'Sub-Folder containing training experiments')
gflags.DEFINE_string('val_dir', 			None,   'Sub-Folder containing validation experiments')
gflags.DEFINE_string('test_dir', 			None,   'Sub-Folder containing testing experiments')
gflags.DEFINE_string('eval_dir', 			None,   'Sub-Folder containing evaluation results')
gflags.DEFINE_string('input_dir', 			None,   'Folder containing the input of script (step 1)')
gflags.DEFINE_string('output_dir', 			None,   'Folder for the output of script')

# Model
gflags.DEFINE_string('weights_fname', 		None, 	'(Relative) filename of model weights')
gflags.DEFINE_string('json_model_fname', 	None,	'Model struct json serialization, filename')

