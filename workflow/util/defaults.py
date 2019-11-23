# default options

_DEFAULT_ARGS = {
    # names
    'name': "same",
    'settings_fname': "settings.json",
    'weights_fname': "model_weights.h5",
    'json_model_fname': "model_struct.json",

    # folders
    'model_dir': "model",
    'train_dir': "train",
    'val_dir': "val",
    'test_dir': "test",
    'eval_dir': "eval",

    # input options
    'img_mode': "rgb",
    'img_width': 448,
    'img_height': 256,
    'crop_img_width': 448,
    'crop_img_height': 256,

    # training options
    'batch_size': 128,
    'epochs': 50,
    'log_rate': 1,
    'restore_model': False,
    'initial_epoch': 1,

    # testing options
    'show_activations': True,
}

_DEFAULT_SETTINGS = {
    ### STEP 1 ###
    # preprocessing options
    'gaussian_blur_min': 3,
    'gaussian_blur_max': 15,
    'median_blur_min': 5,
    'median_blur_max': 11,
    'saltnpepper_amount': 0.02,
    'cloud_file': "/../misc/images/cloud5k.png",
    'cloud_quota': 0.4,
    'cloud_max': 0.5,
    'channel_randomness': 0.1,
    'center_flow': False,
    'scale_flow_to_mag1': False,

    # data validation options
    'zero_quota': {'steering': 0.4,
                   'collision': 1.0},
    'ignore_last_images': 3,
    'ignore_first_images': 2,
    'ignore_before_coll': 2,
    'min_data_per_exp': 10,
    'min_real_data': 5,
    'max_coll_in_steer_data': 10,
    'dark_limit': 30,
    'bright_limit': 800,

    # other options
    'scaler_filename': "steering_scaler.dat",
    'train_val_qouta': 0.1,
    'max_steer_angle': 5,
    'flow_limit': 1,
    'flow_stat_bin_count': 200,

    ### STEP 2 ###
    # augmentation settings
    'rotation_range': 0.2,
    'rescale_factor': 1. / 255,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,

    # loss rates
    'alpha_loss_weight': 1.0,
    'beta_loss_weight': 1.0,

    ### STEP 5 ###
    # grad cam settings
    'softmax_layer': "dense_2",                 # softmax layer to be swapped
    'penultimate_layer': "conv2d_9",            # layer before output
    'filter_indices': 1                         # 1 for collision 0 for steering
}