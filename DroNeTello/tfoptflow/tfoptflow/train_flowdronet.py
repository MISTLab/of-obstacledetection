"""
pwcnet_train.ipynb

PWC-Net model training.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)

Tensorboard:
    [win] tensorboard --logdir=E:\\repos\\tf-optflow\\tfoptflow\\pwcnet-sm-6-2-cyclic-chairsthingsmix-fp16
    [ubu] tensorboard --logdir=/media/EDrive/repos/tf-optflow/tfoptflow/pwcnet-sm-6-2-cyclic-chairsthingsmix-fp16
"""
import sys
from copy import deepcopy
from model_flowdronet import _DEFAULT_FLOWDRONET_OPTS
from dataset_flowdronet import FlowDroNetDataset
from model_pwcnet import ModelPWCNet

# TODO: You MUST set paths to the correct path on your machine!
if sys.platform.startswith("linux"):
    _MODEL_PATH = '/home/recherche/dev-msperling/misc/tfoptflow/tfoptflow/models/' \
                + 'pwcnet-sm-6-2-cyclic-chairsthingsmix/pwcnet.ckpt-49000'
    _DATASET_ROOT = '/home/recherche/data/collision_new'
    _TRAIN_IMGS = 'all_real_newres'
    _TRAIN_FLOW = 'all_liteflownet_newres'
    _TEST_IMGS = 'all_real_newres'
else:
    _MODEL_PATH = '/Users/nfinite/Library/Mobile Documents/com~apple~CloudDocs/MISTLab/Dev/misc/tfoptflow/tfoptflow/models/' \
                + 'pwcnet-sm-6-2-cyclic-chairsthingsmix/pwcnet.ckpt-49000'
    _DATASET_ROOT = '/Users/nfinite/data/test'
    _TRAIN_IMGS = 'treal'
    _TRAIN_FLOW = 'traw2'
    _TEST_IMGS = 'treal'

print(_MODEL_PATH)
# Set controller device and devices
# A one-gpu setup would be something like controller='/device:GPU:0' and gpu_devices=['/device:GPU:0']
gpu_devices = ['/device:GPU:0']
controller = '/device:GPU:0'

# Batch size
batch_size = 4

# Dataset options
ds_opts = {
    'batch_size': batch_size,
    'verbose': True,
    'in_memory': False,  # True loads all samples upfront, False loads them on-demand
    'crop_preproc': (256, 448),  # None or (h, w), use (384, 768) for FlyingThings3D
    'scale_preproc': None,  # None or (h, w),
    # ['clean' | 'final'] for MPISintel, ['noc' | 'occ'] for KITTI, 'into_future' for FlyingThings3D
    'type': 'into_future',
    'tb_test_imgs': False,  # If True, make test images available to model in training mode
    # Sampling and split options
    'random_seed': 1337,  # random seed used for sampling
    'val_split': 0.04,  # portion of data reserved for the validation split
    # Augmentation options
    'aug_type': 'basic',  # in [None, 'basic', 'heavy'] to add augmented data to training set
    'aug_labels': True,  # If True, augment both images and labels; otherwise, only augment images
    'fliplr': 0.5,  # Horizontally flip 50% of images
    'flipud': 0.1,  # Vertically flip 50% of images
    # Translate 50% of images by a value between -5 and +5 percent of original size on x- and y-axis independently
    'translate': 0.5,
    'scale': 0.5,  # Scale 50% of images by a factor between 95 and 105 percent of original size
    'train_imgs': _TRAIN_IMGS,
    'train_flow': _TRAIN_FLOW,
    'test_imgs': _TEST_IMGS,
}

# Load train dataset
ds = FlowDroNetDataset(mode='train_with_val', ds_root=_DATASET_ROOT, options=ds_opts)

# Display dataset configuration
ds.print_config()

# Training options
nn_opts = deepcopy(_DEFAULT_FLOWDRONET_OPTS)
nn_opts['batch_size'] = batch_size
nn_opts['controller'] = controller
nn_opts['gpu_devices'] = gpu_devices
nn_opts['ckpt_path'] = _MODEL_PATH

# Instantiate the model and display the model configuration
nn = ModelPWCNet(mode='train_with_val', options=nn_opts, dataset=ds)
nn.print_config()

# Train the model
nn.train()
