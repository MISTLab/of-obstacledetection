"""
pwcnet_predict_from_img_pairs.py

Run inference on a list of images pairs.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)
"""
from __future__ import absolute_import, division, print_function

import os
import cv2
import glob
import time
from copy import deepcopy
from model_flowdronet import ModelFlowDroNet, _DEFAULT_FLOWDRONET_OPTS
from model_pwcnet import _DEFAULT_PWCNET_TEST_OPTIONS, _DEFAULT_PWCNET_VAL_OPTIONS
from optflow import flow_to_img


imsize = (256, 448)
outres = (640, 480)


# Paths
data = '/Users/nfinite/data/collision_new/all_real/'
dronet_model_path = '/Users/nfinite/data/test/model_flow2/model_graph_w27.pb'


"""
#test small model
ckpt_path = '/Users/nfinite/Library/Mobile Documents/com~apple~CloudDocs/MISTLab/Dev/misc/tfoptflow/tfoptflow/models/pwcnet-sm-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-592000'
nn_opts = deepcopy(_DEFAULT_FLOWDRONET_OPTS)
nn_opts['verbose'] = True
nn_opts['batch_size'] = 1
nn_opts['ckpt_path'] = ckpt_path
nn_opts['dronet_model_path'] = dronet_model_path
"""


# test large model
nn_opts = deepcopy(_DEFAULT_PWCNET_VAL_OPTIONS)
ckpt_path = '/Users/nfinite/data/tfoptflow/coll_newres_sm_ckpt_training/pwcnet.ckpt-7000'
nn_opts['verbose'] = True
nn_opts['batch_size'] = 1
nn_opts['ckpt_path'] = ckpt_path
nn_opts['dronet_model_path'] = dronet_model_path
nn_opts['gpu_devices'] = ['/device:CPU:0'],
nn_opts['x_shape'] = [2, imsize[0], imsize[1], 3]
nn_opts['y_shape'] = [imsize[0], imsize[1], 2]
nn_opts['use_tf_data'] = False          # Don't use tf.data reader for this simple task
nn_opts['use_dense_cx'] = False
nn_opts['use_res_cx'] = False
nn_opts['pyr_lvls'] = 6
nn_opts['flow_pred_lvl'] = 2


"""
# test refined large model
nn_opts = deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
ckpt_path = '/Users/nfinite/data/test/kitti_flow_ckpt_training/pwcnet.ckpt-3000'
nn_opts['verbose'] = True
nn_opts['batch_size'] = 1
nn_opts['ckpt_path'] = ckpt_path
nn_opts['dronet_model_path'] = dronet_model_path
nn_opts['gpu_devices'] = ['/device:CPU:0'],
nn_opts['x_shape'] = [2, imsize[0], imsize[1], 3]
nn_opts['y_shape'] = [imsize[0], imsize[1], 2]
nn_opts['use_tf_data'] = False          # Don't use tf.data reader for this simple task
nn_opts['use_dense_cx'] = True
nn_opts['use_res_cx'] = True
nn_opts['pyr_lvls'] = 6
nn_opts['flow_pred_lvl'] = 2
nn_opts['adapt_info'] = (1, 256, 448, 2)
nn_opts['sparse_gt_flow'] = True
"""

# Instantiate the model in inference mode and display the model configuration
nn = ModelFlowDroNet(mode='val', options=nn_opts)
nn.print_config()
dirs = sorted(os.listdir(data))


for subdir in dirs:
    path = os.path.join(data, subdir, 'images/*.jpg')
    imgs = sorted(glob.glob(path))

    prv  = cv2.imread(imgs[0])
    prv = cv2.resize(prv, (imsize[1], imsize[0]))

    for item in imgs:

        cur = cv2.imread(item)
        cur = cv2.resize(cur, (imsize[1], imsize[0]))
        img_pairs = ([prv, cur],)

        # Generate the predictions
        ts = time.time()
        flow, steer_coll = nn.flowdronet_predict(img_pairs)
        print("time: {:4.3f}".format(time.time() - ts))

        img_out = flow_to_img(flow*100, normalize=True, flow_mag_max=None)
        img_out = cv2.resize(img_out, outres, interpolation=cv2.INTER_AREA)

        prv = cur

        # ouput of results
        stats = ["Predictions:"]
        stats.append("[C: {:4.3f}] [SA: {:4.3f}]".format(float(steer_coll[1]), float(steer_coll[0])))

        # place output on image
        for idx, stat in enumerate(stats):
            text = stat.lstrip()
            cv2.putText(img_out, text, (5, 30 + (idx * 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, lineType=30)

        # show output
        cv2.imshow('frame', img_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()