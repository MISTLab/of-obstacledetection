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
import shutil
from tqdm import tqdm
from copy import deepcopy
from model_flowdronet import ModelFlowDroNet
from model_pwcnet import _DEFAULT_PWCNET_VAL_OPTIONS
from optflow import flow_write

data = '/media/Slave/thesis/data/finalstuff/tftest/real'
out = '/media/Slave/thesis/data/finalstuff/tftest/orig'

imsize = (256, 448)
nthimage = 1

# Paths
dronet_model_path = '/home/moritz/Dev/oflowoavoidance/DroNeTello/models/FlowDroNet/model_graph_w27 2.pb'

# Here, we're using a GPU (use '/device:CPU:0' to run inference on the CPU)
gpu_devices = ['/device:GPU:0']  

# test large model
nn_opts = deepcopy(_DEFAULT_PWCNET_VAL_OPTIONS)
ckpt_path = '/media/Slave/thesis/dev/tfoptflow/tfoptflow/models/pwcnet-sm-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-592000'
nn_opts['verbose'] = True
nn_opts['batch_size'] = 1
nn_opts['ckpt_path'] = ckpt_path
nn_opts['dronet_model_path'] = dronet_model_path
nn_opts['gpu_devices'] = gpu_devices,
nn_opts['x_shape'] = [2, imsize[0], imsize[1], 3]
nn_opts['y_shape'] = [imsize[0], imsize[1], 2]
nn_opts['use_tf_data'] = False          # Don't use tf.data reader for this simple task
nn_opts['use_dense_cx'] = False
nn_opts['use_res_cx'] = False
nn_opts['pyr_lvls'] = 6
nn_opts['flow_pred_lvl'] = 2

# Instantiate the model in inference mode and display the model configuration
nn = ModelFlowDroNet(mode='test', options=nn_opts)
nn.print_config()
dirs = sorted(os.listdir(data))

j = 0
t = 0

for subdir in tqdm(dirs):
	m = 0
	path = os.path.join(data, subdir, 'images/*.jpg')
	imgs = sorted(glob.glob(path))

	prv  = cv2.imread(imgs[0])

	# delete and recreate output folder
	outfolder = os.path.join(out, subdir, 'images')
	if os.path.exists(outfolder):
		shutil.rmtree(outfolder)
	os.makedirs(outfolder)

	# copy labels file
	infile = os.path.join(data, subdir, 'labels.txt')
	if os.path.isfile(infile):
		with open(infile, 'r') as fin:
			dat = fin.read().splitlines(True)
		with open(os.path.join(out, subdir, 'labels.txt'), 'w') as fout:
			fout.writelines(dat[1::nthimage])
		fin.close()
		fout.close()

	for item in imgs[1::nthimage]:

		cur = cv2.imread(item)

		img_pairs = ([prv, cur],)

		# Generate the predictions
		ts = time.time()
		flow = nn.predict_from_img_pairs(img_pairs, batch_size=1, verbose=False)
		t = t + time.time() - ts

		imgname = os.path.split(item)[1].replace('.jpg', '.flo')
		outfile = os.path.join(outfolder, imgname)
		flow_write(flow[0], outfile)

		prv = cur
		j = j + 1

cv2.destroyAllWindows()
print("Average inference time: {:4.3f}".format(t / j))