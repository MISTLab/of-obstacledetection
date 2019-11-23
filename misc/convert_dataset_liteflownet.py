#!/usr/bin/env python
##################
#
# Use this script with LiteFlowNet to convert a dataset to OF
# The script needs to be in the models/testing subfolder to work.
#
##################
import glob
import csv
import os, sys
import subprocess
import cv2
import shutil
from math import ceil
from optflow import flow_read,flow_to_img

caffe_bin = 'bin/caffe.bin'
template = './deploy.prototxt'
cnn_model = 'liteflownet-ft-kitti'    # MODEL = liteflownet, liteflownet-ft-sintel or liteflownet-ft-kitti

input_fol = '/data/original'
outputfol = '/data/converted'
extension = 'jpg'
width = 1242
height = 375
outsize = (320,240)
nthimage = 1

os.environ['GLOG_minloglevel'] = '1'

# =========================================================

#setup folders
my_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(my_dir)
imginfol = os.path.join(my_dir, 'tmp/images_in')
imgoutfol = os.path.join(my_dir, 'tmp/images_out')

# create prototxt file
divisor = 32.
adapted_width = ceil(width/divisor) * divisor
adapted_height = ceil(height/divisor) * divisor
rescale_coeff_x = width / adapted_width
rescale_coeff_y = height / adapted_height
replacement_list = {
    '$ADAPTED_WIDTH': ('%d' % adapted_width),
    '$ADAPTED_HEIGHT': ('%d' % adapted_height),
    '$TARGET_WIDTH': ('%d' % width),
    '$TARGET_HEIGHT': ('%d' % height),
    '$SCALE_WIDTH': ('%.8f' % rescale_coeff_x),
    '$SCALE_HEIGHT': ('%.8f' % rescale_coeff_y),
    '$OUTFOLDER': ('%s' % '"' + imgoutfol + '"'),
    '$CNN': ('%s' % '"' + cnn_model + '-"')
}

proto = ''
with open(template, "r") as tfile:
    proto = tfile.read()

for r in replacement_list:
    proto = proto.replace(r, replacement_list[r])

with open('tmp/deploy.prototxt', "w") as tfile:
    tfile.write(proto)

# iterate through subfolders of input
dirs = sorted(os.listdir(input_fol))
for subdir in dirs:

	# reset temporary folders	
	if os.path.exists(imginfol):
		shutil.rmtree(imginfol)
	os.makedirs(imginfol)
	
	if os.path.exists(imgoutfol):
		shutil.rmtree(imgoutfol)
	os.makedirs(imgoutfol)

	# create output subdir
	outputfolder = os.path.join(outputfol, subdir)
	if os.path.exists(os.path.join(outputfolder, 'images')):
		shutil.rmtree(outputfolder)
	os.makedirs(os.path.join(outputfolder, 'images'))

	# get all images
	files = sorted(glob.glob(os.path.join(input_fol, subdir, ('images/*.' + extension))))
	files = files[::nthimage]

	# convert images to right size
	for file in files:
		img = cv2.imread(file, cv2.IMREAD_COLOR)
		img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
		fname = os.path.split(file)[1]
		cv2.imwrite(os.path.join(imginfol, fname), img)

	# create txt files containing the images
	with open(os.path.join(my_dir, 'tmp/img1.txt'), 'w') as myfile1:
		for file in files[:-1]:
			fname = os.path.join(imginfol, os.path.split(file)[1])
			myfile1.write(fname + '\n')

	with open(os.path.join(my_dir, 'tmp/img2.txt'), 'w') as myfile2:
		for file in files[1:]:
			fname = os.path.join(imginfol, os.path.split(file)[1])
			myfile2.write(fname + '\n')
	myfile1.close()
	myfile2.close()

	# Run caffe
	args = [caffe_bin, 'test', '-model', 'tmp/deploy.prototxt',
	        '-weights', '../trained/' + cnn_model + '.caffemodel',
	        '-iterations', str(len(files[:-1])),
	        '-gpu', '0']

	cmd = str.join(' ', args)
	print('Executing %s' % cmd)

	subprocess.call(args)

	# convert .flo output files to images
	outfiles = sorted(glob.glob(os.path.join(imgoutfol, '*.flo')))
	for idx, file in enumerate(outfiles):
		uv  = flow_read(file)
		img_out = flow_to_img(uv, normalize=True, flow_mag_max=None)
		outfile = os.path.join(outputfolder, 'images', os.path.split(files[idx])[1])
		img_out = cv2.resize(img_out, outsize, interpolation=cv2.INTER_AREA)
		print("Writing: " + outfile)
		cv2.imwrite(outfile, img_out)

	# copy labels file
	infile = os.path.join(input_fol, subdir, 'labels.txt')
	if os.path.isfile(infile):
		with open(infile, 'r') as fin:
			dat = fin.read().splitlines(True)
		with open(os.path.join(outputfolder,'labels.txt'), 'w') as fout:
			fout.writelines(dat[1::nthimage])
		fin.close()
		fout.close()
	print('Done: ' + os.path.join(input_fol, subdir))
print('DONE DONE!')