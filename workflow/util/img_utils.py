"""
img_utils.py

Optical flow I/O and visualization functions.

Written by Moritz Sperling
Based on the works of P. Ferriere (https://github.com/philferriere/tfoptflow)
and of A. Loquercio et al., 2018 (https://github.com/uzh-rpg/rpg_public_dronet)

Licensed under the MIT License (see LICENSE for details)

Refs:
    - Per MPI-Sintel/flow_code/C/flowIO.h and flowIO.cpp:

    // the "official" threshold - if the absolute value of either
    // flow component is greater, it's considered unknown
    #define UNKNOWN_FLOW_THRESH 1e9

    // value to use to represent unknown flow
    #define UNKNOWN_FLOW 1e10

    // first four bytes, should be the same in little endian
    #define TAG_FLOAT 202021.25  // check for this when READING the file
    #define TAG_STRING "PIEH"    // use this when WRITING the file

    // ".flo" file format used for optical flow evaluation
    //
    // Stores 2-band float image for horizontal (u) and vertical (v) flow components.
    // Floats are stored in little-endian order.
    // A flow value is considered "unknown" if either |u| or |v| is greater than 1e9.
    //
    //  bytes  contents
    //
    //  0-3     tag: "PIEH" in ASCII, which in little endian happens to be the float 202021.25
    //          (just a sanity check that floats are represented correctly)
    //  4-7     width as an integer
    //  8-11    height as an integer
    //  12-end  data (width*height*2*4 bytes total)
    //          the float values for u and v, interleaved, in row order, i.e.,
    //          u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...

    - Numpy docs:
    ndarray.tofile(fid, sep="", format="%s")
    https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.ndarray.tofile.html#numpy.ndarray.tofile

    numpy.fromfile(file, dtype=float, count=-1, sep='')
    https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.fromfile.html
"""

from __future__ import absolute_import, division, print_function

import os
import warnings

import cv2
import numpy as np
import scipy.io as sio
from skimage.io import imsave

##
# I/O utils
##

TAG_FLOAT = 202021.25


def clean_dst_file(dst_file):
    """
    Create the output folder, if necessary; empty the output folder of previous predictions, if any

    Args:
        dst_file: Destination path
    """
    # Create the output folder, if necessary
    dst_file_dir = os.path.dirname(dst_file)
    if not os.path.exists(dst_file_dir):
        os.makedirs(dst_file_dir)

    # Empty the output folder of previous predictions, if any
    if os.path.exists(dst_file):
        os.remove(dst_file)


def flow_read(src_file, target_size=None):
    """
    Read optical flow stored in a .flo, .pfm, or .png file

    Args:
        src_file: Path to flow file
        target_size: Resize flow to target size
    Returns:
        flow: optical flow in [h, w, 2] format
    Refs:
        - Interpret bytes as packed binary data
        Per https://docs.python.org/3/library/struct.html#format-characters:
        format: f -> C Type: float, Python type: float, Standard size: 4
        format: d -> C Type: double, Python type: float, Standard size: 8
    Based on:
        - To read optical flow data from 16-bit PNG file:
        https://github.com/ClementPinard/FlowNetPytorch/blob/master/datasets/KITTI.py
        Written by Clément Pinard, Copyright (c) 2017 Clément Pinard
        MIT License
        - To read optical flow data from PFM file:
        https://github.com/liruoteng/OpticalFlowToolkit/blob/master/lib/pfm.py
        Written by Ruoteng Li, Copyright (c) 2017 Ruoteng Li
        License Unknown
        - To read optical flow data from FLO file:
        https://github.com/daigo0927/PWC-Net_tf/blob/master/flow_utils.py
        Written by Daigo Hirooka, Copyright (c) 2018 Daigo Hirooka
        MIT License
    """
    # Read in the entire file, if it exists
    assert(os.path.exists(src_file))

    if src_file.lower().endswith('.flo'):

        with open(src_file, 'rb') as f:

            # Parse .flo file header
            tag = float(np.fromfile(f, np.float32, count=1)[0])
            assert(tag == TAG_FLOAT)
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]

            # Read in flow data and reshape it
            flow = np.fromfile(f, np.float32, count=h * w * 2)
            flow.resize((h, w, 2))

    elif src_file.lower().endswith('.png'):

        # Read in .png file
        flow_raw = cv2.imread(src_file, -1)

        # Convert from [H,W,1] 16bit to [H,W,2] float formet
        flow = flow_raw[:, :, 2:0:-1].astype(np.float32)
        flow = flow - 32768
        flow = flow / 64

        # Clip flow values
        flow[np.abs(flow) < 1e-10] = 1e-10

        # Remove invalid flow values
        invalid = (flow_raw[:, :, 0] == 0)
        flow[invalid, :] = 0

    elif src_file.lower().endswith('.pfm'):

        with open(src_file, 'rb') as f:

            # Parse .pfm file header
            tag = f.readline().rstrip().decode("utf-8")
            assert(tag == 'PF')
            dims = f.readline().rstrip().decode("utf-8")
            w, h = map(int, dims.split(' '))
            scale = float(f.readline().rstrip().decode("utf-8"))

            # Read in flow data and reshape it
            flow = np.fromfile(f, '<f') if scale < 0 else np.fromfile(f, '>f')
            flow = np.reshape(flow, (h, w, 3))[:, :, 0:2]
            flow = np.flipud(flow)

    elif src_file.lower().endswith('.bin'):
        ts = str(target_size[1]) + ', ' + str(target_size[0]) or '256, 448'
        f = open(src_file, 'rb')
        flow = np.fromfile(f, dtype=np.dtype('(' + ts + ', 4)=f4'))
        flow = flow[0, ..., :2]
        flow = np.flipud(flow)

    elif src_file.lower().endswith('.npy'):

        flow = np.load(src_file)

    else:
        raise IOError

    return flow


def flow_write(flow, dst_file, compress=False):
    """
    Write optical flow to a .flo file

    Args:
        flow: optical flow
        compress: store optical flow as png instead of raw .flo
        dst_file: Path where to write optical flow
    """
    # Create the output folder, if necessary
    # Empty the output folder of previous predictions, if any
    clean_dst_file(dst_file)

    if not compress:
        # Save optical flow to disk
        with open(dst_file, 'wb') as f:
            np.array(TAG_FLOAT, dtype=np.float32).tofile(f)
            height, width = flow.shape[:2]
            np.array(width, dtype=np.uint32).tofile(f)
            np.array(height, dtype=np.uint32).tofile(f)
            flow.astype(np.float32).tofile(f)
    else:
        # Convert to png, then save
        flow_back = flow.astype(np.float32) * 64 + 32768
        flow_save = np.ones((flow_back.shape[0], flow_back.shape[1], 3), dtype=np.uint16)
        flow_save[..., 2:0:-1] = flow_back
        cv2.imwrite(dst_file, flow_save)
##
# Visualization utils
##


def flow_mag_stats(flow):
    """
    Get the average flow magnitude from a flow field.

    Args:
        flow: optical flow
    Returns:
        Average flow magnitude
    Ref:
        - OpenCV 3.0.0-dev documentation » OpenCV-Python Tutorials » Video Analysis »
        https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
    """
    # Convert the u,v flow field to angle,magnitude vector representation
    flow_magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # A couple times, we've gotten NaNs out of the above...
    nans = np.isnan(flow_magnitude)
    if np.any(nans):
        nans = np.where(nans)
        flow_magnitude[nans] = 0.

    return np.min(flow_magnitude), np.mean(flow_magnitude), np.max(flow_magnitude)


def flow_to_img(flow, normalize=True, info=None, flow_mag_max=None, return_mag=False):
    """
    Convert flow to viewable image, using color hue to encode flow vector orientation, and color saturation to
    encode vector length. This is similar to the OpenCV tutorial on dense optical flow, except that they map vector
    length to the value plane of the HSV color model, instead of the saturation plane, as we do here.

    Args:
        flow: optical flow
        normalize: Normalize flow to 0..255
        info: Text to superimpose on image (typically, the epe for the predicted flow)
        flow_mag_max: Max flow to map to 255
        return_mag: return magnitude only
    Returns:
        img: viewable representation of the dense optical flow in RGB format
        flow_avg: optionally, also return average flow magnitude
    Ref:
        - OpenCV 3.0.0-dev documentation » OpenCV-Python Tutorials » Video Analysis »
        https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
    """
    hsv = np.ones((flow.shape[0], flow.shape[1], 3), dtype=np.uint8) * 255
    flow_magnitude, flow_angle = cv2.cartToPolar(flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32))

    # A couple times, we've gotten NaNs out of the above...
    nans = np.isnan(flow_magnitude)
    if np.any(nans):
        nans = np.where(nans)
        flow_magnitude[nans] = 0.

    # Normalize
    hsv[..., 0] = (flow_angle * 180 / np.pi / 2 - 65) % 180
    if normalize is True:
        if flow_mag_max is None:
            hsv[..., 2] = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        else:
            hsv[..., 2] = np.clip(flow_magnitude * 255 / flow_mag_max, 0, 255)
    else:
        hsv[..., 2] = flow_magnitude

    if return_mag:
        # For magnitude mode
        return np.asarray(hsv[..., 2], dtype=np.uint8)
    else:
        # Convert
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Add text to the image, if requested
        if info is not None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, info, (20, 20), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        return np.asarray(img, dtype=np.uint8)


def flow_write_as_png(flow, dst_file, info=None, flow_mag_max=None):
    """
    Write optical flow to a .PNG file

    Args:
        flow: optical flow
        dst_file: Path where to write optical flow as a .PNG file
        info: Text to superimpose on image (typically, the epe for the predicted flow)
        flow_mag_max: Max flow to map to 255
    """
    # Convert the optical flow field to RGB
    img = flow_to_img(flow, flow_mag_max=flow_mag_max)

    # Create the output folder, if necessary
    # Empty the output folder of previous predictions, if any
    clean_dst_file(dst_file)

    # Add text to the image, if requested
    if info is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, info, (20, 20), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    # Save RGB version of optical flow to disk
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imsave(dst_file, img)


def load_img(path, img_mode="rgb", target_size=None, crop_size=None):
    """
    Load an image.

    # Arguments
        path: Path to image file.
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_width, img_height)`.
        crop_size: Either `None` (default to original size)
            or tuple of ints `(img_width, img_height)`.

    # Returns
        Image as numpy array.
    """

    if img_mode in ["flow", "flow_as_rgb", "flow_as_mag"]:
        img = flow_read(path, target_size)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    origshape = img.shape

    if img_mode in ["grayscale", "depth"]:
        if len(img.shape) != 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if target_size is not None:
        if (img.shape[1], img.shape[0]) != target_size:
            img = cv2.resize(img, target_size)

            # scale flow if image was resized
            if img_mode in ["flow", "flow_as_rgb", "flow_as_mag"]:
                img[..., 0] = img[..., 0] * img.shape[0] / origshape[0]
                img[..., 1] = img[..., 1] * img.shape[1] / origshape[1]
    if crop_size:
        if (img.shape[1], img.shape[0]) != crop_size:
            img = central_image_crop(img, crop_size[0], crop_size[1])

    if img_mode in ["grayscale", "depth"]:
        img = img.reshape((img.shape[0], img.shape[1], 1))

    return img


def central_image_crop(img, crop_width=150, crop_heigth=150):
    """
    Crop the input image centered in width and starting from the bottom
    in height.

    # Arguments:
        crop_width: Width of the crop.
        crop_heigth: Height of the crop.

    # Returns:
        Cropped image.
    """
    half_the_width = int(img.shape[0] / 2)
    img = img[half_the_width - int(crop_width / 2):half_the_width + int(crop_width / 2),
              img.shape[1] - crop_heigth: img.shape[1]]
    return img


def sp_generator(image, amount, img_mode="rgb"):
    """
    Render salt & pepper noise to a given image.
    :param image:   Input image
    :param amount:  Desired percentage of noise
    :param mode:    Input mode (grayscale, rgb or flow)
    :return:        Image with salt & pepper noise added
    """
    s_vs_p = 0.5

    # Maximum values for salt&pepper
    minval = 0
    maxval = 255
    if img_mode in ["flow", "flow_as_rgb", "flow_as_mag"]:
        maxval = np.maximum(np.amax(image), np.abs(np.amin(image)))
        minval = -maxval


    # Generate Salt '1' noise
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
    image[tuple(coords)] = maxval

    # Generate Pepper '0' noise
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
    image[tuple(coords)] = minval
    return image


def pred_as_bar(pred, size, label, mode="vertical"):
    """
    Creates a vertical bar for displaying values of a prediction/label.
    :param pred:    Collision prediction [0 .. 1]
    :param size:    Desired size of bar in pixel
    :param label:   Put text on the bar
    :param mode:    Vertical or Horizontal bar
    :return:        Image of a prediction bar
    """

    # make sure input is in range
    if 0 <= pred <= 1:
        # build bar
        bar = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        index = int(float(1 - pred) * size[0])
        bar[:, index:, 2] = np.ones_like(bar[:, index:, 2]) * 255
        bar[:, :index, 1] = np.ones_like(bar[:, :index, 1]) * 255

        # print label and rotate for vertical bar
        cv2.putText(bar, label, (5, 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1)

    else:
        print("Value Error for predictbar.")
        bar = np.ones((size[0], size[1], 3), dtype=np.uint8) * 255

    if mode == "vertical":
        for i in range(3):
            bar = np.rot90(bar, 3)

    return bar


def pred_as_indicator(pred, size, label, mode="horizontal"):
    """
    Creates a horizontal bar for displaying values of a prediction/label.
    :param pred:    Steering prediction [-1 .. 1]
    :param size:    Desired size of bar in pixel
    :param label:   Put text on the bar
    :param mode:    Vertical or Horizontal bar
    :return:        Image of a prediction bar
    """

    # make sure input is in range
    if -1 <= pred <= 1:
        bar = np.ones((size[0], size[1], 3), dtype=np.uint8) * 255

        # decide color (which channels t
        if abs(pred) < 0.3:
            c = [0, 2]
        elif abs(pred) < 0.6:
            c = 0
        else:
            c = [0, 1]

        mid = int(size[1] / 2)
        off = int(float(pred) * size[1] / 2)
        bar[:, min([mid, mid + off]):max([mid, mid + off]), c] = 0

        # print label
        cv2.putText(bar, label, (int(size[1] / 2 - 40), 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1)
    else:
        print("Value Error for predictbar.")
        bar = np.ones((size[0], size[1], 3), dtype=np.uint8) * 255

    if mode == "vertical":
        for i in range(3):
            bar = np.rot90(bar, 3)

    return bar


class FlowStatsCollector:
    """
    Class for collecting the stats of flow images. Stats include:
    - Histograms of U & V channel
    - Max, Min and Mean of U & V channel
    - Max, Min and Mean of magnitude
    Intended to use with "eval_flow_stats.m"
    """

    def __init__(self, filepath, lim=100, n=200, img_mode="flow", name="stats", clip=True):
        self.name = name
        self.folder_out = filepath
        self.img_mode = img_mode
        self.n = n
        self.lim = lim
        self.clip = clip

        # init
        self.n_px = 0
        self.n_im = 0
        self.mmaxx = []
        self.mminx = []
        self.mmeax = []
        self.mmaxy = []
        self.mminy = []
        self.mmeay = []
        self.mmaxm = []
        self.mminm = []
        self.mmeam = []
        self.histx = []
        self.histy = []
        self.binsx = []
        self.binsy = []

    def collect_exp_flow_stats(self, exp_flow):
        """
        Collect stats of an entire experiment for the lazy.
        :param exp_flow: input experiment array containing flow images
        """
        for flow in exp_flow:
            self.collect_flow_stats(flow)

    def collect_flow_stats(self, flow):
        """
        Collects the stats from a flow image
        :param flow: input flow
        """
        if self.clip:
            data = np.clip(flow, -self.lim, self.lim)
        else:
            data = flow

        self.n_px = flow.shape[0] * flow.shape[1]
        self.n_im = self.n_im + 1

        # get max min mean
        mmin, mmea, mmax = flow_mag_stats(flow)
        self.mmaxx.append(float(np.amax(data[..., 0])))
        self.mminx.append(float(np.amin(data[..., 0])))
        self.mmeax.append(float(np.mean(data[..., 0])))
        self.mmaxy.append(float(np.amax(data[..., 1])))
        self.mminy.append(float(np.amin(data[..., 1])))
        self.mmeay.append(float(np.mean(data[..., 1])))
        self.mmaxm.append(float(mmax))
        self.mminm.append(float(mmin))
        self.mmeam.append(float(mmea))

        # get histograms
        if self.clip:
            bins = np.linspace(-self.lim, self.lim, self.n + 1)
            thistx, binsx = np.histogram(data[..., 0], bins=bins)
            thisty, binsy = np.histogram(data[..., 1], bins=bins)
        else:
            thistx, binsx = np.histogram(data[..., 0], bins=self.n)
            thisty, binsy = np.histogram(data[..., 1], bins=self.n)

        # store histograms
        self.binsx.append(binsx)
        self.binsy.append(binsy)
        self.histx.append(thistx)
        self.histy.append(thisty)

    def write_flow_stats(self):
        """
        Writes json file containing the stats to predefined folder.
        """

        data = {'histogram_x': self.histx,
                'histogram_y': self.histy,
                'bins_x': self.binsx,
                'bins_y': self.binsy,
                'n_img': self.n_im,
                'n_pix': self.n_px,
                'max_x': self.mmaxx,
                'min_x': self.mminx,
                'mean_x': self.mmeax,
                'max_y': self.mmaxy,
                'min_y': self.mminy,
                'mean_y': self.mmeay,
                'max_mag': self.mmaxm,
                'min_mag': self.mminm,
                'mean_mag': self.mmeam,
                }

        fname = os.path.join(self.folder_out, ("flow_data_" + self.name + ".mat"))
        sio.savemat(fname, data)
        print("Written to file: " + fname)

    def print_flow_stats(self):
        # Prints out some Stats.
        print("Stats for: " + self.name)
        print("Pixel Count: {:d}".format(self.n_px))
        print("----------------------------")
        print("Mean Max X: {:6.2f}".format(np.mean(self.mmaxx)))
        print("Mean Max Y: {:6.2f}".format(np.mean(self.mmaxy)))
        print("Mean Min X: {:6.2f}".format(np.mean(self.mminx)))
        print("Mean Min Y: {:6.2f}".format(np.mean(self.mminy)))
        print("----------------------------")
        print("Min  Mag: {:6.2f}".format(np.min(self.mminm)))
        print("Mean Mag: {:6.2f}".format(np.mean(self.mmeam)))
        print("Max  Mag: {:6.2f}".format(np.max(self.mmaxm)))

    def get_mean_magnitude(self):
        # Get Current mean magnitude
        return np.mean(self.mmeam)
