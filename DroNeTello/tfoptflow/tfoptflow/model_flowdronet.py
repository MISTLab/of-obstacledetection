#!/usr/bin/env python3
"""
model_flowdronet.py

FlowDroNet model class.

Written by Moritz Sperling
Based on work by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)
"""

import numpy as np
import tensorflow as tf
from model_pwcnet import ModelPWCNet
from optflow import flow_to_img

_DEBUG_USE_REF_IMPL = False

# Default options
_DEFAULT_FLOWDRONET_OPTS = {
    'verbose': True,
    'ckpt_path': None,  # original checkpoint to finetune
    'dronet_model_path': None,  # path of the DroNet model to be used
    'dronet_mode': 'rgb',  # set flowdronet mode: 'rgb' to use with rgb-style flow images, 'raw' for raw flow images
    'ckpt_dir': './ckpts_finetuned/',  # where finetuning checkpoints are stored
    'max_to_keep': 50,
    'x_dtype': tf.float32,  # image pairs input type
    'x_shape': [2, 256, 448, 3],  # image pairs input shape [2, H, W, 3]
    'y_dtype': tf.float32,  # u,v flows output type
    'y_shape': [256, 448, 2],  # u,v flows output shape [H, W, 2]
    'train_mode': 'fine-tune',  # in ['train', 'fine-tune']
    'adapt_info': None,  # if predicted flows are padded by the model, crop them back by to this size
    'sparse_gt_flow': False,  # if gt flows are sparse (KITTI), only compute average EPE where gt flows aren't (0., 0.)
    # Logging/Snapshot params^
    'display_step': 100,  # show progress every 100 training batches
    'snapshot_step': 1000,  # save trained model every 1000 training batches
    'val_step': 1000,  # Test trained model on validation split every 1000 training batches
    'val_batch_size': -1,  # Use -1 to use entire validation split, or set number of val samples (0 disables it)
    'tb_val_imgs': 'top_flow',  # None, 'top_flow', or 'pyramid'; runs model on batch_size val images, log results
    'tb_test_imgs': None,  # None, 'top_flow', or 'pyramid'; runs trained model on batch_size test images, log results
    # Multi-GPU config
    # list devices on which to run the model's train ops (can be more than one GPU)
    'gpu_devices': ['/device:GPU:0'],
    # controller device to put the model's variables on (usually, /cpu:0 or /gpu:0 -> try both!)
    'controller': '/device:GPU:0',
    # Training config and hyper-params
    'use_tf_data': False,  # Set to True to get data from tf.data.Dataset; otherwise, use feed_dict with numpy
    'use_mixed_precision': False,  # Set to True to use mixed precision training (fp16 inputs)
    'loss_scaler': 128.,  # Loss scaler (only used in mixed precision training)
    'batch_size': 4,
    'lr_policy': 'cyclic',  # choose between None, 'multisteps', and 'cyclic'; adjust the max_steps below too
    # Multistep lr schedule
    'init_lr': 1e-05,  # initial learning rate
    'max_steps': 100000,  # max number of training iterations (i.e., batches to run)
    'lr_boundaries': [25000, 50000, 100000, 150000],  # step schedule boundaries
    'lr_values': [1e-05, 5e-06, 2.5e-06, 1.25e-06, 6.25e-07],  # step schedule values
    # Cyclic lr schedule
    'cyclic_lr_max': 5e-04,  # maximum bound
    'cyclic_lr_base': 1e-05,  # min bound
    'cyclic_lr_stepsize': 20000,  # step schedule values
    # 'max_steps': 200000, # max number of training iterations
    # Loss functions hyper-params
    'loss_fn': 'loss_multiscale',  # 'loss_robust' doesn't really work; the loss goes down but the EPE doesn't
    'alphas': [0.32, 0.08, 0.02, 0.01, 0.005],  # See 'Implementation details" on page 5 of ref PDF
    'gamma': 0.0004,  # See 'Implementation details" on page 5 of ref PDF
    'q': 0.4,  # See 'Implementation details" on page 5 of ref PDF
    'epsilon': 0.01,  # See 'Implementation details" on page 5 of ref PDF
    # Model hyper-params
    'pyr_lvls': 6,  # number of feature levels in the flow pyramid
    'flow_pred_lvl': 2,  # which level to upsample to generate the final optical flow prediction
    'search_range': 4,  # cost volume search range
    # if True, use model with dense connections (4705064 params w/o, 9374274 params with (no residual conn.))
    'use_dense_cx': False,
    # if True, use model with residual connections (4705064 params w/o, 6774064 params with (+2069000) (no dense conn.))
    'use_res_cx': False,
}


class ModelFlowDroNet(ModelPWCNet):
    def __init__(self, name='flowdronet', mode='train', session=None, options=None, dataset=None):
        """Initialize the ModelFloDroNet object
        Args:
            name: Model name
            mode: Possible values: 'train', 'val', 'test'
            session: optional TF session
            options: see _DEFAULT_PWCNET_TRAIN_OPTIONS comments
            dataset: Dataset loader
        Training Ref:
            See original PWC-Net Model
        """
        super().__init__(name, mode, session, options)
        if options is None:
            options = _DEFAULT_FLOWDRONET_OPTS
        self.ds = dataset
        self.dronet_graph = None
        self.dronet_x_tnsr = None
        self.dronet_y_tnsr = None
        self.dronet_sess = None
        self.set_dronet_graph(options['dronet_model_path'])

    def set_dronet_graph(self, filename):
        """
        Loads a FlowDroNet graph file.
        :param filename: Path to protobuf model file (.pb)
        """

        # We load the protobuf file from the disk and parse it to retrieve the
        # unserialized graph_def
        try:
            with tf.gfile.GFile(filename, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
        except IOError:
            print("DroNet model file not found.")

        # Then, we can use again a convenient built-in function to import a graph_def into the
        # current default Graph
        with tf.Graph().as_default() as graph:
            # Import graph and
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="flowdronet",
                producer_op_list=None
            )
            self.dronet_graph = graph

            self.dronet_sess = tf.Session(graph=graph)

            # Get tensor names for input, steering & collision
            input_name = graph.get_operations()[0].name + ':0'
            output_steer_name = graph.get_operations()[-6].name + ':0'
            output_coll_name = graph.get_operations()[-5].name + ':0'

            # Create tensors for model input and output
            self.dronet_x_tnsr = graph.get_tensor_by_name(input_name)
            y0 = graph.get_tensor_by_name(output_steer_name)
            y1 = graph.get_tensor_by_name(output_coll_name)
            self.dronet_y_tnsr = [y0, y1]

    def flowdronet_predict(self, img_pairs):
        """
        Inference loop. Run inference on a pair of images.
        Args:
            img_pairs: image pairs in (img_1, img_2) format.
        Returns:
            Predicted Steering Angle and Collision Probability
        """
        assert self.dronet_sess is not None, "DroNet session not loaded."

        # Predict Flow first
        with self.graph.as_default():
            # Chunk image pair list

            # Repackage input image pairs as np.ndarray
            x = np.array(img_pairs)

            # Make input samples conform to the network's requirements
            # x: [batch_size,2,H,W,3] uint8; x_adapt: [batch_size,2,H,W,3] float32
            x_adapt, x_adapt_info = self.adapt_x(x)
            if x_adapt_info is not None:
                y_adapt_info = (x_adapt_info[0], x_adapt_info[2], x_adapt_info[3], 2)
            else:
                y_adapt_info = None

            # Run the adapted samples through the network
            feed_dict = {self.x_tnsr: x_adapt}
            y_hat = self.sess.run(self.y_hat_test_tnsr, feed_dict=feed_dict)
            flow, _ = self.postproc_y_hat_test(y_hat, y_adapt_info)

        if self.opts['dronet_mode'] == 'raw':
            """
            Rescale flow to compensate for changing flow magnitudes.
            These operations could precede the DroNet network as tf ops (pseudo tf-code):
            
            mag = tf.norm(flow, axis=2)
            magmean = tf.reduce_mean(mag)
            flow = tf.divide(flow, magmean)
            """

            # Scale flow so that mag = 1
            flow_magnitude = np.linalg.norm(flow, axis=2)
            x = flow / np.mean(flow_magnitude)
        else:
            """
            Converting the flow to a rgb image is certainly not the optimal solution.
            However using raw flow doesn't work so far, maybe due to flow scaling issues or 
            incompatibilities in the res-net model architecture (e.g. ReLU-units).
            """

            f_img = flow_to_img(flow[0])
            x = np.asarray(f_img, dtype=np.float32) * np.float32(1.0 / 255.0)
            x = x[np.newaxis, ...]

        # Predict DroNet output from flow input if possible
        with self.dronet_graph.as_default():
            steer_coll = self.dronet_sess.run(self.dronet_y_tnsr, feed_dict={self.dronet_x_tnsr: x})

        return flow[0], steer_coll
