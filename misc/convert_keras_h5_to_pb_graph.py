#!/usr/bin/env python3
"""
convert_keras_h5_to_pb_graph.py

Converts a .h5 model created by keras (DroNet) into a regular Tensorflow protobuf graph.
This conversion is necessary to use the trained model with FlowDroNet

Written by Moritz Sperling

Licensed under the MIT License (see LICENSE for details)
"""
import sys
import os.path as osp
import tensorflow as tf
import absl.flags as gflags
from keras import backend as k
from keras.models import model_from_json
localpath = osp.dirname(osp.realpath(__file__))
sys.path.insert(0, localpath + '/../workflow/util/')
from common_flags import FLAGS


def _main():
    """
    Creates a protobuf .pb file from the frozen graph of a keras model stored in a .h5 file format.
    The output file is stored in the same location as input.

    Uses the Common Flags:
        input_dir, json_model_fname, weights_fname
    """

    # setup paths
    json_model_path = osp.join(FLAGS.input_dir, FLAGS.json_model_fname)
    weights_path = osp.join(FLAGS.input_dir, FLAGS.weights_fname)
    save_path = osp.splitext(json_model_path)[0][:-6] + "graph_w" + str(weights_path.split("_")[-1][:-3]) + ".pb"
    print("Loading Model: " + json_model_path)
    print("Loading Weights: " + weights_path)

    # Set keras to test phase
    k.set_learning_phase(0)

    # Load json and weights, then compile model
    with open(json_model_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)
    model.compile(loss='mse', optimizer='sgd')

    # Freeze graph
    frozen_graph = freeze_session(k.get_session(), output_names=[out.op.name for out in model.outputs])

    # Write graph to protobuf file
    tf.train.write_graph(frozen_graph, "model", save_path, as_text=False)
    print("Written Graph to: " + save_path)


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


def main(argv):
    # Utility main to load flags
    try:
        FLAGS(argv)  # parse flags
    except gflags.Error:
        print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
        sys.exit(1)
    _main()


if __name__ == "__main__":
    main(sys.argv)
