#!/bin/sh

if [ "$#" -ne 1 ]; then
    echo "give dir as arg"
    exit 1
fi

for f in $1/settings*
do
    inptdir="/data/test_model/Output/"
    rootdir="/data/test_model/"
    testdir="/data/collision_new/"
    weights="model_weights_19.h5"
    args1="--input_dir=$inptdir --experiment_rootdir=$rootdir --settings_fname=$f --output_dir=$rootdir"
    args2="--experiment_rootdir=$rootdir --settings_fname=$f --output_dir=$rootdir"
    args3="--experiment_rootdir=$rootdir --settings_fname=$f --test_dir=$testdir"
    args4="--experiment_rootdir=$rootdir --settings_fname=$f --weights_fname=$weights --show_activations=true --test_dir=$testdir"
    
    python "/dev/oflowoavoidance/workflow/step_01_prep_output.py" $args1
    python "/dev/oflowoavoidance/workflow/step_02_train.py" $args2
    python "/dev/oflowoavoidance/workflow/step_03_evaluate_all.py" $args3
    python "/dev/oflowoavoidance/workflow/step_05_evaluate_set.py" $args4
done
