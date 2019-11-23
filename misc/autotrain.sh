#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "give dir as arg"
    exit 1
fi

valdir="$1/val"
traindir="$1/train"
testdir="/data/collision_new/"
outdir="/data/test/eval/"
workflowpath="/dev/workflow/"

mkdir $traindir

for i in $(seq 0 1 8)
do
	modeldir="$1/model_${i}ep"
	for in in $1/input/ep_$i*
	do
	    echo $in $traindir
    done

    cd $workflowpath
    export PATH=$PATH:$workflowpath/util

    args="--experiment_rootdir=$modeldir --train_dir=$traindir --val_dir=$valdir --img_mode=flow"
    python3  "step_02_train.py" $args

    args2="--experiment_rootdir=$modeldir --test_dir=$testdir --img_mode=flow --output_dir=$outdir --name=eval_model_${i}ep"
    python3 "step_03_evaluate_all.py" $args2

done
