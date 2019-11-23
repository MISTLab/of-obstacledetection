#!/bin/sh

if [ "$#" -ne 1 ]; then
    echo "give dir as arg"
    exit 1
fi

for f in $1/model*
do
    #testdir="$1/val"
    modeldir=$(echo $f | cut -d '/' -f 7)
    rootdir="data/all_models/"
    testdir="/collision_new/"
    evaldir="/data/all_models/eval/"
    args="--experiment_rootdir=$rootdir --model_dir=$f --test_dir=$testdir --eval_dir=$evaldir --img_mode=rgb --name=$modeldir"
    python "/dev/oflowoavoidance/workflow/step_03_evaluate_all.py" $args
done
