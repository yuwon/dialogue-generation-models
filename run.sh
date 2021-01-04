#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

BASE_DIR="/home/lim/research/dialogue-generation-models"
PYTHON="/home/lim/.anaconda3/envs/pytorch/bin/python"
SCRIPT="$BASE_DIR/server.py"

cd $BASE_DIR
$PYTHON $SCRIPT

