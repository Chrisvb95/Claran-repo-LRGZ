#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128g
#SBATCH --job-name=rgz_train

#module load tensorflow/1.4.0-py27
#module load opencv openmpi

# if cuda driver is not in the system path, customise and add the following paths
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/extras/CUPTI/lib64

export PYTHONPATH=$PYTHONPATH:/data/s1587064/venv/lib/python2.7/site-packages
LRGZ_RCNN=/data/s1587064/Claran-repo-LRGZ/
CUDA_VISIBLE_DEVICES=1 python $LRGZ_RCNN/tools/train_net.py \
                    --device gpu \
                    --device_id 1 \
                    --imdb rgz_2017_trainD3 \
                    --iters 100000 \
                    --cfg $LRGZ_RCNN/experiments/cfgs/faster_rcnn_end2end.yml \
                    --network rgz_train \
                    --weights $LRGZ_RCNN/data/pretrained_model/imagenet/VGG_imagenet.npy 