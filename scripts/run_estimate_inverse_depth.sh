#!bin/bash

ROOT_DIR=/home/gkinoshita/workspace/Insta-DM
PRETRAINED_PATH=pretrained/KITTI/resnet18_disp_kt.tar
ROOT_IMAGE_DIR=/home/gkinoshita/dugong/workspace/Insta-DM/kitti_256/image
ROOT_SAVE_DIR=/home/gkinoshita/dugong/workspace/Insta-DM/kitti_256/inverse_depth

CUDA_VISIBLE_DEVICES=0 python3 ${ROOT_DIR}/kitti_eval/estimate_inverse_depth.py \
--root-image-dir $ROOT_IMAGE_DIR \
--root-save-dir $ROOT_SAVE_DIR \
--pretrained-disp $PRETRAINED_PATH \
# --save-in-local