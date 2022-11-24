#!/bin/bash

PRETRAINED=pretrained/KITTI

# KITTI_DIR=/home/gkinoshita/workspace/Insta-DM/kitti_256
KITTI_DIR=/home/gkinoshita/dugong/workspace/Insta-DM/kitti_256

VALIDATION_TXT=$KITTI_DIR/val.txt
while read LINE
do
    echo $LINE
    ### Unified Visual Odometry ###
    CUDA_VISIBLE_DEVICES=0 python demo.py \
    --data $KITTI_DIR/image/$LINE \
    --pretrained-disp $PRETRAINED/resnet18_disp_kt.tar \
    --pretrained-ego-pose $PRETRAINED/resnet18_ego_kt.tar \
    --pretrained-obj-pose $PRETRAINED/resnet18_obj_kt.tar \
    --mni 3 \
    --name demo_val \
    # --save-fig
done < "${VALIDATION_TXT}"

# Errorが出るものだけをピックアップ
# for LINE in '2011_09_26_drive_0011_sync_03' '2011_09_26_drive_0014_sync_03' # '2011_09_26_drive_0011_sync_02'
# do
#     echo $LINE
#     ### Unified Visual Odometry ###
#     CUDA_VISIBLE_DEVICES=0 python demo.py \
#     --data $KITTI_DIR/image/$LINE \
#     --pretrained-disp $PRETRAINED/resnet18_disp_kt.tar \
#     --pretrained-ego-pose $PRETRAINED/resnet18_ego_kt.tar \
#     --pretrained-obj-pose $PRETRAINED/resnet18_obj_kt.tar \
#     --mni 3 \
#     --name demo \
#     --save-fig
#     echo ""
# done

# SCENE=/home/gkinoshita/workspace/Insta-DM/kitti_256/image/

############################################################################################################

### Unified Visual Odometry ###
# CUDA_VISIBLE_DEVICES=0 python demo.py \
# --data $SCENE \
# --pretrained-disp $PRETRAINED/resnet18_disp_kt.tar \
# --pretrained-ego-pose $PRETRAINED/resnet18_ego_kt.tar \
# --pretrained-obj-pose $PRETRAINED/resnet18_obj_kt.tar \
# --mni 3 \
# --name demo \
# --save-fig \
