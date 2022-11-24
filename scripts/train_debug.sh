############################################################################################################

### Dataset directory ###
# TRAIN_DIR = ../kitti_odometry
ROOT_DIR=/home/gkinoshita/dugong/workspace/Insta-DM
TRAIN_DIR=${ROOT_DIR}/kitti_odometry

DEST_DIR_NAME="without_transform"

### KITTI model ###
PRETRAINED=${ROOT_DIR}/pretrained/KITTI

############################################################################################################

### For training ###
CUDA_VISIBLE_DEVICES=0 python train_debug.py $TRAIN_DIR \
--epoch-size 1000 \
--with-ssim --with-mask --with-auto-mask \

# ### For debugging ###
# CUDA_VISIBLE_DEVICES=2,3 python train.py $TRAIN_SET \
# --pretrained-disp $PRETRAINED/resnet18_disp_kt.tar \
# --pretrained-ego-pose $PRETRAINED/resnet18_ego_kt.tar \
# --pretrained-obj-pose $PRETRAINED/resnet18_obj_kt.tar \
# -b 1 -p 2.0 -c 1.0 -s 0.1 -o 0.02 -mc 0.1 -hp 0 -dm 0 -mni 2 \
# --epoch-size 1000 \
# --with-ssim --with-mask --with-auto-mask \
# --with-gt \
# -j 0 --name debug --debug-mode \
# --seed 0 \
