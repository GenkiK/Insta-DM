############################################################################################################

### Dataset directory ###
ROOT_DIR="/home/gkinoshita/workspace/Insta-DM"
TRAIN_DIR="${ROOT_DIR}/kitti_256"
DEST_DIR_NAME="with_category_mni_3"
CKPT_TIMESTAMP="12-06-14:50"

############################################################################################################

### For training ###
CUDA_VISIBLE_DEVICES=0,1 python $ROOT_DIR/train_height_prior.py $TRAIN_DIR \
-b 4 -p 2.0 -c 1.0 -s 0.1 -o 0.5 -mc 0.1 -mni 3 \
--epoch-size 1000 \
--with-ssim --with-mask --with-auto-mask \
--dest-dir-name $DEST_DIR_NAME \
--resume --ckpt-timestamp $CKPT_TIMESTAMP --start-epoch 99
