ROOT_DIR="/home/gkinoshita/dugong/workspace/Insta-DM"
# IMG_PATH="/home/gkinoshita/dugong/workspace/Insta-DM/kitti_256/image/2011_09_26_drive_0022_sync_02"
IMG_PATH="/home/gkinoshita/dugong/dataset/packnet-kitti-raw/KITTI_raw/2011_09_26/2011_09_26_drive_0022_sync/image_02/data"
# MODEL_NAME="with_category_mni_3"
MODEL_NAME="with_category_mni_5"

OUTPUT_DIR="${ROOT_DIR}/outputs/depth_map/${MODEL_NAME}"
CHECKPOINT_DIR="${ROOT_DIR}/checkpoints"
# DISP_NET="${CHECKPOINT_DIR}/${MODEL_NAME}/12-07-18:42/dispnet_model_best.pth.tar"
DISP_NET="${CHECKPOINT_DIR}/${MODEL_NAME}/12-07-17:34/dispnet_model_best.pth.tar"

CUDA_VISIBLE_DEVICES=0 python3 ${ROOT_DIR}/kitti_eval/save_single_depth_map.py --img-height 256 --img-width 832 \
--pretrained-dispnet $DISP_NET --img-path $IMG_PATH --output_dir $OUTPUT_DIR --save-in-local
