
ROOT_DIR="/home/gkinoshita/dugong/workspace/Insta-DM"
DATA_DIR="${ROOT_DIR}/eigen_test_files"
TEST_FILE="${ROOT_DIR}/kitti_eval/test_files_eigen.txt"
MODEL_NAME="with_category_mni_5"
RESULTS_DIR="${ROOT_DIR}/outputs/eigen_test/${MODEL_NAME}"
PRED_FILE="${RESULTS_DIR}/disp_predictions.npy"

CHECKPOINT_DIR="${ROOT_DIR}/checkpoints"
# DISP_NET="${CHECKPOINT_DIR}/${MODEL_NAME}/12-07-18:42/dispnet_model_best.pth.tar"
DISP_NET="${CHECKPOINT_DIR}/${MODEL_NAME}/12-07-17:34/dispnet_model_best.pth.tar"


# CUDA_VISIBLE_DEVICES=0 python3 ${ROOT_DIR}/kitti_eval/compare_depth.py --img-height 256 --img-width 832 \
CUDA_VISIBLE_DEVICES=0 python3 ${ROOT_DIR}/kitti_eval/my_save_depth.py --img-height 256 --img-width 832 \
--pretrained-dispnet $DISP_NET --data_dir $DATA_DIR --dataset-list $TEST_FILE --output_dir $RESULTS_DIR --save-in-local

# python3 ${ROOT_DIR}/kitti_eval/my_eval_depth.py --data_dir $DATA_DIR --pred_file $PRED_FILE --test_file_list $TEST_FILE

