############################################################################################################


ROOT_DIR="/home/gkinoshita/dugong/workspace/Insta-DM"
DATA_DIR="${ROOT_DIR}/kitti_odometry_test"
TEST_FILE="${ROOT_DIR}/kitti_eval/odometry_test.txt"
RESULTS_DIR="${ROOT_DIR}/outputs/pose_test"
MODEL_NAME="without_direction_loss_bug_fixed"
RESULTS_DIR="${ROOT_DIR}/outputs/pose_test/${MODEL_NAME}"

CHECKPOINT_DIR="${ROOT_DIR}/checkpoints"
POSE_NET="${CHECKPOINT_DIR}/${MODEL_NAME}/11-10-18:02/ego_pose_model_best.pth.tar"

### (1) Predict depth and save results to "$RESULTS_DIR/predictions.npy" ###
CUDA_VISIBLE_DEVICES=0 python3 ${ROOT_DIR}/kitti_eval/my_save_pose.py --img-height 256 --img-width 832 \
--pretrained-posenet $POSE_NET --data_dir $DATA_DIR --dataset-list $TEST_FILE --output_dir $RESULTS_DIR

### (2) Evaluate depth with GT ###
python3 ${ROOT_DIR}/kitti_eval/my_eval_pose_2.py --gt-dir "${DATA_DIR}/poses" --result $RESULTS_DIR --seqs 9 10


