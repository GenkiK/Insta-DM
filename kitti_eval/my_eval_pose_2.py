import copy
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


class KittiEvalOdom:
    """Evaluate odometry result
    Usage example:
        vo_eval = KittiEvalOdom()
        vo_eval.eval(gt_pose_txt_dir, result_pose_txt_dir)
    """

    def __init__(self):
        self.lengths = [100, 200, 300, 400, 500, 600, 700, 800]
        self.num_lengths = len(self.lengths)

    def load_poses_from_txt(self, file_path: Path):
        """Load poses from txt (KITTI format)
        Each line in the file should follow one of the following structures
            (1) idx pose(3x4 matrix in terms of 12 numbers)
            (2) pose(3x4 matrix in terms of 12 numbers)
        Args:
            file_name (str): txt file path
        Returns:
            poses (dict): {idx: 4x4 array}
        """
        # with open(file_path, "r") as f:
        #     s = f.readlines()
        # poses = {}
        # for cnt, line in enumerate(s):
        #     P = np.eye(4)
        #     line_split = [float(i) for i in line.split(" ") if i != ""]
        #     withIdx = len(line_split) == 13
        #     for row in range(3):
        #         for col in range(4):
        #             P[row, col] = line_split[row * 4 + col + withIdx]
        #     if withIdx:
        #         frame_idx = line_split[0]
        #     else:
        #         frame_idx = cnt
        #     poses[frame_idx] = P
        # return poses

        poses: dict[int, np.ndarray] = {}
        with open(file_path, "r") as f:
            for idx, line in enumerate(f):
                if line == "\n":
                    break
                arr = np.fromiter(map(float, line.rstrip().split(" ")), dtype=float).reshape(3, 4)
                arr = np.concatenate((arr, np.array([[0, 0, 0, 1]])), axis=0)
                poses[idx] = arr
        return poses

    def plot_trajectory(self, poses_gt, poses_result, seq):
        """Plot trajectory for both GT and prediction
        Args:
            poses_gt (dict): {idx: 4x4 array}; ground truth poses
            poses_result (dict): {idx: 4x4 array}; predicted poses
            seq (int): sequence index.
        """
        plot_keys = ["Ground Truth", "Ours"]
        fontsize_ = 20

        poses_dict = {}
        poses_dict["Ground Truth"] = poses_gt
        poses_dict["Ours"] = poses_result

        fig = plt.figure()
        ax = plt.gca()
        ax.set_aspect("equal")

        for key in plot_keys:
            pos_xz = []
            frame_idx_list = sorted(poses_dict["Ours"].keys())
            for frame_idx in frame_idx_list:
                pose = poses_dict[key][frame_idx]
                pos_xz.append([pose[0, 3], pose[2, 3]])
            pos_xz = np.asarray(pos_xz)
            plt.plot(pos_xz[:, 0], pos_xz[:, 1], label=key)

        plt.legend(loc="upper right", prop={"size": fontsize_})
        plt.xticks(fontsize=fontsize_)
        plt.yticks(fontsize=fontsize_)
        plt.xlabel("x (m)", fontsize=fontsize_)
        plt.ylabel("z (m)", fontsize=fontsize_)
        fig.set_size_inches(10, 10)
        plt.show()

    def eval(self, gt_dir: Path, result_dir: Path, alignment: str, seqs: list[str]):
        """Evaluate required/available sequences
        Args:
            gt_dir (str): ground truth poses txt files directory
            result_dir (str): pose predictions txt files directory
            alignment (str): if not None, optimize poses by
                - scale: optimize scale factor for trajectory alignment and evaluation
                - scale_7dof: optimize 7dof for alignment and use scale for trajectory evaluation
                - 7dof: optimize 7dof for alignment and evaluation
                - 6dof: optimize 6dof for alignment and evaluation
            seqs (list[str]):
                - list: list of sequence indices to be evaluated
        """
        # Initialization
        self.gt_dir = gt_dir

        # Create result directory
        error_dir = result_dir / "errors"
        self.plot_path_dir = result_dir / "plot_path"
        self.plot_error_dir = result_dir / "plot_error"
        result_txt = result_dir / "result.txt"

        error_dir.mkdir(parents=True, exist_ok=True)
        self.plot_path_dir.mkdir(exist_ok=True)
        self.plot_error_dir.mkdir(exist_ok=True)

        self.eval_seqs = seqs

        with open(result_txt, "w") as f:
            # evaluation
            for seq_idx in self.eval_seqs:
                self.curr_seq_idx = seq_idx
                # Read pose txt
                self.curr_seq_idx = "{:02}".format(seq_idx)
                file_name = "{:02}.txt".format(seq_idx)

                poses_result = self.load_poses_from_txt(result_dir / file_name)
                poses_gt = self.load_poses_from_txt(self.gt_dir / file_name)
                self.result_file_name = result_dir / file_name

                # Pose alignment to first frame
                idx_first_frame = sorted(list(poses_result.keys()))[0]
                pred_first_frame = poses_result[idx_first_frame]
                gt_first_frame = poses_gt[idx_first_frame]
                for pose_idx in poses_result:
                    poses_result[pose_idx] = np.linalg.inv(pred_first_frame) @ poses_result[pose_idx]
                    poses_gt[pose_idx] = np.linalg.inv(gt_first_frame) @ poses_gt[pose_idx]

                # get XYZ
                xyz_gt = []
                xyz_result = []
                for pose_idx in poses_result:
                    xyz_gt.append(poses_gt[pose_idx][:, 3])
                    xyz_result.append(poses_result[pose_idx][:, 3])
                xyz_gt = np.asarray(xyz_gt).transpose(1, 0)
                xyz_result = np.asarray(xyz_result).transpose(1, 0)

                # Plotting
                self.plot_trajectory(poses_gt, poses_result, seq_idx)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KITTI evaluation")
    parser.add_argument("--gt-dir", type=Path, required=True, help="Test dataset directory")
    parser.add_argument("--result", type=Path, required=True, help="Result directory")
    parser.add_argument(
        "--align", type=str, choices=["scale", "scale_7dof", "7dof", "6dof"], default="6dof", help="alignment type"
    )
    parser.add_argument("--seqs", nargs="+", type=int, help="sequences to be evaluated", default=None)
    args = parser.parse_args()

    eval_tool = KittiEvalOdom()
    gt_dir = args.gt_dir
    result_dir = args.result

    # continue_flag = input("Evaluate result in {}? [y/n]".format(result_dir))
    eval_tool.eval(
        gt_dir,
        result_dir,
        alignment=args.align,
        seqs=args.seqs,
    )
