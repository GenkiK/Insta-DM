import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from imageio import imread
from skimage.transform import resize as imresize
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Script for EgoPoseNet testing with corresponding groundTruth",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--data_dir", default=None, required=True, type=Path, help="path to dataset")
parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument(
    "--output_dir",
    default=None,
    required=True,
    type=Path,
    help="Output directory for saving predictions in a big 3D numpy file",
)
parser.add_argument("--pretrained-posenet", required=True, type=str, help="pretrained EgoPoseNet path")
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")
parser.add_argument("--no-resize", action="store_true", help="no resizing is done")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def euler2mat(angle):
    """
    Convert euler angles to rotation matrix.
    Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    angle = angle.view((-1, 3))
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack([cosz, -sinz, zeros, sinz, cosz, zeros, zeros, zeros, ones], dim=1).view(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny, zeros, ones, zeros, -siny, zeros, cosy], dim=1).view(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros, zeros, cosx, -sinx, zeros, sinx, cosx], dim=1).view(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat.squeeze()


def load_tensor_image(filename: Path, args):
    img = imread(filename).astype(np.float32)
    h, w, _ = img.shape
    if (not args.no_resize) and (h != args.img_height or w != args.img_width):
        img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    # tensor_img = ((torch.from_numpy(img).unsqueeze(0) / 255 - 0.5) / 0.5).to(device)
    tensor_img = (torch.from_numpy(img).unsqueeze(0) / 255).to(device)
    return tensor_img


@torch.no_grad()
def main():
    args = parser.parse_args()
    sys.path.insert(1, os.path.join(sys.path[0], ".."))
    import models

    pose_net = models.EgoPoseNet().to(device)
    pose_weights = torch.load(args.pretrained_posenet)
    pose_net.load_state_dict(pose_weights["state_dict"])
    pose_net.eval()

    data_dir: Path = args.data_dir
    test_files: list[str] = []
    with open(args.dataset_list, "r") as f:
        for line in f:
            line = line.rstrip()
            if len(line) > 0:
                test_files.append(line)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for test_filename in test_files:
        image_dir = data_dir / "sequences" / test_filename / "image_2"
        img_filepaths: list[Path] = sorted(image_dir.glob("*.png"))

        # for training dataset
        # image_dir = data_dir / "image" / f"{test_filename}_2"
        # img_filepaths: list[Path] = sorted(image_dir.glob("*.jpg"))

        global_pose = np.eye(4)
        save_mat = np.zeros((len(img_filepaths), 12))
        save_mat[0] = global_pose[:3, :].reshape(1, 12)

        prev_img = load_tensor_image(img_filepaths[0], args)
        for i in tqdm(range(1, len(img_filepaths))):
            curr_img = load_tensor_image(img_filepaths[i], args)
            # pred_pose = pose_net(prev_img, curr_img).cpu()[0]
            pred_pose = pose_net(curr_img, prev_img).cpu()[0]
            pred_pose_mat = np.concatenate(
                (euler2mat(pred_pose[3:]).numpy(), pred_pose[:3].numpy().reshape(3, 1)), axis=1
            )
            pred_pose_mat = np.concatenate((pred_pose_mat, np.array([[0, 0, 0, 1]])), axis=0)
            # global_pose = global_pose @ np.linalg.inv(pred_pose_mat)
            global_pose = global_pose @ pred_pose_mat
            save_mat[i] = global_pose[:3, :].reshape(1, 12)
            prev_img = curr_img
        np.savetxt(output_dir / f"{test_filename}.txt", save_mat, delimiter=" ")


if __name__ == "__main__":
    main()
