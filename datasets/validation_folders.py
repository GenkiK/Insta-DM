from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from imageio import imread

from datasets.my_sequence_folders import convert_poses_abs2rel

# def crawl_folders(folders_list):
#     imgs = []
#     depth = []
#     segs = []
#     for folder in folders_list:
#         current_imgs = sorted(folder.files("*.jpg"))
#         imgs.extend(current_imgs)
#         for img in current_imgs:
#             # Fetch depth file
#             dd = img.dirname() / (img.name[:-4] + ".npy")
#             assert dd.isfile(), "depth file {} not found".format(str(dd))
#             depth.append(dd)
#             # Fetch segmentation file
#             ss = folder.dirname().parent / "segmentation" / folder.basename() / (img.name[:-4] + ".npy")
#             assert ss.isfile(), "segmentation file {} not found".format(str(ss))
#             segs.append(ss)
#     return imgs, depth, segs


def make_paths_and_poses(root_dir: Path, scene_names: list[str]):
    img_paths = []
    depth_paths = []
    segm_paths = []
    rel_poses = []
    for scene_name in scene_names:
        scene_img_dir = root_dir / "image" / scene_name
        scene_depth_dir = root_dir / "depth" / scene_name
        scene_segm_dir = root_dir / "segmentation" / scene_name

        current_img_paths = sorted(scene_img_dir.glob("*.jpg"))
        img_paths.extend(current_img_paths)

        current_depth_paths = sorted(scene_depth_dir.glob("*.npy"))
        depth_paths.extend(current_depth_paths)

        current_segm_paths = sorted(scene_segm_dir.glob("*.npy"))
        segm_paths.extend(current_segm_paths)

        scene_poses: list[torch.Tensor] = []
        with open(root_dir / "poses" / f"{scene_name[:-2]}.txt", "r") as f:
            for line in f:
                line = line.rstrip("\n")
                if len(line) == 0:
                    continue
                scene_poses.append(torch.Tensor([float(elem) for elem in line.split(" ")]).view(3, 4))
        scene_rel_poses = convert_poses_abs2rel(scene_poses)
        rel_poses.extend(scene_rel_poses)
    return img_paths, depth_paths, segm_paths, rel_poses


def load_as_float(path):
    return imread(path).astype(np.float32)


class ValidationSet(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
    root/scene_1/0000000.jpg
    root/scene_1/0000000.npy
    root/scene_1/0000001.jpg
    root/scene_1/0000001.npy
    ..
    root/scene_2/0000000.jpg
    root/scene_2/0000000.npy
    .

    transform functions must take in a list images and a numpy array which can be None
    """

    def __init__(self, root_dir: Path, transform=None):
        scenes_txt_path = root_dir / "val.txt"
        scene_names: list[str] = []
        with open(scenes_txt_path, "r") as f:
            for scene_name in f:
                scene_name = scene_name.rstrip("\n")
                if len(scene_name) == 0:
                    continue
            scene_names.append(scene_name)
        self.img_paths, self.depth_paths, self.segm_paths, self.rel_poses = make_paths_and_poses(root_dir, scene_names)
        self.transform = transform

    def __getitem__(self, idx):
        img = load_as_float(self.img_paths[idx])  # H x W x 3
        depth = np.load(self.depth_paths[idx]).astype(np.float32)  # H x W
        segm = torch.from_numpy(np.load(self.segm_paths[idx]).astype(np.float32))  # N x H X W
        rel_pose = self.rel_poses[idx]

        # Sum segmentation for every mask
        segm = segm.sum(dim=0, keepdim=False).clamp(min=0.0, max=1.0)  # H x W

        if self.transform is not None:
            img, _ = self.transform([img])
            img = img[0]
        return img, depth, segm, rel_pose

    def __len__(self):
        return len(self.img_paths)
