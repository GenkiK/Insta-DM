from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from imageio import imread

from custom_transforms_val import Compose


def make_paths(root_dir: Path, scene_names: list[str]):
    img_paths = []
    depth_paths = []
    segm_paths = []
    for scene_name in scene_names:
        scene_img_dir = root_dir / "image" / scene_name
        scene_segm_dir = root_dir / "segmentation_OneFormer_modified" / scene_name

        current_img_paths = sorted(scene_img_dir.glob("*.jpg"))
        img_paths.extend(current_img_paths)

        current_depth_paths = sorted(scene_img_dir.glob("*.npy"))
        depth_paths.extend(current_depth_paths)

        current_segm_paths = sorted(scene_segm_dir.glob("*.npz"))
        segm_paths.extend(current_segm_paths)
    return img_paths, depth_paths, segm_paths


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

    def __init__(self, root_dir: Path, transform: Compose | None = None):
        scenes_txt_path = root_dir / "val.txt"
        with open(scenes_txt_path, "r") as f:
            scene_names = f.read().splitlines()
        self.img_paths, self.depth_paths, self.segm_paths = make_paths(root_dir, scene_names)
        self.transform = transform

    def __getitem__(self, idx):
        img = load_as_float(self.img_paths[idx])  # H x W x 3
        depth = np.load(self.depth_paths[idx]).astype(np.float32)  # H x W
        segm = torch.from_numpy(np.load(self.segm_paths[idx])["masks"].astype(np.float32))

        # Sum segmentation for every mask
        segm = segm.sum(dim=0, keepdim=False).clamp(min=0.0, max=1.0)  # H x W

        if self.transform is not None:
            img = self.transform([img])
            img = img[0]
        return img, depth, segm

    def __len__(self):
        return len(self.img_paths)
