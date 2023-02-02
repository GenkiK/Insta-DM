from pathlib import Path

import numpy as np
import torch
from torchvision.ops import masks_to_boxes
from tqdm import tqdm


def calc_moments(segms: np.ndarray) -> np.ndarray:
    n, h, w = segms.shape
    x_weights = segms.sum(axis=1)
    m_x = (x_weights @ np.arange(w, dtype=np.float32).reshape(w, 1)).squeeze() / x_weights.sum(axis=1)
    y_weights = segms.sum(axis=2)
    m_y = (y_weights @ np.arange(h, dtype=np.float32).reshape(h, 1)).squeeze() / y_weights.sum(axis=1)
    return np.stack((m_x, m_y), axis=1)


def get_box_centers(boxes: np.ndarray) -> np.ndarray:
    return np.stack(((boxes[:, 2] + boxes[:, 0]) / 2, (boxes[:, 3] + boxes[:, 1]) / 2), axis=1)


MAX_CENTER_OFFSET = 0.15


def remove_noise(segms: np.ndarray, boxes: np.ndarray):
    box_sizes = np.stack((boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]), axis=1)
    x_y_offsets = box_sizes * MAX_CENTER_OFFSET
    moments = calc_moments(segms)
    centers = get_box_centers(boxes)
    center_moment_diffs = (centers - moments).astype(int)
    idxs = np.where(
        (moments[:, 0] < centers[:, 0] - x_y_offsets[:, 0])
        | (moments[:, 1] < centers[:, 1] - x_y_offsets[:, 1])
        | (moments[:, 0] > centers[:, 0] + x_y_offsets[:, 0])
        | (moments[:, 1] > centers[:, 1] + x_y_offsets[:, 1])
    )[0]
    for idx in idxs:
        box_w, box_h = box_sizes[idx]
        x0, y0, x1, y1 = boxes[idx].astype(int) - np.tile(center_moment_diffs[idx], 2)
        y0 -= (box_h * 0.1).astype(int)
        y1 += (box_h * 0.1).astype(int)
        x0 += (box_w * 0.1).astype(int)
        x1 += (box_w * 0.1).astype(int)
        segms[idx, : max(y0, 0), :] = 0.0
        segms[idx, :, : max(x0, 0)] = 0.0
        segms[idx, y1:, :] = 0.0
        segms[idx, :, x1:] = 0.0
    return segms


def modify_segms_boxes(segms: np.ndarray, boxes: np.ndarray):
    segms = remove_noise(segms, boxes)
    return segms


def load_segms_and_categories(path: Path) -> tuple[np.ndarray, np.ndarray]:
    npz = np.load(path)
    segms = npz["masks"].astype(np.float32)
    categories = npz["labels"].astype(np.int8)
    return segms, categories


if __name__ == "__main__":
    root_dir = Path("/home/gkinoshita/workspace/Insta-DM/kitti_256/")
    root_segm_dir = root_dir / "segmentation_OneFormer_modified"
    root_segm_save_dir = root_dir / "segmentation_postprocess"
    root_segm_save_dir.mkdir(parents=False, exist_ok=True)

    for scene_dir in tqdm(root_segm_dir.iterdir()):
        save_dir = root_segm_save_dir / scene_dir.name
        save_dir.mkdir(parents=False, exist_ok=True)

        for segm_path in scene_dir.glob("*.npz"):
            # backgroundを含んでいることに注意
            segms_with_bg, categories_with_bg = load_segms_and_categories(segm_path)
            if categories_with_bg.shape[0] > 1:
                boxes = masks_to_boxes(torch.from_numpy(segms_with_bg[1:])).numpy()
                segms = modify_segms_boxes(segms_with_bg[1:], boxes)
                segms_with_bg = np.concatenate((segms_with_bg[[0]], segms), axis=0)
            np.savez(save_dir / segm_path.name, masks=segms_with_bg, labels=categories_with_bg)
