"""
# (+) customized inputs: images (src/tgt), segmentation mask (src/tgt), intrinsics

"""
from __future__ import division

import random

import cv2
import numpy as np
import torch

from rigid_warp import euler2mat, mat2euler

"""Set of transform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations."""


def rotate_euler(rad_angles: torch.Tensor, axes: str = "yz"):
    """
    rad_angles: torch.Size([N, 3]) → rad_angles: torch.Size(3)では？
    """
    rad_angles = rad_angles.view((-1, 3))
    axis_str2num = {"x": 0, "y": 1, "z": 2}
    for axis_str in list(axes):
        axis_idx = axis_str2num[axis_str]
        rad_angles[:, axis_idx] = -rad_angles[:, axis_idx]
    return rad_angles.squeeze()


class ComposeWithEgoPose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, segms, intrinsics, poses_t_1_to_t, poses_t_to_t_1):
        for transform in self.transforms:
            images, segms, intrinsics, poses_t_1_to_t, poses_t_to_t_1 = transform(
                images, segms, intrinsics, poses_t_1_to_t, poses_t_to_t_1
            )
        return images, segms, intrinsics, poses_t_1_to_t, poses_t_to_t_1


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, segms, intrinsics):
        for transform in self.transforms:
            images, segms, intrinsics = transform(images, segms, intrinsics)
        return images, segms, intrinsics


class NormalizeWithEgoPose:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, segms, intrinsics, poses_t_1_to_t, poses_t_to_t_1):
        for image in images:
            for img_ch, mean_ch, std_ch in zip(image, self.mean, self.std):
                img_ch.sub_(mean_ch).div_(std_ch)
        return images, segms, intrinsics, poses_t_1_to_t, poses_t_to_t_1


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, segms, intrinsics):
        for image in images:
            for img_ch, mean_ch, std_ch in zip(image, self.mean, self.std):
                img_ch.sub_(mean_ch).div_(std_ch)
        return images, segms, intrinsics


class ArrayToTensorWithEgoPose:
    """
    Converts a list of ndarray (HxWxC) along with a intrinsics matrix to a list of FloatTensor of shape (CxHxW) with a intrinsics tensor.
    """

    def __call__(self, images, segms, intrinsics, poses_t_1_to_t, poses_t_to_t_1):
        img_tensors = []
        seg_tensors = []
        for img in images:
            img = np.transpose(img, (2, 0, 1))  # put it from HWC to CHW format
            img_tensors.append(torch.from_numpy(img).float() / 255)  # handle numpy array
        for segm in segms:
            segm = np.transpose(segm, (2, 0, 1))
            seg_tensors.append(torch.from_numpy(segm).float())
        return img_tensors, seg_tensors, intrinsics, poses_t_1_to_t, poses_t_to_t_1


class ArrayToTensor:
    """
    Converts a list of ndarray (HxWxC) along with a intrinsics matrix to a list of FloatTensor of shape (CxHxW) with a intrinsics tensor.
    """

    def __call__(self, images, segms, intrinsics):
        img_tensors = []
        seg_tensors = []
        for img in images:
            img = np.transpose(img, (2, 0, 1))  # put it from HWC to CHW format
            img_tensors.append(torch.from_numpy(img).float() / 255)  # handle numpy array
        for segm in segms:
            segm = np.transpose(segm, (2, 0, 1))
            seg_tensors.append(torch.from_numpy(segm).float())
        return img_tensors, seg_tensors, intrinsics


class RandomHorizontalFlipWithEgoPose:
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, images, segms, intrinsics, poses_t_1_to_t, poses_t_to_t_1):
        if random.random() < 0.5:
            output_images = [np.copy(np.fliplr(img)) for img in images]
            output_segms = [np.copy(np.fliplr(segm)) for segm in segms]

            w = output_images[0].shape[1]
            intrinsics[0, 2] = w - intrinsics[0, 2]

            # assert len(rel_poses) == 2 and rel_poses[0].shape == (3, 4)
            for i in range(len(poses_t_1_to_t)):
                flip_rot_t_1_to_t = poses_t_1_to_t[i][:, :3]
                flip_rot_t_1_to_t = euler2mat(rotate_euler(mat2euler(flip_rot_t_1_to_t)))
                flip_trans_t_1_to_t = poses_t_1_to_t[i][:, 3]
                flip_trans_t_1_to_t[0] = -flip_trans_t_1_to_t[0]
                poses_t_1_to_t[i] = torch.cat((flip_rot_t_1_to_t, flip_trans_t_1_to_t.view(3, 1)), dim=1)

                flip_rot_t_to_t_1 = poses_t_to_t_1[i][:, :3]
                flip_rot_t_to_t_1 = euler2mat(rotate_euler(mat2euler(flip_rot_t_to_t_1)))
                flip_trans_t_to_t_1 = poses_t_to_t_1[i][:, 3]
                flip_trans_t_to_t_1[0] = -flip_trans_t_to_t_1[0]
                poses_t_to_t_1[i] = torch.cat((flip_rot_t_to_t_1, flip_trans_t_to_t_1.view(3, 1)), dim=1)
        else:
            output_images = images
            output_segms = segms
        return output_images, output_segms, intrinsics, poses_t_1_to_t, poses_t_to_t_1


class RandomHorizontalFlip:
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, images, segms, intrinsics):
        if random.random() < 0.5:
            output_images = [np.copy(np.fliplr(img)) for img in images]
            output_segms = [np.copy(np.fliplr(segm)) for segm in segms]

            w = output_images[0].shape[1]
            intrinsics[0, 2] = w - intrinsics[0, 2]
        else:
            output_images = images
            output_segms = segms
        return output_images, output_segms, intrinsics


class RandomScaleCropWithEgoPose:
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images, segms, intrinsics, poses_t_1_to_t, poses_t_to_t_1):
        h, w, _ = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1, 1.15, 2)
        scaled_h, scaled_w = int(h * y_scaling), int(w * x_scaling)

        intrinsics[0] *= x_scaling
        intrinsics[1] *= y_scaling

        # scaled_images = [np.array(Image.fromarray(img.astype(np.uint8)).resize((scaled_h, scaled_w), resample=2)) for img in images]
        scaled_images = [
            cv2.resize(img.astype(np.uint8), (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR) for img in images
        ]
        scaled_segms = [
            cv2.resize(segm, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST) for segm in segms
        ]  # 이 부분에서 1채널 세그먼트 [256 x 832 x 1] >> [256 x 832]로 변환됨!

        offset_y = np.random.randint(scaled_h - h + 1)
        offset_x = np.random.randint(scaled_w - w + 1)
        cropped_images = [img[offset_y : offset_y + h, offset_x : offset_x + w] for img in scaled_images]
        cropped_segms = [segm[offset_y : offset_y + h, offset_x : offset_x + w] for segm in scaled_segms]

        intrinsics[0, 2] -= offset_x
        intrinsics[1, 2] -= offset_y

        return cropped_images, cropped_segms, intrinsics, poses_t_1_to_t, poses_t_to_t_1


class RandomScaleCrop:
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images, segms, intrinsics):
        h, w, _ = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1, 1.15, 2)
        scaled_h, scaled_w = int(h * y_scaling), int(w * x_scaling)

        intrinsics[0] *= x_scaling
        intrinsics[1] *= y_scaling

        # scaled_images = [np.array(Image.fromarray(img.astype(np.uint8)).resize((scaled_h, scaled_w), resample=2)) for img in images]
        scaled_images = [
            cv2.resize(img.astype(np.uint8), (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR) for img in images
        ]
        scaled_segms = [
            cv2.resize(segm, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST) for segm in segms
        ]  # 이 부분에서 1채널 세그먼트 [256 x 832 x 1] >> [256 x 832]로 변환됨!

        offset_y = np.random.randint(scaled_h - h + 1)
        offset_x = np.random.randint(scaled_w - w + 1)
        cropped_images = [img[offset_y : offset_y + h, offset_x : offset_x + w] for img in scaled_images]
        cropped_segms = [segm[offset_y : offset_y + h, offset_x : offset_x + w] for segm in scaled_segms]

        intrinsics[0, 2] -= offset_x
        intrinsics[1, 2] -= offset_y

        return cropped_images, cropped_segms, intrinsics
