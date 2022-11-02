"""
# (+) customized inputs: images (src/tgt), segmentation mask (src/tgt), intrinsics

"""
from __future__ import division

import random

import cv2
import numpy as np
import torch
from PIL import Image

"""Set of transform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations."""


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, segms, intrinsics):
        for transform in self.transforms:
            images, segms, intrinsics = transform(images, segms, intrinsics)
        return images, segms, intrinsics


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, segms, intrinsics):
        for image in images:
            for img_ch, mean_ch, std_ch in zip(image, self.mean, self.std):
                img_ch.sub_(mean_ch).div_(std_ch)
        return images, segms, intrinsics


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

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


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, images, segms, intrinsics):
        if random.random() < 0.5:
            output_intrinsics = np.copy(intrinsics)
            output_images = [np.copy(np.fliplr(img)) for img in images]
            output_segms = [np.copy(np.fliplr(segm)) for segm in segms]

            w = output_images[0].shape[1]
            output_intrinsics[0, 2] = w - output_intrinsics[0, 2]
        else:
            output_images = images
            output_segms = segms
            output_intrinsics = intrinsics
        return output_images, output_segms, output_intrinsics


class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images, segms, intrinsics):
        output_intrinsics = np.copy(intrinsics)

        h, w, _ = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1, 1.15, 2)
        scaled_h, scaled_w = int(h * y_scaling), int(w * x_scaling)

        output_intrinsics[0] *= x_scaling
        output_intrinsics[1] *= y_scaling

        scaled_images = [np.array(Image.fromarray(img).resize((scaled_h, scaled_w), resample=2)) for img in images]
        scaled_segms = [
            cv2.resize(segm, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST) for segm in segms
        ]  # 이 부분에서 1채널 세그먼트 [256 x 832 x 1] >> [256 x 832]로 변환됨!

        offset_y = np.random.randint(scaled_h - h + 1)
        offset_x = np.random.randint(scaled_w - w + 1)
        cropped_images = [img[offset_y : offset_y + h, offset_x : offset_x + w] for img in scaled_images]
        cropped_segms = [segm[offset_y : offset_y + h, offset_x : offset_x + w] for segm in scaled_segms]

        output_intrinsics[0, 2] -= offset_x
        output_intrinsics[1, 2] -= offset_y

        return cropped_images, cropped_segms, output_intrinsics
