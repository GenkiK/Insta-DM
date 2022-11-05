from __future__ import division

import numpy as np
import torch

"""Set of transform random routines that takes list of inputs as arguments in order to have random but coherent transformations."""


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images):
        for t in self.transforms:
            images = t(images)
        return images


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images):
        for image in images:
            for img_ch, mean_ch, std_ch in zip(image, self.mean, self.std):
                img_ch.sub_(mean_ch).div_(std_ch)
        return images


class ArrayToTensor(object):
    """
    Converts a list of ndarray (HxWxC) matrix to a list of FloatTensor of shape (CxHxW) with a intrinsics tensor.
    """

    def __call__(self, images):
        tensors = []
        for img in images:
            # put it from HWC to CHW format
            img = np.transpose(img, (2, 0, 1))
            # handle numpy array
            tensors.append(torch.from_numpy(img).float() / 255)
        return tensors
