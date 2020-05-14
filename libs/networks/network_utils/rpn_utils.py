"""
Mask R-CNN
Common utility functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import math
import random
import warnings

import numpy as np
import scipy.misc
import scipy.ndimage
import skimage.transform
import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.

    Args:
        boxes: [batch_size, N, 4] where each row is y1, x1, y2, x2
        deltas: [batch_size, N, 4] where each row is [dy, dx, log(dh), log(dw)]

    Returns:
        results: [batch_size, N, 4], where each row is [y1, x1, y2, x2]
    """
    # Convert to y, x, h, w
    height = boxes[:, :, 2] - boxes[:, :, 0]
    width = boxes[:, :, 3] - boxes[:, :, 1]
    center_y = boxes[:, :, 0] + 0.5 * height
    center_x = boxes[:, :, 1] + 0.5 * width
    # Apply deltas
    center_y = center_y + deltas[:, :, 0] * height
    center_x = center_x + deltas[:, :, 1] * width
    height = height * torch.exp(deltas[:, :, 2])
    width = width * torch.exp(deltas[:, :, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], dim=2)
    return result



def clip_boxes(boxes, window, squeeze=False):
    """
    Clip to image boundaries. 
    boxes: [B, N, 4] each col is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2, 
            Config.RPN.CLIP_WINDOW = np.array([0, 0, height, width]).astype(np.float32)

    return:
        cliped_boxes: [batch, N, (y1, x1, y2, x2)]
    """
    if squeeze:
        boxes = boxes.unsqueeze(0)
    boxes = torch.stack(
        [boxes[:, :, 0].clamp(float(window[0]), float(window[2])), # y1
         boxes[:, :, 1].clamp(float(window[1]), float(window[3])), # x1 
         boxes[:, :, 2].clamp(float(window[0]), float(window[2])), # y2
         boxes[:, :, 3].clamp(float(window[1]), float(window[3]))], 2) # x2
    if squeeze:
        boxes = boxes.squeeze(0)
    return boxes

def unique1d(tensor):
    if tensor.size()[0] == 0 or tensor.size()[0] == 1:
        return tensor
    tensor = tensor.sort()[0]
    unique_bool = tensor[1:] != tensor [:-1]
    first_element = torch.tensor([True]).to(tensor.device)
    unique_bool = torch.cat((first_element, unique_bool),dim=0)
    return tensor[unique_bool.data]

def set_intersection(tensor1, tensor2):
    """Intersection of elements present in tensor1 and tensor2.
    Note: it only works if elements are unique in each tensor.
    """
    aux = torch.cat((tensor1, tensor2), dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).detach()]