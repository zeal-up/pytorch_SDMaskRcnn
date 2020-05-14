"""Implements ROI Pooling on multiple levels of the feature pyramid."""

import os,sys 
sys.path.append(os.path.abspath('.'))
import torch
from libs.networks.model_component.roi_align_and_mask_resize import roi_align


def pyramid_roi_align(boxes, feature_maps, pool_size, image_shape):
    """
    Implements ROI Pooling on multiple levels of the feature pyramid.
    In single batch mode. All input params is only one batch.
    The accept params can has dimension=1 in the first dim. 
    But the return feature map will always squeeze the first dimension.

    

    Inputs:
    - boxes: [num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [channels, height, width]
    - pool_size: (height, width) of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, channels]. Shape of input image in pixels

    Output:
    - Pooled regions in the shape: [num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
    # Assign each ROI to a level in the pyramid based on the ROI area.
    boxes = boxes.squeeze(0)
    feature_maps = [feature_map.squeeze(0) for feature_map in feature_maps]
    y1, x1, y2, x2 = boxes.chunk(4, dim=1)
    h = (y2 - y1).float()
    w = (x2 - x1).float()

    # Equation 1 in the Feature Pyramid Networks paper. Account for
    # the fact that our coordinates are normalized here.
    # e.g. a 224x224 ROI (in pixels) maps to P4; 
    # and a 112x112 ROI (in pixels) maps to P3
    image_area = torch.tensor([float(image_shape[0]*image_shape[1])],
                              dtype=torch.float32).to(h.device)
    roi_level = 4 + torch.log2(torch.sqrt(h*w)/(224.0/torch.sqrt(image_area)))
    roi_level = roi_level.round().int()
    roi_level = roi_level.clamp(2, 5)

    # Loop through levels and apply ROI pooling to each. P2 to P5.
    pooled = []
    box_to_level = []
    for i, level in enumerate(range(2, 6)):
        ix = roi_level == level
        if not ix.any():
            continue
        ix = torch.nonzero(ix)[:, 0]
        level_boxes = boxes[ix.detach(), :]

        # Keep track of which box is mapped to which level
        box_to_level.append(ix.detach())

        # Stop gradient propogation to ROI proposals
        level_boxes = level_boxes.detach()

        # Crop and Resize
        # From Mask R-CNN paper: "We sample four regular locations, so
        # that we can evaluate either max or average pooling. In fact,
        # interpolating only a single value at each bin center (without
        # pooling) is nearly as effective."
        #
        # Here we use the simplified approach of a single value per bin,
        # which is how it's done in tf.crop_and_resize()
        # Result: [batch * num_boxes, pool_height, pool_width, channels]

        # Apply RoI Align
        pooled_features = roi_align(feature_maps[i], level_boxes, pool_size)
        pooled.append(pooled_features)

    # Pack pooled features into one tensor
    pooled = torch.cat(pooled, dim=0)

    # Pack box_to_level mapping into one array and add another
    # column representing the order of pooled boxes
    box_to_level = torch.cat(box_to_level, dim=0)

    # Rearrange pooled features to match the order of the original boxes
    _, box_to_level = torch.sort(box_to_level)
    pooled = pooled[box_to_level, :, :]

    return pooled # N x C x H x W 


if __name__ == "__main__":
    images1 = torch.rand((3, 112, 112))
    images2 = torch.rand((3, 224, 224))
    images3 = torch.rand((3, 448, 448))
    images4 = torch.rand((3, 500, 500))
    images = [images1, images2, images3, images4]
    boxes = torch.tensor([
        [0.1, 0.1, 0.7, 0.7],
        [-0.1, -0.1, 1.1, 1.1], 
        [0.2, 0.2, 0.9, 0.9],
        [0.3, 0.3, 0.8, 0.8],
        [0.4, 0.4, 0.6, 0.6]
    ])

    pooled = pyramid_roi_align(boxes, images, pool_size=(28, 28), image_shape=[224, 224])
    print(pooled.size())
