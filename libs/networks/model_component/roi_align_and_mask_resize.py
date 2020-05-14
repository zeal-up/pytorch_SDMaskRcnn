'''
The PyTorch roi_align function take unnormalized boxed coordinate.
However, we normally use normalized coordinate in our code.
This file provide function to wrap PyTorch roi_align.
Roi align function also used to resize gt mask to match the proposal bounding box.
'''
import torch 
import torchvision.ops.roi_align


def roi_align(images, boxes, output_size):
    '''
    Input:
    - images: C x H x W 
    - boxes: Nx4, normalized roi coordinates, in (y1, x1, y2, x2) format
    - output_size: (int, int), the output size

    Return:
    - roi_region: N x C x H x W, roi_pooled feature_map
    '''
    if images.ndim == 2: # for mask, squeeze a channel dimension
        images = images.unsqueeze(0)
    images = images.unsqueeze(0) # torchvision.ops.roi_align accept a batch input # 1 x C x H x W
    N = boxes.size()[0]
    box_indices = [0] * N 

    y1, x1, y2, x2 = boxes.chunk(4, dim=1) 
    h, w = images.size()[-2:]
    y1, x1, y2, x2 = (y1*h, x1*w, y2*h, x2*w)

    box_indices = torch.tensor(box_indices).to(y1)

    # unnormalized boxes, since the torchvision.ops.roi_align accept unnormalized coordinate 
    # in (class_indices, y1, x2, y2) format.
    boxes_unnormalized = torch.cat((box_indices.unsqueeze(1), x1, y1, x2, y2), dim=1) # N x 5

    roi_region = torchvision.ops.roi_align(images, boxes_unnormalized, output_size, spatial_scale=1., sampling_ratio=-1) # N x C x H x W


    return roi_region



def batched_roi_align(images, boxes, output_size):
    '''
    Input:
    - images: B x C x H x W 
    - boxes: [N1 x 4, N2x4, N3x4, NBx4], list of boxes, normalized roi coordinates, in (y1, x1, y2, x2) format
    - output_size: (int, int), the output size

    Return:
    - roi_region: [N1 x C x H x W, N2xCxHxW, N3xCxHxW, NBxCxHxW] list of feature
    '''
    B = images.size()[0]
    boxes_list = boxes
    assert len(boxes) == B
    assert len(images.size()) == 4
    box_indices = []
    for i, b in enumerate(boxes):
        box_indices.extend([i]*b.size()[0])
    box_indices = torch.tensor(box_indices, dtype=float) # [0,0, 1,1,1, 2,2,2,2]
    boxes = torch.cat(boxes, dim=0) # B1+B2... X 4
    y1, x1, y2, x2 = boxes.chunk(4, dim=1) 
    h, w = images.size()[-2:]
    y1, x1, y2, x2 = (y1*h, x1*w, y2*h, x2*2)

    box_indices = box_indices.to(y1)

    # unnormalized boxes, since the torchvision.ops.roi_align accept unnormalized coordinate 
    # in (class_indices, y1, x2, y2) format.
    boxes_unnormalized = torch.cat((box_indices.unsqueeze(1), x1, y1, x2, y2), dim=1) # N x 5

    roi_region = torchvision.ops.roi_align(images, boxes_unnormalized, output_size, spatial_scale=1., sampling_ratio=-1) # BxN x C x H x W

    offset = 0
    roi_region_list = []
    for i in range(B):
        N = boxes_list[i].size()[0]
        idx = range(offset, offset+N)
        offset += N 
        roi_region_list.append(roi_region[idx, :, :, :])


    return roi_region_list

if __name__ == "__main__":
    # images = torch.rand((2, 3, 50, 50))
    # boxes1 = torch.tensor([
    #     [0.1, 0.1, 0.7, 0.7],
    #     [-0.1, -0.1, 1.1, 1.1]
    # ])
    # boxes2 = torch.tensor([
    #     [0.1, 0.1, 0.7, 0.7],
    #     [-0.1, -0.1, 1.1, 1.1],
    #     [-0.1, -0.1, 1.1, 1.1]
    # ])
    # boxes_list = [boxes1, boxes2]
    # output_size = (28, 28)

    # roi_region = batched_roi_align(images, boxes_list, output_size)
    # print(len(roi_region))
    # for region in roi_region:
    #     print(region.size())

    images = torch.rand((3, 50, 50))
    boxes = torch.tensor([
        [0.1, 0.1, 0.7, 0.7],
        [-0.1, -0.1, 1.1, 1.1]
    ])
    roi_region = roi_align(images, boxes, (28, 28))
    print(roi_region.size())