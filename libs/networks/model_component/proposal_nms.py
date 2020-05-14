import torch
import numpy as np 
from torchvision.ops.boxes import batched_nms
# nms_wrapper is written in cuda
from libs.networks.network_utils.rpn_utils import apply_box_deltas, clip_boxes
from configs.config import Config



def proposal_layer(scores, deltas, proposal_count, nms_threshold, anchors):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinment to anchors. Proposals are zero padded.

    Inputs:
        scores: [batch, anchors, 2], the bounding box softmax scores returned in RPN
                Notice that batch==1 in this function. The torchvision.ops.nms only support batch=1
                It's rpn_probs: [batch, anchors, (bg prob, fg prob)]
        
        deltas(rpn_bbox): [batch, anchors, (dy, dx, log(dh), log(dw))]
        proposal_count: Config.POST_NMS_ROIS, training=2000, inference=1000
        nums_threshold: Config.RPN.NMS_THRESHOLD, default=0.7
        anchors: [batch, num_anchors, 4], the initial anchor coordinates.(unnormalized)

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
    scores = scores[:, :, 1]

    deltas = deltas * Config.RPN.BBOX_STD_DEV_GPU.to(deltas.device)

    # Improve performance by trimming to top anchors by score
    # and doing the rest on the smaller subset.
    # A.K.A limit the RoIs before NMS
    pre_nms_limit = min(Config.PROPOSALS.PRE_NMS_LIMIT, anchors.shape[1])

    scores, order = scores.topk(pre_nms_limit, dim=1)
    
    # gather the topk anchors and deltas
    order = order.unsqueeze(2).expand(-1, -1, 4)
    deltas = deltas.gather(1, order)
    anchors = anchors.gather(1, order)

    # Apply deltas to anchors to get refined anchors.
    # [batch, N, (y1, x1, y2, x2)]
    boxes = apply_box_deltas(anchors, deltas)

    # Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
    boxes = clip_boxes(boxes, Config.RPN.CLIP_WINDOW)
    # Config.RPN.CLIP_WINDOW = np.array([0, 0, height, width]).astype(np.float32)


    # Non-max suppression
    B, N = scores.size()
    ind = torch.tensor(range(B), dtype=torch.long)
    ind = ind.view(-1, 1)
    ind = ind.repeat(1, N).view(-1)
    boxes = boxes.view(-1, 4) # B*N
    keeps = batched_nms(
        boxes,
        scores.view(-1),
        ind,
        nms_threshold)
    # np.savetxt('./keeps.txt', keeps.detach().cpu().numpy())

    nms_boxes = []
    for i in range(B):
        high_ind = (i+1) * N 
        ind_cur = keeps[keeps<high_ind][:proposal_count]
        if len(ind_cur) < proposal_count:
            ind_cur_pad = torch.empty((proposal_count,)).fill_(ind_cur[-1]).to(ind_cur)
            ind_cur_pad[:len(ind_cur)] = ind_cur
            ind_cur = ind_cur_pad
            # ind_cur = torch.nn.functional.pad(ind_cur, ((0, proposal_count-len(ind_cur))), mode='constant', constant_value=ind_cur[-1])
        
        keeps = keeps[keeps>=high_ind]
        assert len(ind_cur) == proposal_count, 'illegal ind_cur length'
        nms_boxes.append(boxes[ind_cur]) # proposal_count x 4 

    nms_boxes = torch.stack(nms_boxes, dim=0) # B x proposal_count x 4

    # Normalize dimensions to range of 0 to 1.
    normalized_boxes = nms_boxes/Config.RPN.NORM.to(nms_boxes.device)

    return normalized_boxes
