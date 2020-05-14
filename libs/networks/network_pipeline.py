import os,sys
sys.path.append(os.path.abspath('.'))
import torch
import torch.nn as nn 
import torch.nn.functional as F 

import logging

from libs.networks.model_component.resnet import ResNet
from libs.networks.model_component.fpn import FPN
from libs.networks.model_component.rpn import RPN
from libs.networks.model_component.proposal_nms import proposal_layer
from libs.networks.model_component.detection_target import detection_target_layer
from libs.networks.model_component.classifier_head import Classifier
from libs.networks.model_component.mask_head import Mask
from libs.networks.model_component.anchors import generate_pyramid_anchors
from libs.networks.model_component.detection_inference import detection_inference

from libs.networks.tensor_container.rpn_output import RPNOutput
from libs.networks.tensor_container.mrcnn_output import MRCNNOutput
from libs.networks.tensor_container.rpn_target import RPNTarget
from libs.networks.tensor_container.mrcnn_ground_truth import MRCNNGroundTruth

from configs.config import Config

class MaskRCNN(nn.Module):

    def __init__(self, backbone='resnet34'):
        super().__init__()
        assert backbone in ['resnet34', 'resnet50', 'resnet101']
        self.backbone = backbone

        resnet = ResNet(self.backbone, stage5=True)
        
        C1, C2, C3, C4, C5 = resnet.stages()


        fpn_output_channels = 256
        # feature pyramid layers
        self.fpn = FPN(C1, C2, C3, C4, C5, out_channels=fpn_output_channels)

        # Generate Anchors
        # [batch, anchors_num, 4]
        anchors = generate_pyramid_anchors(
            Config.RPN.ANCHOR.SCALES,
            Config.RPN.ANCHOR.RATIOS,
            Config.BACKBONE.SHAPES,
            Config.BACKBONE.STRIDES,
            Config.RPN.ANCHOR.STRIDE,
            Config.TRAINING.BATCH_SIZE
        )
        self.register_buffer('anchors', anchors[0])

        # region proposal layers, [rpn_logits, rpn_probs, rpb_deltas]
        self.rpn = RPN(len(Config.RPN.ANCHOR.RATIOS),
                       Config.RPN.ANCHOR.STRIDE, fpn_output_channels)

        # classifier head, output class probability and box refinement deltas
        self.classifier = Classifier(
            256, Config.HEADS.POOL_SIZE,
            Config.DATASET.IMAGE.SHAPE, Config.DATASET.NUM_CLASSES)

        # mask head, output roi mask
        self.mask = Mask(256, Config.HEADS.MASK.POOL_SIZE,
                         Config.DATASET.IMAGE.SHAPE, Config.DATASET.NUM_CLASSES)

        # load pretrained backbone
        keys = self.state_dict().keys()
        if Config.TRAINING.PRE_MODEL:
            self.load_pretrain(Config.TRAINING.PRE_MODEL)

        # flag for inference 
        self.inference = False
            
    def _rpn_forward(self, images):

        # Feature extraction
        [p2_out, p3_out, p4_out, p5_out, p6_out] = self.fpn(images)

        # Note that P6 is used in RPN, but not in the ROI_Pooling and the sibling head
        rpn_feature_maps = [p2_out, p3_out, p4_out, p5_out, p6_out]
        mrcnn_feature_maps = [p2_out, p3_out, p4_out, p5_out]

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for rpn_feature_map in rpn_feature_maps:
            layer_outputs.append(self.rpn(rpn_feature_map)) # rpn_logits, rpn_probs, rpn_deltas

        
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. 
        # [[rpn_logits_01, rpn_probs_01, rpn_deltas_01],
        #  [rpn_logits_02, rpn_probs_02, rpn_deltas_02]] => 
        # [[rpn_logits_01, rpn_logits_02], [rpn_probs_01, rpn_probs_02], [rpn_deltas_01, rpn_deltas_02]]
        outputs = list(zip(*layer_outputs))

        # Concatenate layer outputs
        outputs = [torch.cat(list(o), dim=1) for o in outputs]

        # store [rpn_logits, rpn_probs, rpn_deltas] in tensorcontainer
        rpn_out = RPNOutput(*outputs)

        return mrcnn_feature_maps, rpn_out


    def forward(self, images, gt=None):
        mrcnn_feature_maps, rpn_out = \
            self._rpn_forward(images)
        batch_size = rpn_out.classes.size()[0]

        if self.training:
            assert gt.boxes.size()[1] != self.anchors.size()[1],\
                'After transformed, the gt_boxes has the same size as the anchors'
        with torch.no_grad():
            if self.inference:
                proposal_count = Config.PROPOSALS.POST_NMS_ROIS.INFERENCE
            else:
                proposal_count = Config.PROPOSALS.POST_NMS_ROIS.TRAINING
            # Generate RoIs, apply_deltas->NMS
            rpn_rois = proposal_layer(
                rpn_out.classes,
                rpn_out.deltas,
                proposal_count=proposal_count,
                nms_threshold=Config.RPN.NMS_THRESHOLD,
                anchors=self.anchors.unsqueeze(0).repeat(batch_size, 1, 1)
            )

        if self.inference:
            return self._inference(mrcnn_feature_maps, rpn_rois)
        # normalize coordinates
        gt.boxes = gt.boxes / Config.RPN.NORM.to(gt.boxes.device)
        
        mrcnn_targets, mrcnn_outs = [], []
        for img_idx in range(0, batch_size):
            with torch.no_grad():
                # Select positive and negative proposals
                # Asign each proposal to one ground truth labels(id, box, mask)
                # Normaly, 1/3 rois are positive rois
                # Also notice that the mask loss is only apply in positive rois,
                # to maintain the same length, the mrcnn_target are fill with 0 in tail.
                rois, mrcnn_target = detection_target_layer(
                        rpn_rois[img_idx], gt.class_ids[img_idx],
                        gt.boxes[img_idx], gt.masks[img_idx])

            # if no rois meet the requirement
            # MRCNNOutput is a tensor_container contain 
            #   1.class_logits. 2.deltas 3.mask
            if rois.nelement() == 0:
                mrcnn_out = MRCNNOutput().to(Config.DEVICE)
                logging.debug('Rois size is empty')
            else:
                # Network Heads
                # Proposal classifier and BBox regressor heads
                rois = rois.unsqueeze(0)
                mrcnn_feature_maps_batch = [x[img_idx].unsqueeze(0).detach()
                                            for x in mrcnn_feature_maps]
                                            
                mrcnn_class_logits_, _, mrcnn_deltas_ = \
                    self.classifier(mrcnn_feature_maps_batch, rois)

                # Create masks
                mrcnn_mask_ = self.mask(mrcnn_feature_maps_batch, rois)

                mrcnn_out = MRCNNOutput(mrcnn_class_logits_,
                                        mrcnn_deltas_, mrcnn_mask_)

            mrcnn_outs.append(mrcnn_out)
            mrcnn_targets.append(mrcnn_target)

        return rpn_out, mrcnn_targets, mrcnn_outs

    def inf(self):
        self.inference = True
        self.eval()


    def _inference(self, mrcnn_feature_maps, rpn_rois):
        '''
        Generate bounding boxes and masks during inference
        Input:
        - mrcnn_feature_maps: list of feature map, [B x C x H1 x W1, B x C x H2 x W2 ...]
        - rpn_rois: [B x POST_NMS_ROIS x 4]

        Return:
        -
        -
        '''
        assert rpn_rois.size()[0] == 1, 'Only support batch=1 during inference'
        with torch.no_grad():
            mrcnn_feature_maps_batch = mrcnn_feature_maps
            _, mrcnn_probs, mrcnn_deltas = self.classifier(mrcnn_feature_maps_batch, rpn_rois[0])

            ## Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
            detections = detection_inference(rpn_rois[0], mrcnn_probs, mrcnn_deltas)
            detection_boxes = detections[:, :4] / Config.RPN.NORM.to(detections.device)
            detection_boxes = detection_boxes.unsqueeze(0)
            mrcnn_mask = self.mask(mrcnn_feature_maps, detection_boxes)
        
        return detections, mrcnn_mask


        

    @staticmethod
    def _prepare_inputs(inputs):
        '''
        Reconstruct the inputs to self-customized data structure
        Input:
            - list:(image, image_metas.to_numpy(), rpn_match, rpn_bbox,
                gt_class_ids, gt_boxes, gt_masks)
        Output:
            - images: B x 3 x H x W 
            - image_metas: window, scale, padding, crop, image_id
            - rpn_target: rpn_match, rpn_bbox
            - gt: gt_class_ids, gt_boxes, gt_masks
        '''
        images = inputs[0].to(Config.DEVICE)
        image_metas = inputs[1]
        rpn_target = (RPNTarget(inputs[2], inputs[3])
                      .to(Config.DEVICE))
        gt = (MRCNNGroundTruth(inputs[4], inputs[5], inputs[6])
              .to(Config.DEVICE))

        return (images, image_metas, rpn_target, gt)

    def load_pretrain(self, filepath):
        EXCLUDE = ['classifier.linear_class.weight',
           'classifier.linear_class.bias',
           'classifier.linear_bbox.weight',
           'classifier.linear_bbox.bias',
           'mask.conv5.weight',
           'mask.conv5.bias']
        if os.path.exists(filepath):
            state_dict = torch.load(filepath)
            state_dict = {key: value for key, value in state_dict.items()
                        if key not in EXCLUDE}
            self.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":
    from configs.mrcnn_config import init_config
    init_config(['./configs/base_config.yml'])
    net = MaskRCNN(backbone='resnet50')
    net.load_pretrain('./pretrained_model/resnet50_imagenet.pth')

    B = Config.TRAINING.BATCH_SIZE
    B, H, W = 2, 768, 576
    num_anchors = 1000
    num_pos_roi = 128
    num_neg_roi = 128
    images = torch.randn((B, 3, H, W))
    rpn_match = torch.zeros((B, num_anchors))
    rpn_match[:, :num_pos_roi] = 1
    rpn_match[:, -num_neg_roi:] = -1
    rpn_bbox = torch.randn((B, num_anchors, 4))
    gt_class_ids = torch.ones((B, num_anchors))
    gt_boxes = torch.ones((B, num_anchors, 4))
    gt_masks = torch.randn((B, 56, 56, num_anchors))
    image_metas = torch.randn((B, 20))

    inputs = (images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks)

    images, images_metas, rpn_target, gt = \
        MaskRCNN._prepare_inputs(inputs)

    rpn_out, mrcnn_target, mrcnn_outs = net(images, gt)
