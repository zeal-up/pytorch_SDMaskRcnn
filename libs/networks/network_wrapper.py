import torch
import torch.nn as nn 
from libs.networks.Losses.losses import compute_losses
from libs.networks.network_pipeline import MaskRCNN
import imageio

class Network_Wrapper(MaskRCNN):

    def forward(self, inputs):
        images, image_metas, rpn_target, gt = self._prepare_inputs(inputs)

        rpn_out, mrcnn_targets, mrcnn_outs = super().forward(images, gt)
        loss_structure = compute_losses(rpn_target, rpn_out, mrcnn_targets, mrcnn_outs)

        # loss_structure is a data structure store the whole 5 losses during trainging
        # call loss_structure.total.backward() to train
        return loss_structure.rpn_class,\
            loss_structure.rpn_bbox,\
            loss_structure.mrcnn_class,\
            loss_structure.mrcnn_bbox,\
            loss_structure.mrcnn_mask
