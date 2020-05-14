import os,sys
sys.path.append(os.path.abspath('.'))
from torch import nn
from libs.networks.network_utils.utils import SamePad2d

############################################################
#  Region Proposal Network
############################################################


class RPN(nn.Module):
    """Builds the model of Region Proposal Network.

    anchors_per_location: number of anchors per pixel in the feature map
                        in such FPN(Feature Pyramid Network) architecture, each scale of pyramid
                        will output a feature map use to propose bbox(each scale of feature map 
                        according to one anchor scale). Thus the anchors_per_location here is
                        len(anchor_ratios)
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    input_channels: the channels of the input feature.

    Returns:
        rpn_logits: [batch, H*W*A, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H*W*A, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H*W*A, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
        * where H is the height, W is the width and A is number of anchors per location
    """

    def __init__(self, anchors_per_location, anchor_stride, input_channels):
        super(RPN, self).__init__()
        self.anchors_per_location = anchors_per_location
        self.anchor_stride = anchor_stride
        self.input_channels = input_channels
        self.padding = SamePad2d(kernel_size=3, stride=self.anchor_stride)
        self.conv_shared = nn.Conv2d(self.input_channels, 512, kernel_size=3,
                                     stride=self.anchor_stride)
        self.relu = nn.ReLU()


        self.conv_class = nn.Conv2d(512, 2 * anchors_per_location,
                                    kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=2)
        self.conv_bbox = nn.Conv2d(512, 4 * anchors_per_location,
                                   kernel_size=1, stride=1)

    def forward(self, x):
        # Shared convolutional base of the RPN
        x = self.relu(self.conv_shared(self.padding(x)))
        # x = self.conv_shared(x)

        # Anchor Score. [batch, anchors per location * 2, height, width].
        rpn_class_logits = self.conv_class(x)

        # Reshape to [batch, anchors, 2]
        rpn_class_logits = rpn_class_logits.permute(0, 2, 3, 1)
        rpn_class_logits = rpn_class_logits.contiguous()
        rpn_class_logits = rpn_class_logits.view(x.size()[0], -1, 2)

        # Softmax on last dimension of BG/FG.
        rpn_probs = self.softmax(rpn_class_logits)

        # Bounding box refinement. [batch, 4 * anchors per location, H, W,]
        # where 4 is [x, y, log(w), log(h)]
        rpn_bbox = self.conv_bbox(x)

        # Reshape to [batch, anchors, 4]
        rpn_bbox = rpn_bbox.permute(0, 2, 3, 1)
        rpn_bbox = rpn_bbox.contiguous()
        rpn_bbox = rpn_bbox.view(x.size()[0], -1, 4)

        return [rpn_class_logits, rpn_probs, rpn_bbox] # [B, num_anchors, 2], [B, num_anchors, 2], [B, num_anchors, 4]



if __name__ == "__main__":
    import torch
    images = torch.rand(4, 256, 50, 50)
    rpn = RPN(3, 1, 256)
    output = rpn(images)
    for out in output:
        print(out.size())


