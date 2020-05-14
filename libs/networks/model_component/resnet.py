import os,sys 
sys.path.append(os.path.abspath('.'))
import torch
import torch.nn as nn
from libs.networks.network_utils.utils import SamePad2d


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = (nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
                      .float())
        self.bn1 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.01)
        self.padding2 = SamePad2d(kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.01)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes * 4, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.padding2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, architecture, stage5=True):
        super(ResNet, self).__init__()
        assert architecture in ["resnet34", "resnet50", "resnet101"]
        self.inplanes = 64
        self.layers = [3, 4, {"resnet34":1, "resnet50": 6, "resnet101": 23}[architecture], 3]
        self.block = Bottleneck
        self.stage5 = stage5

        self.C1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3).float(),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.01).float(),
            nn.ReLU().float(),
            SamePad2d(kernel_size=3, stride=2).float(),
            nn.MaxPool2d(kernel_size=3, stride=2).float(),
        )
        self.C2 = self.make_layer(self.block, 64, self.layers[0]).float()
        self.C3 = self.make_layer(self.block, 128, self.layers[1], stride=2)
        self.C4 = self.make_layer(self.block, 256, self.layers[2], stride=2)
        if self.stage5:
            self.C5 = self.make_layer(self.block, 512, self.layers[3],
                                      stride=2)
        else:
            self.C5 = None

    def forward(self, x):
        x = self.C1(x) # B x 64 x H/4 x W/4
        x = self.C2(x) # B x 256 x H/4 x W/4
        x = self.C3(x) # B x 512 x H/8 x H/8
        x = self.C4(x) # B x 1024 x H/16 x H/16
        x = self.C5(x) # B x 2048 x H/32 x H/32
        return x

    def stages(self):
        return [self.C1, self.C2, self.C3, self.C4, self.C5]

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion, eps=0.001,
                               momentum=0.01),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

if __name__ == "__main__":
    network = ResNet(architecture="resnet50", stage5=True)
    state_dict = torch.load('./pretrained_model/resnet50_imagenet.pth')
    network.load_state_dict(state_dict)
    image = torch.rand((4,3,800,800))
    output = network(image)
