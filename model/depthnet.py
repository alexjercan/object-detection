import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models.resnet import ResNet
from torchvision.models._utils import IntermediateLayerGetter
import torchvision.models.resnet as rn
from torchvision.models.segmentation.fcn import FCNHead


class ConvNet(nn.Module):

    def __init__(self, resnet: ResNet, num_classes=1000):
        super(ConvNet, self).__init__()
        resnet.conv1 = _reset_conv1(resnet)
        self.resnet = IntermediateLayerGetter(
            resnet, return_layers={'layer4': 'out'})
        self.segmentation = FCNHead(resnet.fc.in_features, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(resnet.fc.in_features, 512),
                                nn.ReLU(inplace=True))
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.fr1 = nn.Linear(256, 4)

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)

        input_shape = x.shape[-2:]

        features = self.resnet(x)

        x = features['out']
        x = self.segmentation(x)
        seg = F.interpolate(x, size=input_shape,
                            mode='bilinear', align_corners=False)

        x = features['out']
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = self.fc1(x)
        x = self.relu(x)
        cl = self.fc2(x)
        bb = self.fr1(x)

        return cl, bb, seg


def _reset_conv1(resnet: ResNet):
    out_channels = resnet.conv1.out_channels
    kernel_size = resnet.conv1.kernel_size
    stride = resnet.conv1.stride
    padding = resnet.conv1.padding
    bias = resnet.conv1.bias
    return nn.Conv2d(4, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


def depthnet18(pretrained=False, num_classes=1000, **kwargs):
    resnet = rn.resnet18(pretrained, **kwargs)
    return ConvNet(resnet, num_classes)


def depthnet34(pretrained=False, num_classes=1000, **kwargs):
    resnet = rn.resnet34(pretrained, **kwargs)
    return ConvNet(resnet, num_classes)


def depthnet50(pretrained=False, num_classes=1000, **kwargs):
    resnet = rn.resnet50(pretrained, **kwargs)
    return ConvNet(resnet, num_classes)


def depthnet101(pretrained=False, num_classes=1000, **kwargs):
    resnet = rn.resnet101(pretrained, **kwargs)
    return ConvNet(resnet, num_classes)


def depthnet152(pretrained=False, num_classes=1000, **kwargs):
    resnet = rn.resnet152(pretrained, **kwargs)
    return ConvNet(resnet, num_classes)
