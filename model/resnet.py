import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models.resnet import ResNet
import torchvision.models.resnet as rn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.fcn import FCNHead


class ConvNet(nn.Module):

    def __init__(self, resnet: ResNet, num_classes=1000):
        super(ConvNet, self).__init__()
        self.resnet = IntermediateLayerGetter(resnet, return_layers={'layer4': 'out'})
        self.segmentation = FCNHead(resnet.fc.in_features, num_classes)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(resnet.fc.in_features, 512),
                                  nn.ReLU(inplace=True))
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.fr1 = nn.Linear(256, 4)

    def forward(self, x, _):
        input_shape = x.shape[-2:]
        
        features = self.resnet(x)
        
        x = features['out']
        x = self.segmentation(x)
        seg = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        x = features['out']
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        cl = self.fc2(x)
        bb = self.fr1(x)

        return cl, bb, seg


def resnet18(pretrained=False, num_classes=1000, **kwargs):
    resnet = rn.resnet18(pretrained, **kwargs)
    return ConvNet(resnet, num_classes)


def resnet34(pretrained=False, num_classes=1000, **kwargs):
    resnet = rn.resnet34(pretrained, **kwargs)
    return ConvNet(resnet, num_classes)


def resnet50(pretrained=False, num_classes=1000, **kwargs):
    resnet = rn.resnet50(pretrained, **kwargs)
    return ConvNet(resnet, num_classes)


def resnet101(pretrained=False, num_classes=1000, **kwargs):
    resnet = rn.resnet101(pretrained, **kwargs)
    return ConvNet(resnet, num_classes)


def resnet152(pretrained=False, num_classes=1000, **kwargs):
    resnet = rn.resnet152(pretrained, **kwargs)
    return ConvNet(resnet, num_classes)
