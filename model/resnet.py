import torch.nn as nn
from torchvision.models.resnet import ResNet
import torchvision.models.resnet as rn


class ConvNet(nn.Module):

    def __init__(self, resnet: ResNet, num_classes=1000):
        super(ConvNet, self).__init__()
        resnet.fc = nn.Sequential(nn.Linear(resnet.fc.in_features, 512),
                                  nn.ReLU(inplace=True))
        self.resnet = resnet
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.fr1 = nn.Linear(256, 4)

    def forward(self, x, _):
        x = self.resnet(x)

        x = self.fc1(x)
        x = self.relu(x)
        cl = self.fc2(x)
        bb = self.fr1(x)

        return cl, bb


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
