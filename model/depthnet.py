import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet
import torchvision.models.resnet as rn


class DepthNet(nn.Module):

    def __init__(self, out_features=512):
        super(DepthNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=11, stride=2, padding=5,
                               bias=False)
        self.conv16 = nn.Conv2d(16, 64, kernel_size=11, stride=4, padding=5,
                                bias=False)
        self.bn16 = nn.BatchNorm2d(16)
        self.bn64 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(4096, out_features)

    def forward(self, y):
        y = self.conv1(y)
        y = self.bn16(y)
        y = self.relu(y)
        y = self.maxpool(y)

        y = self.conv16(y)
        y = self.bn64(y)
        y = self.relu(y)
        y = self.maxpool(y)

        y = y.view(y.size(0), -1)
        y = self.fc(y)
        y = self.relu(y)

        return y


class ConvNet(nn.Module):

    def __init__(self, resnet: ResNet, num_classes=1000):
        super(ConvNet, self).__init__()
        resnet.fc = nn.Sequential(nn.Linear(resnet.fc.in_features, 512),
                                  nn.ReLU(inplace=True))
        self.resnet = resnet
        self.depthnet = DepthNet(out_features=512)

        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(512 * 2, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.fr1 = nn.Linear(256, 4)

    def forward(self, x, y):
        x = self.resnet(x)
        y = self.depthnet(y)

        z = torch.cat((x.view(x.size(0), -1), y.view(y.size(0), -1)), dim=1)
        z = self.fc1(z)
        z = self.relu(z)
        cl = self.fc2(z)
        bb = self.fr1(z)

        return cl, bb


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
