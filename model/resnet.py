from model.utils import ResNetModule
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.resnet_module = ResNetModule(
            block, layers, zero_init_residual, out_size=512)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.fr1 = nn.Linear(256, 4) 

    def forward(self, x, _):
        x = self.resnet_module(x)

        x = self.fc1(x)
        x = self.relu(x)
        cl = self.fc2(x)
        bb = self.fr1(x)

        return cl, bb


def resnet18(pretrained=False, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(pretrained=False, **kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(pretrained=False, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(pretrained=False, **kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(pretrained=False, **kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
