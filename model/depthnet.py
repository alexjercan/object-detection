from model.utils import DepthModule, ResNetModule
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck


class DepthNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(DepthNet, self).__init__()
        self.depth_module = DepthModule(out_size=512)
        self.resnet_module = ResNetModule(
            block, layers, zero_init_residual, out_size=512)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(512 * 2, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.fr1 = nn.Linear(256, 4) 

    def forward(self, x, y):
        x = self.resnet_module(x)
        y = self.depth_module(y)

        z = torch.cat((x.view(x.size(0), -1), y.view(y.size(0), -1)), dim=1)
        z = self.fc1(z)
        z = self.relu(z)
        cl = self.fc2(z)
        bb = self.fr1(z)

        return cl, bb


def depthnet18(pretrained=False, **kwargs):
    return DepthNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def depthnet34(pretrained=False, **kwargs):
    return DepthNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def depthnet50(pretrained=False, **kwargs):
    return DepthNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def depthnet101(pretrained=False, **kwargs):
    return DepthNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def depthnet152(pretrained=False, **kwargs):
    return DepthNet(Bottleneck, [3, 8, 36, 3], **kwargs)
