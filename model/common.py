import torch
import torch.nn as nn


def autopad(kernel_size, p=None):
    if p is None:
        # auto-pad
        p = kernel_size // 2 if isinstance(kernel_size,
                                           int) else [x // 2 for x in kernel_size]
    return p


class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, act=True):
        super(Focus, self).__init__()
        self.conv = Conv(in_channels * 4, out_channels,
                         kernel_size, stride, padding, groups, act)

    def forward(self, x):
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, autopad(
            kernel_size, padding), groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, groups=1, expansion=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(c_, out_channels, 3, 1, groups=groups)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    def __init__(self, in_channels, out_channels, count=1, shortcut=True, groups=1, expansion=0.5):
        super(C3, self).__init__()
        c_ = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(in_channels, c_, 1, 1)
        self.cv3 = Conv(2 * c_, out_channels, 1)
        self.m = nn.Sequential(
            *[Bottleneck(c_, c_, shortcut, groups, expansion=1.0) for _ in range(count)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPP(nn.Module):
    def __init__(self, in_channels, out_channels, kernels=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = in_channels // 2
        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(kernels) + 1), out_channels, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in kernels])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Detect(nn.Module):
    # stride = None

    def __init__(self, num_classes=80, anchors=(), channels=()):
        super(Detect, self).__init__()
        self.num_classes = num_classes
        self.num_outputs = num_classes + 5
        self.num_layers = len(anchors)
        self.num_anchors = len(anchors[0]) // 2
        self.module = nn.ModuleList(
            nn.Conv2d(x, self.num_outputs * self.num_anchors, 1) for x in channels)

    def forward(self, x):
        for i in range(self.num_layers):
            x[i] = self.module[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(
                0, 1, 3, 4, 2).contiguous()

        return x
