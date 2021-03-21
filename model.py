import yaml
import torch
import torch.nn as nn

from pathlib import Path
from common import count_channles

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, *x):
        return torch.cat(x, self.d)


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        c_ = in_channels // 2
        self.conv1 = Conv(in_channels, c_, kernel_size=1)
        self.conv2 = Conv(c_, in_channels, kernel_size=3, padding=1)
        self.conv3 = Conv(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv3(self.conv2(self.conv1(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    Conv(channels, channels // 2, kernel_size=1),
                    Conv(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        c_ = 2 * in_channels
        self.num_anchors = 3
        self.num_classes = num_classes
        self.num_outputs = num_classes + 5
        self.pred = nn.Sequential(
            Conv(in_channels, c_, kernel_size=3, padding=1),
            nn.Conv2d(c_, self.num_outputs * self.num_anchors, kernel_size=1),
        )

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], self.num_anchors, self.num_outputs, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )


class Detect(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_predictions = len(in_channels)
        self.m = nn.ModuleList(ScalePrediction(x, num_classes) for x in in_channels)

    def forward(self, *layers):
        layers = list(layers)
        for i in range(self.num_predictions):
            layers[i] = self.m[i](layers[i])
        return layers


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.yaml_file = Path(cfg).name
        with open(cfg) as f:
            self.model_dict = yaml.load(f, Loader=yaml.SafeLoader)
        self.in_channels = count_channles(self.model_dict['layers'])
        self.num_classes = self.model_dict["num_classes"]
        self.layers = self._create_layers()

    def forward(self, x):
        outputs = [x]
        for i, layer in enumerate(self.layers):
            x = layer(*[outputs[j] for j in layer.from_layers])
            if i == 0:
                outputs = []
            outputs.append(x)
        return x

    def _create_layers(self):
        d = self.model_dict
        layers = nn.ModuleList()
        num_classes = self.num_classes
        channels = [self.in_channels]

        for i, (from_layers, module, args) in enumerate(d['backbone'] + d['head']):
            module = eval(module) if isinstance(module, str) else module
            for j, a in enumerate(args):
                try:
                    args[j] = eval(a) if isinstance(a, str) else a
                except:
                    pass

            if module is Conv:
                in_channels, out_channels = channels[from_layers[0]], args[0]
                args = [in_channels, out_channels, *args[1:]]
            elif module is CNNBlock:
                in_channels = channels[from_layers[0]]
                out_channels = in_channels // 2
                args = [in_channels, out_channels]
            elif module is ResidualBlock:
                out_channels = channels[from_layers[0]]
                args = [out_channels, *args]
            elif module is Detect:
                in_channels = [channels[x] for x in from_layers]
                args = [in_channels, num_classes]
            elif module is Concat:
                out_channels = sum([channels[x] for x in from_layers])

            module_ = module(*args)
            module_.from_layers = from_layers
            layers.append(module_)
            if i == 0:
                channels = []
            channels.append(out_channels)
                

        return layers


if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = Model("model.yaml")
    x = torch.randn((2, 9, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")
