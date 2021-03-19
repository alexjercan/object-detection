import yaml
import logging
import torch.nn as nn

from pathlib import Path
from copy import deepcopy

from utils.common import count_channles, make_divisible
from model.common import *
from utils.torch import initialize_weights, model_info

logger = logging.getLogger(__name__)


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', num_classes=None):
        super(Model, self).__init__()
        self.yaml_file = Path(cfg).name
        with open(cfg) as f:
            self.yaml = yaml.load(f, Loader=yaml.SafeLoader)

        self.layers = self.yaml.get('layers', ['rgb'])
        self.in_channels = count_channles(self.layers)
        if num_classes and num_classes != self.yaml['num_classes']:
            logger.info(
                f"Overriding model.yaml num_classes={self.yaml['num_classes']} with num_classes={num_classes}")
            self.yaml['num_classes'] = num_classes
        self.num_classes = self.yaml['num_classes']
        self.anchors = self.yaml['anchors']
        self.model, self.save = parse_model(
            deepcopy(self.yaml), channels=[self.in_channels])
        self.names = [str(i) for i in range(self.yaml['num_classes'])]

        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x):
        y, = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [
                    x if j == -1 else y[j] for j in m.f]  # from earlier layers

            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    def info(self, verbose=False, img_size=640):
        model_info(self, verbose, img_size)


def parse_model(model_dict, channels):
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' %
                ('', 'from_block', 'reps', 'params', 'module', 'arguments'))
    anchors, num_classes, gain_depth, gain_width = model_dict['anchors'], model_dict[
        'num_classes'], model_dict['gain_depth'], model_dict['gain_width']
    num_anchors = (len(anchors[0]) //
                   2) if isinstance(anchors, list) else anchors
    num_outputs = num_anchors * (num_classes + 5)

    layers, save, out_channels = [], [], channels[-1]
    for i, (from_block, reps, module, args) in enumerate(model_dict['backbone'] + model_dict['head']):
        module = eval(module) if isinstance(module, str) else module
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except:
                pass
        reps = max(round(reps * gain_depth), 1) if reps > 1 else reps

        if module in [Conv, Bottleneck, SPP, Focus]:
            in_channels, out_channels = channels[from_block], args[0]
            out_channels = make_divisible(
                out_channels * gain_width, 8) if out_channels != num_outputs else out_channels
            args = [in_channels, out_channels, *args[1:]]
        elif module is C3:
            in_channels, out_channels, count = channels[from_block], args[0], args[1]
            out_channels = make_divisible(
                out_channels * gain_width, 8) if out_channels != num_outputs else out_channels
            count = max(round(count * gain_depth), 1) if count > 1 else count
            args = [in_channels, out_channels, count, *args[2:]]
        elif module is nn.BatchNorm2d:
            args = [channels[from_block]]
        elif module is Concat:
            out_channels = sum([channels[x] for x in from_block])
        elif module is Detect:
            args.append([channels[x] for x in from_block])
        else:
            out_channels = channels[from_block]

        module_ = nn.Sequential(
            *[module(*args) for _ in range(reps)]) if reps > 1 else module(*args)
        module_type = str(module)[8:-2].replace('__main__.', '')
        num_params = sum([x.numel()
                          for x in module_.parameters()])  # number params
        module_.i, module_.from_block, module_.type, module_.num_params = i, from_block, module_type, num_params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' %
                    (i, from_block, reps, num_params, module_type, args))
        save.extend(x % i for x in ([from_block] if isinstance(
            from_block, int) else from_block) if x != -1)
        layers.append(module_)
        if i == 0:
            channels = []
        channels.append(out_channels)
    return nn.Sequential(*layers), sorted(save)
