import os
from typing import List
import cv2
import glob
import numpy as np
from pathlib import Path
import torch
import random

img_formats = ['bmp', 'jpg', 'jpeg', 'png',
               'tif', 'tiff', 'dng', 'webp', 'exr']

L_RGB = 'rgb'
L_DEPTH = 'depth'
L_NORMAL = 'normal'

CHANNELS = {L_RGB: 3, L_DEPTH: 3, L_NORMAL: 3}


def count_channles(layers):
    return sum([CHANNELS[chan] for chan in layers])


def load_img_paths(path: str, used_layers: List):
    layer_files = {}
    path = Path(path)
    f = glob.glob(str(path / '**' / '*.*'), recursive=True)
    for layer in used_layers:
        layer_files[layer] = sorted([x.replace('/', os.sep) for x in f if x.split(
            '.')[-1].lower() in img_formats and layer.lower() in x.lower()])
    return layer_files


def load_label_paths(path: str):
    path = Path(path)
    f = glob.glob(str(path / '**' / '*label.txt'), recursive=True)
    label_files = sorted([x.replace('/', os.sep) for x in f])
    return label_files


def load_data(self, index):
    layers0 = {}
    hw0, hw = (0, 0), (0, 0)
    if L_RGB in self.used_layers:
        img0, hw0, hw = load_image(self, index)
        layers0[L_RGB] = img0
    if L_DEPTH in self.used_layers:
        depth0, hw0, hw = load_depth(self, index)
        layers0[L_DEPTH] = depth0
    if L_NORMAL in self.used_layers:
        normal0, hw0, hw = load_normal(self, index)
        layers0[L_NORMAL] = normal0

    # The images have the format wxhxc
    return layers0, hw0, hw


def load_image(self, index):
    path = self.layer_files[L_RGB][index]
    img = cv2.imread(path)  # BGR
    assert img is not None, 'Image Not Found ' + path
    h0, w0 = img.shape[:2]  # orig hw
    r = self.image_size / max(h0, w0)  # resize image to image_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 and not self.transform else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized


def load_depth(self, index):
    path = self.layer_files[L_DEPTH][index]
    depth = exr2depth(path)  # 3 channel depth
    assert depth is not None, 'Image Not Found ' + path
    h0, w0 = depth.shape[:2]  # orig hw
    r = self.image_size / max(h0, w0)  # resize image to image_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 and not self.transform else cv2.INTER_LINEAR
        depth = cv2.resize(depth, (int(w0 * r), int(h0 * r)),
                           interpolation=interp)
    return depth, (h0, w0), depth.shape[:2]  # img, hw_original, hw_resized


def load_normal(self, index):
    path = self.layer_files[L_NORMAL][index]
    normal = exr2normal(path)  # 3 channel normal
    assert normal is not None, 'Image Not Found ' + path
    h0, w0 = normal.shape[:2]  # orig hw
    r = self.image_size / max(h0, w0)  # resize image to image_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 and not self.transform else cv2.INTER_LINEAR
        normal = cv2.resize(
            normal, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return normal, (h0, w0), normal.shape[:2]  # img, hw_original, hw_resized


def exr2depth(path):
    if not os.path.isfile(path):
        return None

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)

    # get the maximum value from the array, aka the most distant point
    # everything above that value is infinite, thus i clamp it to maxvalue
    # then divide by maxvalue to obtain a normalized map
    # multiply by 255 to obtain a colormap from the depthmap
    maxvalue = np.max(img[img < np.max(img)])
    img[img > maxvalue] = maxvalue
    img = img / maxvalue * 255

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = np.array(img).astype(np.uint8).reshape(
        img.shape[0], img.shape[1], -1)

    return img


def exr2normal(path):
    if not os.path.isfile(path):
        return None

    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    img[img > 1] = 1
    img[img < 0] = 0
    img = img * 255

    img = np.array(img).astype(np.uint8).reshape(
        img.shape[0], img.shape[1], -1)

    return img


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] +
        boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def build_targets(bboxes, anchors, S):
    """[summary]

    Args:
        bboxes (tensor): contains Nx[img,C,X,Y,W,H]
        anchors (list): list of anchors
        S (list): layer scales at detection

    Returns:
        list: NSx[shape(num_achors,S,S,[img,C,x,y,w,h,c])]
    """
    anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
    num_anchors = anchors.shape[0]
    num_anchors_per_scale = num_anchors // 3
    ignore_iou_thresh = 0.5

    targets = [torch.zeros((num_anchors // 3, s, s, 7)) for s in S]
    for box in bboxes:
        box = box.tolist()
        iou_anchors = iou_width_height(torch.tensor(box[4:6]), anchors)
        anchor_indices = iou_anchors.argsort(descending=True, dim=0)
        img, class_label, x, y, width, height = box
        has_anchor = [False] * 3
        for anchor_idx in anchor_indices:
            scale_idx = anchor_idx // num_anchors_per_scale
            anchor_on_scale = anchor_idx % num_anchors_per_scale
            s = S[scale_idx]
            i, j = int(s * y), int(s * x)
            anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 1]
            targets[scale_idx][anchor_on_scale, i, j, 0] = img
            if not anchor_taken and not has_anchor[scale_idx]:
                x_cell, y_cell = s * x - j, s * y - i
                width_cell, height_cell = (
                    width * s,
                    height * s,
                )
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                targets[scale_idx][anchor_on_scale,
                                   i, j, 1] = int(class_label)
                targets[scale_idx][anchor_on_scale,
                                   i, j, 2:6] = box_coordinates
                targets[scale_idx][anchor_on_scale, i, j, 6] = 1
                has_anchor[scale_idx] = True

            elif not anchor_taken and iou_anchors[anchor_idx] > ignore_iou_thresh:
                targets[scale_idx][anchor_on_scale,
                                   i, j, 6] = -1

    return targets


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
