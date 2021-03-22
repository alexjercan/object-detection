import os
import cv2
import glob
import yaml
import torch
import random
import numpy as np

from tqdm import tqdm
from typing import List
from pathlib import Path
from collections import Counter

img_formats = ['bmp', 'jpg', 'jpeg', 'png',
               'tif', 'tiff', 'dng', 'webp', 'exr']

L_RGB = 'rgb'
L_DEPTH = 'depth'
L_NORMAL = 'normal'

CHANNELS = {L_RGB: 3, L_DEPTH: 3, L_NORMAL: 3}


def count_channles(layers):
    return sum([CHANNELS[chan] for chan in layers])


def load_yaml(cfg):
    with open(cfg) as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


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


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def build_targets(bboxes, im0s, anchors, S):
    """[summary]

    Args:
        bboxes (tensor): contains Nx[img,C,x,y,w,h]
        im0s (int): number of images
        anchors (list): list of anchors
        S (list): layer scales at detection

    Returns:
        list: Sx[tensor(img,num_achors,S,S,[c,x,y,w,h,C])]
    """
    anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
    num_anchors = anchors.shape[0]
    num_anchors_per_scale = num_anchors // 3
    ignore_iou_thresh = 0.5

    targets = [torch.zeros((im0s, num_anchors // 3, s, s, 6)) for s in S]
    for box in bboxes:
        box = box.tolist()
        iou_anchors = iou_width_height(torch.tensor(box[3:5]), anchors)
        anchor_indices = iou_anchors.argsort(descending=True, dim=0)
        img, C, x, y, w, h = box
        has_anchor = [False] * 3
        img = int(img)
        for anchor_idx in anchor_indices:
            scale_idx = anchor_idx // num_anchors_per_scale
            anchor_on_scale = anchor_idx % num_anchors_per_scale
            s = S[scale_idx]
            i, j = int(s * y), int(s * x)
            anchor_taken = targets[scale_idx][img, anchor_on_scale, i, j, 0]
            if not anchor_taken and not has_anchor[scale_idx]:
                x_cell, y_cell = s * x - j, s * y - i
                w_cell, h_cell = w * s,  h * s
                box_cell = torch.tensor([x_cell, y_cell, w_cell, h_cell])
                targets[scale_idx][img, anchor_on_scale, i, j, 0] = 1
                targets[scale_idx][img, anchor_on_scale, i, j, 1:5] = box_cell
                targets[scale_idx][img, anchor_on_scale, i, j, 5] = int(C)
                has_anchor[scale_idx] = True
            elif not anchor_taken and iou_anchors[anchor_idx] > ignore_iou_thresh:
                targets[scale_idx][img, anchor_on_scale, i, j, 0] = -1

    return targets


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
