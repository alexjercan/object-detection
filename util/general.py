import os
import cv2
import glob
import yaml
import time
import torch
import random
import torchvision
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


def iou_width_height(wh1, wh2):
    """
    Parameters:
        wh1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(wh1[..., 0], wh2[..., 0]) * torch.min(
        wh1[..., 1], wh2[..., 1]
    )
    union = (
        wh1[..., 0] * wh1[..., 1] +
        wh2[..., 0] * wh2[..., 1] - intersection
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


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Video explanation of this function:
    https://youtu.be/FppOzcDvaDI
    This function calculates mean average precision (mAP)
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def build_targets(labels, im0s, anchors, S):
    """[summary]

    Args:
        labels (tensor): contains Nx[img,C,x,y,w,h]
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
    for label in labels:
        label = label.tolist()
        iou_anchors = iou_width_height(torch.tensor(label[4:6]), anchors)
        anchor_indices = iou_anchors.argsort(descending=True, dim=0)
        img, C, x, y, w, h = label
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


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [conf, xyxy, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 0] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, 0] = 1.0  # conf
            v[:, 1:5] = l[:, 1:5]  # box
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 0:1]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, 1:5])

        # Detections matrix nx6 (conf, xyxy, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((x[i, j + 5, None], box[i], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((conf, box, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 0].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, 1:5] + c, x[:, 0]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, 1:5] = torch.mm(weights, x[:, 1:5]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


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
