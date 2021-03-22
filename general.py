import config
import torch

from tqdm import tqdm
from common import build_targets, check_class_accuracy, get_evaluation_bboxes, mean_average_precision

def train(loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(loader, leave=True)
    losses = []

    for _, (im0s, layers, labels) in enumerate(loop):
        targets = build_targets(labels, len(im0s), config.ANCHORS, config.S)
        targets = [target.to(config.DEVICE) for target in targets]
        layers = layers.to(config.DEVICE, non_blocking=True).float() / 255.0

        with torch.cuda.amp.autocast():
            predictions = model(layers)
            loss = loss_fn(predictions, targets, scaled_anchors)

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)


def test(loader, model):
    check_class_accuracy(model, loader, threshold=config.CONF_THRESHOLD,
                         device=config.DEVICE, anchors=config.ANCHORS, S=config.S)
    pred_boxes, true_boxes = get_evaluation_bboxes(loader, model, iou_threshold=config.NMS_IOU_THRESH,
                                                   anchors=config.ANCHORS, threshold=config.CONF_THRESHOLD,
                                                   S=config.S, device=config.DEVICE)
    mapval = mean_average_precision(pred_boxes, true_boxes, iou_threshold=config.MAP_IOU_THRESH,
                                    box_format="midpoint", num_classes=config.NUM_CLASSES)
    print(f"MAP: {mapval.item()}")