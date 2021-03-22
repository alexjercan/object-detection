import config
import torch.optim as optim

from dataset import create_dataloader
from model import Model
from common import (
    count_channles,
    load_yaml,
    mean_average_precision,
    get_evaluation_bboxes,
    load_checkpoint,
    check_class_accuracy,
)
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    model = Model(model_dict=load_yaml(config.MODEL_DICT),
                  in_channels=count_channles(config.LAYERS),
                  num_classes=config.NUM_CLASSES
                  ).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(),
                           lr=config.LEARNING_RATE,
                           weight_decay=config.WEIGHT_DECAY)
    dataset, loader = create_dataloader(config.IMG_DIR + "/test", config.LABEL_DIR + "/test", image_size=config.IMAGE_SIZE,
                                        batch_size=config.BATCH_SIZE, S=config.S, anchors=config.ANCHORS,
                                        transform=config.test_transforms, used_layers=config.LAYERS)

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_FILE, model, optimizer,
                        config.LEARNING_RATE, config.DEVICE)

    check_class_accuracy(model, loader, threshold=config.CONF_THRESHOLD,
                         device=config.DEVICE, anchors=config.ANCHORS, S=config.S)
    pred_boxes, true_boxes = get_evaluation_bboxes(loader, model, iou_threshold=config.NMS_IOU_THRESH,
                                                   anchors=config.ANCHORS, threshold=config.CONF_THRESHOLD,
                                                   S=config.S, device=config.DEVICE)
    mapval = mean_average_precision(pred_boxes, true_boxes, iou_threshold=config.MAP_IOU_THRESH,
                                    box_format="midpoint", num_classes=config.NUM_CLASSES)
    print(f"MAP: {mapval.item()}")
