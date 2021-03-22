import config
import torch.optim as optim

from data.dataset import create_dataloader
from model.model import Model
from util.general import (
    check_class_accuracy,
    get_evaluation_bboxes,
    mean_average_precision,
    count_channles,
    load_yaml,
    load_checkpoint,
)
import warnings
warnings.filterwarnings("ignore")


def test(loader, model):
    check_class_accuracy(model, loader, threshold=config.CONF_THRESHOLD,
                         device=config.DEVICE, anchors=config.ANCHORS, S=config.S)
    pred_boxes, true_boxes = get_evaluation_bboxes(loader, model, iou_threshold=config.NMS_IOU_THRESH,
                                                   anchors=config.ANCHORS, threshold=config.CONF_THRESHOLD,
                                                   S=config.S, device=config.DEVICE)
    mapval = mean_average_precision(pred_boxes, true_boxes, iou_threshold=config.MAP_IOU_THRESH,
                                    box_format="midpoint", num_classes=config.NUM_CLASSES)
    print(f"MAP: {mapval.item()}")


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
                                        transform=None, used_layers=config.LAYERS)

    load_checkpoint(config.CHECKPOINT_FILE, model, optimizer,
                    config.LEARNING_RATE, config.DEVICE)

    test(loader, model)
