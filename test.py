import torch
import config
import torch.optim as optim

from tqdm import tqdm
from data.dataset import create_dataloader
from model.model import Model
from util.general import (
    build_targets,
    count_channles,
    load_yaml,
    load_checkpoint, mean_average_precision, non_max_suppression,
)
import warnings
warnings.filterwarnings("ignore")


def test(loader, model):
    model.eval()
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for idx, (im0s, layersx, labels) in enumerate(tqdm(loader)):
        layers = layersx.to(config.DEVICE, non_blocking=True).float() / 255.0
        labels = labels.to(config.DEVICE)

        with torch.no_grad():
            out = model(layers)

        targets = build_targets(labels, len(im0s), config.ANCHORS, config.S)
        for i in range(len(targets)):
            targets[i] = targets[i].to(config.DEVICE)
            obj = targets[i][..., 0] == 1  # in paper this is Iobj_i
            noobj = targets[i][..., 0] == 0  # in paper this is Iobj_i

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-
                             1) == targets[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > config.CONF_THRESHOLD
            correct_obj += torch.sum(obj_preds[obj] == targets[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj]
                                       == targets[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

        out = torch.cat([o.reshape(len(im0s), -1, o.shape[4]) for o in out], 1)
        out = non_max_suppression(
            out, conf_thres=config.CONF_THRESHOLD, iou_thres=config.NMS_IOU_THRESH, multi_label=True)

    print(f"Class accuracy: {(correct_class/(tot_class_preds+1e-16))*100:2f}%")
    print(f"No obj accuracy: {(correct_noobj/(tot_noobj+1e-16))*100:2f}%")
    print(f"Obj accuracy: {(correct_obj/(tot_obj+1e-16))*100:2f}%")
    model.train()


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
