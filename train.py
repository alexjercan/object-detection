from dataset import create_dataloader
import config
import torch
import torch.optim as optim

from model import Model
from general import train
from common import (
    save_checkpoint,
    load_checkpoint,
    count_channles,
    load_yaml
)
from loss import LossFunction
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    model = Model(model_dict=load_yaml(config.MODEL_DICT),
                  in_channels=count_channles(config.LAYERS),
                  num_classes=config.NUM_CLASSES
                  ).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(),
                           lr=config.LEARNING_RATE,
                           weight_decay=config.WEIGHT_DECAY)
    loss_fn = LossFunction()
    scaler = torch.cuda.amp.GradScaler()

    dataset, loader = create_dataloader(config.IMG_DIR + "/train", config.LABEL_DIR + "/train", image_size=config.IMAGE_SIZE,
                                        batch_size=config.BATCH_SIZE, S=config.S, anchors=config.ANCHORS,
                                        transform=config.test_transforms, used_layers=config.LAYERS)

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_FILE, model, optimizer,
                        config.LEARNING_RATE, config.DEVICE)

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    for epoch in range(config.NUM_EPOCHS):
        train(loader, model, optimizer, loss_fn, scaler, scaled_anchors)
        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer, filename=config.CHECKPOINT_FILE)
