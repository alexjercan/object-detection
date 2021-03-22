from general import test
import config
import torch.optim as optim

from dataset import create_dataloader
from model import Model
from common import (
    count_channles,
    load_yaml,
    load_checkpoint,
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

    load_checkpoint(config.CHECKPOINT_FILE, model, optimizer,
                    config.LEARNING_RATE, config.DEVICE)

    test(loader, model)
