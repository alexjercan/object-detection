from data.dataset import create_dataloader
import config
import torch
import torch.optim as optim

from tqdm import tqdm
from model.model import Model
from model.loss import LossFunction
from util.general import (
    build_targets,
    save_checkpoint,
    load_checkpoint,
    count_channles,
    load_yaml
)
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True


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
