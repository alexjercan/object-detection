from plots import plot_images
from common import build_targets
from model import Model
from dataset import BDataset
import torch
from loss import LossFunction
import config
from torch.utils.data import DataLoader

anchors = config.ANCHORS
device = config.DEVICE
transform = config.test_transforms
scaled_anchors = (
    torch.tensor(anchors)
    * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(device)
model = Model("model.yaml")
dataset = BDataset("../bdataset/images/train",
                   "../bdataset/labels/train",
                   image_size=config.IMAGE_SIZE,
                   S=config.S,
                   anchors=anchors,
                   transform=transform,
                   used_layers=model.model_dict['layers'])
loader = DataLoader(dataset=dataset, batch_size=2,
                    shuffle=True, collate_fn=BDataset.collate_fn)
loss_fn = LossFunction()
for i in range(0,3):
    for im0s, layers, labels in loader:
        targets = build_targets(labels, len(im0s), anchors, config.S)
        layers = layers.to(device, non_blocking=True).float() / 255.0
        predictions = model(layers)
        loss = loss_fn(predictions, targets, scaled_anchors)
        print(loss)
        if i != 0:
            continue
        from common import build_targets
        import matplotlib.pyplot as plt
        targets = build_targets(labels, len(im0s), anchors, config.S)
        img = plot_images(im0s, labels, fname=None)
        imgplot = plt.imshow(img)
        plt.show()
