from typing import Union
import torch
from torch.functional import Tensor
import torchvision.transforms as transforms
from argparse import ArgumentParser
from dataset.blender_dataset import BlenderDataset
from torch.utils.data import DataLoader
from model.depthnet import DepthNet, depthnet152, depthnet18
from torchvision.models.resnet import ResNet, resnet152, resnet18
from time import time


def get_args():
    parser = ArgumentParser(
        description='Trains a nn using the blender dataset.')
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--use_gpu', action="store_true", default=True)
    parser.add_argument('--checkpoint', type=str, default='./checkpoint.pth')
    parser.add_argument('--resnet', action="store_true", default=False)

    args = parser.parse_args()
    return args


def _model(use_resnet: bool):
    return resnet18 if use_resnet else depthnet18


def print_info(images, bboxes, predictions, classes):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    import math
    res = 256
    n = math.ceil((math.sqrt(len(images))))
    _, axs = plt.subplots(nrows=n, ncols=n)
    images = images / 2 + 0.5
    npimgs = np.transpose(np.array(images), (0, 2, 3, 1))
    npbboxes = np.array(bboxes)

    for i, npimg in enumerate(npimgs):
        axs[i].set_title(classes[predictions[i].numpy()])
        npbbox = npbboxes[i]
        rect = patches.Rectangle((npbbox[0] * res, npbbox[2] * res), (npbbox[1] - npbbox[0])
                                 * res, (npbbox[3] - npbbox[2]) * res, linewidth=1, edgecolor='r', facecolor='none')
        axs[i].imshow(npimg)
        axs[i].add_patch(rect)

    plt.show()


if __name__ == '__main__':
    args = get_args()

    use_cuda = torch.cuda.is_available() and args.use_gpu
    device = torch.device('cuda' if use_cuda else 'cpu')

    batch_size = args.batch_size

    render_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    depth_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    test_dataset = BlenderDataset(root_dir='C:/dev/blenderRenderer/output', cvs_fname='test.csv',
                                  render_transform=render_transform, depth_transform=depth_transform, train=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    classes = list(test_dataset.ids.keys())

    use_resnet = args.resnet
    num_classes = test_dataset.num_classes
    model = _model(use_resnet)(
        num_classes=num_classes, zero_init_residual=True)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    image_samples = []
    bbox_samples = []
    prediction_samples = []

    with torch.no_grad():
        t1 = time()
        n_correct = 0
        n_samples = 0
        n_total_steps = len(test_loader)
        for i, (images, depth_images, labels, bboxes) in enumerate(test_loader):
            images: Tensor = images.to(device)
            depth_images: Tensor = depth_images.to(device)
            labels: Tensor = labels.to(device)

            out_labels, out_bboxes = model(images, depth_images)

            _, predictions = torch.max(out_labels, 1)
            n_samples += labels.size(0)
            n_correct += (predictions == labels).sum().item()

            if (i + 1) % 100 == 0:
                print(f'Step [{i + 1}/{n_total_steps}]')
                image_samples.append(images)
                bbox_samples.append(out_bboxes)
                prediction_samples.append(predictions)

        t2 = time()
        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy: {acc}%, Time: {(t2 - t1):.4f}s')
        print_info(image_samples, bbox_samples, prediction_samples, classes)
