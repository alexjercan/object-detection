import torch
from torch.functional import Tensor
import torchvision.transforms as transforms
from argparse import ArgumentParser
from dataset.blender_dataset import BlenderDataset
from torch.utils.data import DataLoader
from model.depthnet import depthnet152, depthnet18
from model.resnet import resnet152, resnet18
from time import time
import numpy as np


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


if __name__ == '__main__':
    args = get_args()

    use_cuda = torch.cuda.is_available() and args.use_gpu
    device = torch.device('cuda' if use_cuda else 'cpu')

    batch_size = args.batch_size

    render_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    depth_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    seg_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])

    test_dataset = BlenderDataset(root_dir='C:/dev/blenderRenderer/output', csv_fname='test.csv', class_fname='class.csv',
                                  render_transform=render_transform, depth_transform=depth_transform, albedo_transform=seg_transform, train=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    classes = test_dataset.classes

    use_resnet = args.resnet
    num_classes = test_dataset.num_classes
    model = (resnet18 if use_resnet else depthnet18)(
        num_classes=num_classes, zero_init_residual=True)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        t1 = time()
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for _ in range(num_classes + 1)]
        n_class_samples = [0 for _ in range(num_classes + 1)]
        n_total_steps = len(test_loader)
        for i, (rgb_images, depth_images, t_labels, t_bboxes, t_seg_masks) in enumerate(test_loader):
            rgb_images: Tensor = rgb_images.to(device)
            depth_images: Tensor = depth_images.to(device)
            t_labels: Tensor = t_labels.to(device)
            t_bboxes: Tensor = t_bboxes.to(device)
            t_seg_masks: Tensor = t_seg_masks.to(device)

            p_labels, p_bboxes, p_seg_masks = model(rgb_images, depth_images)

            _, p_labels = torch.max(p_labels, 1)
            n_samples += t_labels.size(0)
            n_correct += (p_labels == t_labels).sum().item()

            for i, t_label in enumerate(t_labels):
                p_label = p_labels[i]
                if p_label == t_label:
                    n_class_correct[t_label] += 1
                n_class_samples[t_label] += 1

            if (i + 1) % 200 == 0:
                print(f'Step [{i + 1}/{n_total_steps}]')

        t2 = time()
        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy: {acc}%, Time: {(t2 - t1):.4f}s')

        for i in range(0, num_classes):
            if n_class_samples[i] == 0:
                continue
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {classes[i]}: {acc}%')
