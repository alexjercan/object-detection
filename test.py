from typing import Union
import torch
from torch.functional import Tensor
import torchvision.transforms as transforms
from argparse import ArgumentParser
from dataset.blender_dataset import BlenderDataset
from torch.utils.data import DataLoader
from model.depthnet import DepthNet, depthnet152, depthnet18
from torchvision.models.resnet import ResNet, resnet152, resnet18


def get_args():
    parser = ArgumentParser(
        description='Trains a nn using the blender dataset.')
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--use_gpu', action="store_true", default=True)
    parser.add_argument('--checkpoint', type=str, default='./checkpoint.pth')
    parser.add_argument('--resnet', action="store_true", default=False)

    args = parser.parse_args()
    return args


def _model(use_resnet: bool):
    return resnet18 if use_resnet else depthnet18


def forward(model: Union[ResNet, DepthNet], images: Tensor, depth_images: Tensor) -> Tensor:
    if isinstance(model, ResNet):
        return model(images)
    return model(images, depth_images)


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

    use_resnet = args.resnet
    num_classes = args.num_classes
    model = _model(use_resnet)(num_classes=num_classes, zero_init_residual=True)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, depth_images, labels in test_loader:
            images: Tensor = images.to(device)
            depth_images: Tensor = depth_images.to(device)
            labels: Tensor = labels.to(device)

            outputs: Tensor = forward(model, images, depth_images)

            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')
