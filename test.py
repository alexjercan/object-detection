import torch
import torchvision.transforms as transforms
from argparse import ArgumentParser
from dataset.blender_dataset import BlenderDataset
from torch.utils.data import DataLoader
from model.rgb_depth_model import ConvNet


def get_args():
    parser = ArgumentParser(
        description='Trains a nn using the blender dataset.')
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--use_gpu', action="store_true", default=True)
    parser.add_argument('--save_path', type=str, default='./cnn.pth')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available()
                          and args.use_gpu else 'cpu')

    batch_size = args.batch_size

    render_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    depth_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    test_dataset = BlenderDataset(root_dir='C:/dev/blenderRenderer/output', cvs_fname='test.csv',
                                  render_transform=render_transform, depth_transform=depth_transform, train=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    classes = ('object01', 'object02', 'object03')

    model = ConvNet(len(classes)).to(device)

    PATH = args.save_path
    model.load_state_dict(torch.load(PATH))
    model.eval()

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, depth_images, labels in test_loader:
            images = images.to(device)
            depth_images = depth_images.to(device)
            labels = labels.to(device)
            outputs = model(images, depth_images)

            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')
