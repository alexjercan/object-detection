import torch
import torch.nn as nn
import torchvision.transforms as transforms
from argparse import ArgumentParser
from dataset.blender_dataset import BlenderDataset
from torch.utils.data import DataLoader
from model.rgb_depth_model import ConvNet


def get_args():
    parser = ArgumentParser(
        description='Trains a nn using the blender dataset.')
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--use_gpu', action="store_true", default=True)
    parser.add_argument('--save_path', type=str, default='./cnn.pth')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available()
                          and args.use_gpu else 'cpu')

    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

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

    train_dataset = BlenderDataset(root_dir=args.dataset_path, cvs_fname='train.csv',
                                   render_transform=render_transform, depth_transform=depth_transform, train=True)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    classes = ('object01', 'object02', 'object03')

    model = ConvNet(len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for images, depth_images, labels in train_loader:
            images = images.to(device)
            depth_images = depth_images.to(device)
            labels = labels.to(device)

            outputs = model(images, depth_images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print('Finished Training')
    PATH = args.save_path
    torch.save(model.state_dict(), PATH)
