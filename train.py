import torch
import torch.nn as nn
import torch.optim
import torchvision.transforms as transforms
from argparse import ArgumentParser
from dataset.blender_dataset import BlenderDataset
from torch.utils.data import DataLoader
from model.depthnet import depthnet152

def get_args():
    parser = ArgumentParser(
        description='Trains a nn using the blender dataset.')
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--use_gpu', action="store_true", default=False)
    parser.add_argument('--output_path', type=str, default='./checkpoint.pth')
    parser.add_argument('--checkpoint', type=str, default=None)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    use_cuda = torch.cuda.is_available() and args.use_gpu
    device = torch.device('cuda' if use_cuda else 'cpu')

    epoch = 0
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

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

    train_dataset = BlenderDataset(root_dir=args.dataset_path, cvs_fname='train.csv',
                                   render_transform=render_transform, depth_transform=depth_transform, train=True)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    model = depthnet152(num_classes=3, zero_init_residual=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    model = model.to(device)

    for epoch in range(epoch, num_epochs):
        for images, depth_images, labels in train_loader:
            images = images.to(device)
            depth_images = depth_images.to(device)
            labels = labels.to(device)

            outputs = model(images, depth_images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, args.output_path)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print('Finished Training')
