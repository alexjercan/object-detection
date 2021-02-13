import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from dataset.blender_dataset import BlenderDataset
from torch.utils.data import DataLoader


class ConvNet(nn.Module):
    def __init__(self, no_classes):
        super(ConvNet, self).__init__()
        self.conv1_depth = nn.Conv2d(1, 6, 5)
        self.conv1_rgb = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 125 * 125, 120)
        self.fc2 = nn.Linear(2 * 120, 84)
        self.fc3 = nn.Linear(84, no_classes)

    def forward(self, rgb, depth):
        rgb = self.pool(F.relu(self.conv1_rgb(rgb)))
        rgb = self.pool(F.relu(self.conv2(rgb)))
        rgb = rgb.view(-1, 16 * 125 * 125)
        rgb = F.relu(self.fc1(rgb))

        depth = self.pool(F.relu(self.conv1_depth(depth)))
        depth = self.pool(F.relu(self.conv2(depth)))
        depth = depth.view(-1, 16 * 125 * 125)
        depth = F.relu(self.fc1(depth))

        x = torch.cat((rgb.view(-1, 120), depth.view(-1, 120)), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs = 5
    batch_size = 2
    learning_rate = 0.001

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

    train_dataset = BlenderDataset(root_dir='C:/dev/blenderRenderer/output', cvs_fname='train.csv',
                                   render_transform=render_transform, depth_transform=depth_transform, train=True)
    test_dataset = BlenderDataset(root_dir='C:/dev/blenderRenderer/output', cvs_fname='test.csv',
                                  render_transform=render_transform, depth_transform=depth_transform, train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

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
    PATH = './cnn.pth'
    torch.save(model.state_dict(), PATH)

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
