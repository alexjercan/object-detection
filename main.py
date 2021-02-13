import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from dataset.blender_dataset import BlenderDataset
from torch.utils.data import DataLoader

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 125 * 125, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 125 * 125)       
        x = F.relu(self.fc1(x))               
        x = F.relu(self.fc2(x))              
        x = self.fc3(x)                       
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs = 5
    batch_size = 4
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

    train_dataset = BlenderDataset(root_dir='C:/dev/blenderRenderer/output', render_transform=render_transform, depth_transform=depth_transform, train=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    classes = ('object09', 'object19', 'object27')

    model = ConvNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, depth_images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 2000 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

    print('Finished Training')
    PATH = './cnn.pth'
    torch.save(model.state_dict(), PATH)