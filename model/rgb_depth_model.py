import torch
import torch.nn as nn
import torch.nn.functional as F


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
