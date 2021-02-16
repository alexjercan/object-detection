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
        # n x 3 x 512 x 512
        rgb = F.relu(self.conv1_rgb(rgb))
        # n x 6 x 508 x 508
        rgb = self.pool(rgb)
        # n x 6 x 254 x 254
        rgb = F.relu(self.conv2(rgb))
        # n x 16 x 250 x 250
        rgb = self.pool()
        # n x 16 x 125 x 125
        rgb = rgb.view(-1, 16 * 125 * 125)
        # n x 16 * 125 * 125
        rgb = F.relu(self.fc1(rgb))
        # n x 120

        # n x 1 x 512 x 512
        depth = F.relu(self.conv1_depth(depth))
        # n x 6 x 508 x 508
        depth = self.pool(depth)
        # n x 6 x 254 x 254
        depth = F.relu(self.conv2(depth))
        # n x 16 x 250 x 250
        depth = self.pool(depth)
        # n x 16 x 125 x 125
        depth = depth.view(-1, 16 * 125 * 125)
        # n x 16 * 125 * 125
        depth = F.relu(self.fc1(depth))
        # n x 120

        x = torch.cat((rgb.view(-1, 120), depth.view(-1, 120)), dim=1)
        # n x 2 * 120
        x = F.relu(self.fc2(x))
        # n x 84
        x = self.fc3(x)
        # n x no_classes
        return x
