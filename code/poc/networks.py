import torch
from torch import nn
import torch.nn.functional as F


class FirstEyeNet(nn.ModuleList):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 10)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(6, 16, 10)
        self.conv3 = nn.Conv2d(16, 32, 10)
        self.fc1 = nn.Linear(5568, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
