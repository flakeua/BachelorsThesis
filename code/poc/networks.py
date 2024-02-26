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


class SecondEyeNet(nn.ModuleList):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 9, 10)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(9, 26, 10)
        self.fc1 = nn.Linear(5278, 700)
        self.fc2 = nn.Linear(700, 130)
        self.fc3 = nn.Linear(130, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ThirdEyeNet(nn.ModuleList):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 9, 10)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(9, 26, 10)
        self.fc1 = nn.Linear(5281, 700)
        self.fc2 = nn.Linear(700, 130)
        self.fc3 = nn.Linear(130, 2)

    def forward(self, x):
        x, head_pos = x
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.cat((x, head_pos), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FifthEyeNet(nn.ModuleList):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 9, 3)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(9, 26, 3)
        self.fc1 = nn.Linear(939, 300)
        self.fc2 = nn.Linear(300, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x, head_pos = x
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.cat((x, head_pos), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FourthEyeNet(nn.ModuleList):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 9, 3)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(9, 26, 3)
        self.fc1 = nn.Linear(3432, 600)
        self.fc2 = nn.Linear(603, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x, head_pos = x
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = torch.cat((x, head_pos), 1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SixthEyeNet(nn.ModuleList):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 9, 3)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(9, 26, 3)
        self.fc1 = nn.Linear(3432, 600)
        self.fc2 = nn.Linear(600, 50)
        self.fc3 = nn.Linear(53, 2)

    def forward(self, x):
        x, head_pos = x
        head_pos = head_pos
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.cat((x, head_pos), 1)
        x = self.fc3(x)
        return x


class ForthEyeNetNoHeadPose(nn.ModuleList):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 9, 3)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(9, 26, 3)
        self.fc1 = nn.Linear(3432, 600)
        self.fc2 = nn.Linear(600, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x, head_pos = x
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SeventhEyeNet(nn.ModuleList):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 9, 3)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(9, 26, 3)
        self.fc1 = nn.Linear(3432, 600)
        self.fc2 = nn.Linear(600, 50)
        self.fc3 = nn.Linear(53, 2)

    def forward(self, x):
        x, head_pos = x
        head_pos = head_pos
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.cat((x, head_pos), 1)
        x = self.fc3(x)
        return x
