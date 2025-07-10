import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseSubnetwork(nn.Module):
    def __init__(self):
        super(SiameseSubnetwork, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(32 * 49, 128)  # 适用于长度 400 的输入

    def forward(self, x):
        # print(f"Subnetwork input shape: {x.shape}")
        x = F.relu(self.conv1(x))
        # print(f"After conv1: {x.shape}")
        x = self.pool1(x)
        # print(f"After pool1: {x.shape}")
        x = F.relu(self.conv2(x))
        # print(f"After conv2: {x.shape}")
        x = self.pool2(x)
        # print(f"After pool2: {x.shape}")
        x = x.view(x.size(0), -1)
        # print(f"Flattened shape: {x.shape}")
        x = self.fc1(x)
        # print(f"Output shape: {x.shape}")
        return x


class MetaSiameseNetwork(nn.Module):
    def __init__(self):
        super(MetaSiameseNetwork, self).__init__()
        self.subnetwork = SiameseSubnetwork()

    def forward(self, x1, x2):
        feat1 = self.subnetwork(x1)
        feat2 = self.subnetwork(x2)
        epsilon = 1e-7
        distance = torch.sqrt(torch.sum((feat1 - feat2) ** 2, dim=1) + epsilon)
        return distance

    def get_embedding(self, x):
        return self.subnetwork(x)