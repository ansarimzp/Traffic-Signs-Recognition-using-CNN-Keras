import torch
import torch.nn as nn
import torch.nn.functional as F

class RoadSignCNN(nn.Module):
    def __init__(self):
        super(RoadSignCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate correct FC input size
        self.flatten_size = 64 * 30 * 30  # Use correct feature map size

        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.fc2 = nn.Linear(256, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.flatten_size)  # Flatten for FC layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
