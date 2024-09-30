# torch imports
import torch
from torch import nn


# other processing imports
import matplotlib.pyplot as plt

# general python imports
import os
import time

# my other files
import radarprocessing as radar

class CfarModel1(nn.Module):
    def __init__(self, numChannels, numClasses):
        
        super(CfarModel1, self).__init__()

        # Layer 1 (convolutional)
        self.conv1 = nn.Conv2d(numChannels, 6, 3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)

        # Layer 2 (convolutional)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, 2)

        # Layer 4 (linear)
        self.fc1 = nn.LazyLinear(100)
        self.relu3 = nn.ReLU()

        # Layer 5 (linear)
        self.fc2 = nn.LazyLinear(50)
        self.relu4 = nn.ReLU()

        # Layer 6 (linear)
        self.fc3 = nn.LazyLinear(numClasses)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # Layer 2 (convolutional)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # Layer 3 (linear)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc1(x)
        x = self.relu3(x)

        # Layer 4 (linear)
        x = self.fc2(x)
        x = self.relu4(x)

        # Layer 5 (linear)
        x = self.fc3(x)
        x = self.logsoftmax(x)

        return x

