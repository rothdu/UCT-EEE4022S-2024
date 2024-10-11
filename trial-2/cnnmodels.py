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

class CfarModel2(nn.Module):
    def __init__(self, numChannels, numClasses):
        
        super(CfarModel2, self).__init__()

        # Layer 1 (convolutional)
        self.conv1 = nn.Conv2d(numChannels, 16, 3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)

        # Layer 2 (convolutional)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, 2)

        # Layer 4 (linear)
        self.fc1 = nn.LazyLinear(256)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.LazyLinear(128)
        self.relu4 = nn.ReLU()
        # Layer 5 (linear)
        self.fc3 = nn.LazyLinear(64)
        self.relu5 = nn.ReLU()

        # Layer 6 (linear)
        self.fc4 = nn.LazyLinear(numClasses)
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
        x = self.relu5(x)

        x = self.fc4(x)
        x = self.logsoftmax(x)

        return x


class microDopplerModel1(nn.Module):
    def __init__(self, numChannels, numClasses):
        
        super(microDopplerModel1, self).__init__()

        self.conv1 = nn.Conv2d(numChannels, 4, 5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(4, 16, 3)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(16, 64, 5)
        self.relu4 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.LazyLinear(128)
        self.relu5 = nn.ReLU()

        self.fc2 = nn.LazyLinear(256)
        self.relu6 = nn.ReLU()

        self.fc3 = nn.LazyLinear(64)
        self.relu7 = nn.ReLU()

        self.fc4 = nn.LazyLinear(numClasses)
        self.logsoftmax1 = nn.LogSoftmax(dim=1)


    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu4(x)
        x = self.maxpool3(x)

        x = torch.flatten(x, start_dim=1, end_dim=-1)

        x = self.fc1(x)
        x = self.relu5(x)

        x = self.fc2(x)
        x = self.relu6(x)

        x = self.fc3(x)
        x = self.relu7(x)

        x = self.fc4(x)
        x = self.logsoftmax1(x)

        return x


class microDopplerModel2(nn.Module):
    def __init__(self, numChannels, numClasses):
        
        super(microDopplerModel2, self).__init__()

        self.conv1 = nn.Conv2d(numChannels, 4, 5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(4, 16, 3)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(16, 32, 5)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d((2, 4))

        self.conv4 = nn.Conv2d(32, 64, 5)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d((1, 4))

        self.conv5 = nn.Conv2d(64, 128, 5)
        self.relu5 = nn.ReLU()
        self.maxpool5 = nn.MaxPool2d((1, 4))


        self.fc1 = nn.LazyLinear(256)
        self.relu6 = nn.ReLU()

        self.fc2 = nn.LazyLinear(512)
        self.relu7 = nn.ReLU()

        self.fc3 = nn.LazyLinear(128)
        self.relu8 = nn.ReLU()

        self.fc4 = nn.LazyLinear(64)
        self.relu9 = nn.ReLU()

        self.fc5 = nn.LazyLinear(numClasses)
        self.logsoftmax1 = nn.LogSoftmax(dim=1)


    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)


        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        x = torch.flatten(x, start_dim=1, end_dim=-1)

        x = self.fc1(x)
        x = self.relu6(x)

        x = self.fc2(x)
        x = self.relu7(x)

        x = self.fc3(x)
        x = self.relu8(x)

        x = self.fc4(x)
        x = self.relu9(x)

        x = self.fc5(x)
        x = self.logsoftmax1(x)

        return x


class AngleModel1(nn.Module):
    def __init__(self, numChannels, numClasses):
        
        super(AngleModel1, self).__init__()

        # Layer 1 (convolutional)
        self.conv1 = nn.Conv3d(numChannels, 6, 3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool3d(2, 2)

        # Layer 2 (convolutional)
        self.conv2 = nn.Conv3d(6, 16, 3)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

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


class CfarModel3(nn.Module):
    def __init__(self, numChannels, numClasses):
        
        super(CfarModel3, self).__init__()

        # Layer 1 (convolutional)

        # Layer 2 (convolutional)
        self.conv2 = nn.Conv3d(numChannels, 16, 3)
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = nn.Conv3d(16, 64, 3)
        self.relu3 = nn.ReLU()
        self.maxpool2 = nn.MaxPool3d((2, 2, 2), (2, 2, 2))

        # Layer 4 (linear)
        self.fc1 = nn.LazyLinear(64)
        self.relu4 = nn.ReLU()

        # Layer 5 (linear)
        self.fc2 = nn.LazyLinear(128)
        self.relu5 = nn.ReLU()

        # Layer 6 (linear)
        self.fc3 = nn.LazyLinear(32)
        self.relu6 = nn.ReLU()

        self.fc4 = nn.LazyLinear(numClasses)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):


        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)


        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool2(x)

        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc1(x)
        x = self.relu4(x)

        x = self.fc2(x)
        x = self.relu5(x)

        x = self.fc3(x)
        x = self.relu6(x)

        x = self.fc4(x)
        x = self.logsoftmax(x)

        return x
    

class NewModel1(nn.Module):
    def __init__(self, numChannels, numClasses):
        
        super(NewModel1, self).__init__()

        # Layer 1 (convolutional)

        # Layer 2 (convolutional)
        self.conv2 = nn.Conv3d(numChannels, 16, 2)
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool3d((1, 2, 1), (1, 2, 1))

        self.conv3 = nn.Conv3d(16, 64, 2)
        self.relu3 = nn.ReLU()
        self.maxpool2 = nn.MaxPool3d((2, 2, 2), (2, 2, 2))

        # Layer 4 (linear)
        self.fc1 = nn.LazyLinear(64)
        self.relu4 = nn.ReLU()

        # Layer 5 (linear)
        self.fc2 = nn.LazyLinear(128)
        self.relu5 = nn.ReLU()

        # Layer 6 (linear)
        self.fc3 = nn.LazyLinear(32)
        self.relu6 = nn.ReLU()

        self.fc4 = nn.LazyLinear(numClasses)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):


        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)


        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool2(x)

        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc1(x)
        x = self.relu4(x)

        x = self.fc2(x)
        x = self.relu5(x)

        x = self.fc3(x)
        x = self.relu6(x)

        x = self.fc4(x)
        x = self.logsoftmax(x)

        return x