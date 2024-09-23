# torch imports
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

# torchvision imports
from torchvision.transforms import v2

# sklearn imports
from sklearn.model_selection import KFold

# other processing imports
import h5py
import pandas as pd
import matplotlib.pyplot as plt

# general python imports
import os
import time

# my other files
import readdata as read

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

def cfarProcess1(data_cube, range_res, velocity_res):
    '''Take in a data cube tensor and do the various radar processing'''
    data_cube = radar.rangeDoppler(data_cube, torch.hann_window) # compute range doppler

    # calculate range and velocity resolutions

    # generate CFAR over all frames
    data_cube = radar.cfar(data_cube, radar.generateDopplerKernel(25, 11), 1e-6)

    # restrict range to useful portion - bearing in mind that 0 is at the end without a fliplr
    range_max = data_cube.shape[3]
    range_start = range_max - int(1/range_res) # minimum range at 0.4m
    range_end = range_max - int(0.4/range_res) # max range at 1m
    data_cube = data_cube[:,:,:,range_start:range_end]

    # restrict doppler to useful portion
    data_cube = torch.fft.fftshift(data_cube, 2) # fftshift so that doppler axis is centred
    velocity_centre = data_cube.shape[2]//2
    velocity_min = velocity_centre - int(10/velocity_res)
    velocity_max = velocity_centre + int(10/velocity_res)
    data_cube = data_cube[:,:,velocity_min:velocity_max, :]

    # select first frame and channel (i.e., only frame and channel given CFAR), then add "channel" dimension for pytorch
    data_cube = data_cube[0, 0, :, :].unsqueeze(0)

    return data_cube