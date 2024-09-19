import torch.nn as nn
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.optim as optim
import torch
from torch import flatten

from sklearn.model_selection import KFold

import os
import h5py

import pandas as pd
import numpy as np

from scipy import constants

import matplotlib.pyplot as plt

import time

import multiprocessing


def radarData(data_hdf5):
    '''
    Takes a processed hdf5 file of data
    Returns the radar data hdf5 only
    '''
    return data_hdf5['Sensors']['TI_Radar']['Data']

def radarDataTensor(radar_data_hdf5):
    '''
    Takes a processed hdf5 file of data
    Outputs numpy array of (frames, channels, chirps, adc_values)
    '''
    num_frames = len(radar_data_hdf5.keys()) # for tensor creation
    frame_0_shape = radar_data_hdf5['Frame_0']['frame_data'].shape # for tensor creation
    data_cube = torch.zeros((num_frames, frame_0_shape[2], frame_0_shape[1], frame_0_shape[0]), dtype=torch.complex64) # create tensor        
    for frame in range(num_frames): # move axes into a more sensible order (frames, channels, chirps, values)
        raw = torch.from_numpy(radar_data_hdf5['Frame_' + str(frame)]['frame_data'][:])
        data_cube[frame] = torch.permute(raw, (2, 1, 0))
    return data_cube

def rangeFft(data_cube, window=None):
    ''' Compute FFT over range axis with the specific windowing function
    '''
    if window:
        window_tensor = window(data_cube.shape[3])[None, None, None, ...] # create window tensor with correct number of axes
        data_cube = data_cube*window_tensor.expand(data_cube.shape) #
    range_fft = torch.fft.fft(data_cube, dim=3)
    return range_fft

def dopplerFft(data_cube, window=None):
    '''
    Compute FFT over doppler axis with the specified windowing funtion
    '''
    if window:
        window_tensor = window(data_cube.shape[2])[None, None, ..., None] # create window tensor with correct number of axes
        data_cube = data_cube*window_tensor.expand(data_cube.shape) #
    doppler_fft = torch.fft.fft(data_cube, dim=2)
    return doppler_fft

def rangeDoppler(data_cube, window=None):
    '''
    Get range doppler tensor for inputting data cube tensor
    '''
    return dopplerFft(rangeFft(data_cube, window), window)

def getRangeResolution(radar_conf_hdf5):
    '''
    Read radar config and get the range resolution for range-doppler plot
    '''
    profile_cfg = radar_conf_hdf5['profileCfg']
    T_adc_us = (profile_cfg['numAdcSamples'][()])/(profile_cfg['digOutSampleRate'][()]*1e3)*1e6
    # freq_slope = profile_cfg['freqSlopeConst'][()]*3.6e9*900/(2**26) # convert frequency slope constant to value in Hz
    freq_slope = profile_cfg['freqSlopeConst'][()] * 1e6
    bandwidth = T_adc_us*freq_slope
    range_res = constants.c/(2*bandwidth)
    return range_res

def getVelocityResolution(radar_conf_hdf5):
    '''
    Read radar config and get the velocity resolution for range-doppler plot
    '''
    profile_cfg = radar_conf_hdf5['profileCfg']
    chirp_time = profile_cfg['idleTime'][()]*1e-6 + profile_cfg['rampEndTime'][()]*1e-6 # is chirp time the full time or just ADC sampling time?
    num_chirps = radar_conf_hdf5['frameCfg']['numChirps'][()]
    doppler_res = 1/(num_chirps * chirp_time)
    frequency_centre = profile_cfg['startFreq'][()]*1e9 + profile_cfg['freqSlopeConst'][()]*1e6*profile_cfg['rampEndTime'][()]/2
    velocity_res = doppler_res*(constants.c / (2*frequency_centre)) # is this a valid way of determining chirp wavelength?
    return velocity_res

def radarConf(file_hdf5):
    '''
    Takes a processed hdf5 file of data
    Returns the radar configuration hdf5 only
    '''
    return file_hdf5['Sensors']['TI_Radar']['Parameters']

def todB(data_cube):
    '''
    Convert input torch tensor to dB
    '''
    return 20*torch.log10(data_cube)

def CFAR(data_cube, kernel, pfa=1e-6):
    # currently expects a 2d kernel
    num_training_cells = torch.sum(kernel) # number of training cells
    alpha = num_training_cells * (torch.pow(pfa, -1/num_training_cells) -1) # threshold gain

    # convert data cube to power
    data_cube = torch.pow(torch.abs(data_cube), 2)

    # cfar kernel
    # snr = torch.zeros(torch.shape(data_cube)) # extra stuff needs to be done to find SNR as well if wanted
    noise_cube = torch.zeros(data_cube.shape[2:], dtype=data_cube.dtype)
    noise_cube[0:kernel.shape[0], 0:kernel.shape[1]] = kernel # zero pad in range and doppler dimensions
    # print(noise_cube)
    noise_cube = noise_cube[None, None, ...].expand(data_cube.shape) # expand across all channels and frames

    noise_cube = torch.fft.ifft2(torch.conj(torch.fft.fft2(noise_cube))*torch.fft.fft2(data_cube)) # do the fancy frequency domain based convolution
    noise_cube = torch.roll(noise_cube, kernel.shape[0]//2) # Not sure I understand this step but it is apparently necessary

    data_cube = torch.where(torch.abs(data_cube).gt(torch.abs(noise_cube)*alpha), 1.0, 0) # generate CFAR for each range-doppler map in the cube

    data_cube = torch.amax(data_cube, dim=(0, 1), keepdim=True) # sum CFAR across all channels and all frames

    return data_cube

def generateDopplerKernel(len, guard_len):
    len = 2*(len//2)+1 #  might make a kernel bigger than desired
    guard_len = 2*(guard_len//2)+1 # might make guard kernel bigger than desired
    unguarded_len = (len-guard_len)//2
    kernel = torch.ones(len, 1)
    kernel[unguarded_len:-unguarded_len] = 0

    return kernel

def processData(file_hdf5, crop=True):
    # initial processing
    # timePoint("initial")
    data_cube = radarDataTensor(radarData(file_hdf5)) # create tensor
    # timePoint("read data") #2

    data_cube = rangeDoppler(data_cube, torch.hann_window) # compute range doppler
    # timePoint("range-doppler")

    # calculate range and velocity resolutions
    conf_hdf5 = radarConf(file_hdf5) # get config hdf5
    range_resolution = getRangeResolution(conf_hdf5) # read range res
    velocity_resolution = getVelocityResolution(conf_hdf5) # read vel res
    # timePoint("resolutions") #3


    # restrict range to useful portion - bearing in mind that 0 is at the end
    range_max = data_cube.shape[3]
    range_start = range_max - int(1/range_resolution) # minimum range at 0.4m
    range_end = range_max - int(0.4/range_resolution) # max range at 1m
    # timePoint("indexing") #4
    
    # generate CFAR over all frames
    data_cube = CFAR(data_cube, generateDopplerKernel(25, 11), 1e-6)
    # timePoint("CFAR")

    # TODO: Remove this bit
    if crop:
        data_cube = data_cube[:,:,:,range_start:range_end]



    # restrict doppler to useful portion
    data_cube = torch.fft.fftshift(data_cube, 2) # fftshift first
    velocity_centre = data_cube.shape[2]//2
    velocity_min = velocity_centre - int(10/velocity_resolution)
    velocity_max = velocity_centre + int(10/velocity_resolution)
    if crop:
        data_cube = data_cube[:,:,velocity_min:velocity_max, :]
    # timePoint("cropping") #5


    ### This stuff takes the maximum of over all frames for the first channel only
    # # select only first channel - keep dimensions of tensor for now
    # data_cube = data_cube[:,[0],...] # just use first channel

    # # take absolute values
    # data_cube = torch.abs(data_cube)

    # # take max values across all frames - keep dimensions of tensor for now
    # data_cube = torch.amax(data_cube, dim=0, keepdim=True) 

    # # convert to dB
    # data_cube = todB(data_cube)


    # data_cube = data_cube / torch.max(data_cube)
    return data_cube

class GestureDataset(Dataset):
    """Dataset of radar hand gestures"""

    def __init__(self, csv_file, root_dir, classes, transform=None):
        """
        Arguments:
            csv_file (string) Path to the csv file with ground truth information
            root_dir (string) Directory with the data items
            transform (callable, optional): Optional transform to be applied on a sample
        """

        self.gesture_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.classes = classes
    
    def __len__(self):
        return len(self.gesture_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # some of this could be done with transforms... but I am doing it here for now
        file_path = os.path.join(self.root_dir, self.gesture_df.iloc[idx, 0]) # path to the hdf5 file
        file_hdf5 = h5py.File(file_path) # load the hdf5 file

        if self.transform:
            data_cube = self.transform(file_hdf5)

        data_cube = data_cube[0, 0, :, :].unsqueeze(0)

        gesture = self.gesture_df.iloc[idx, 1]
        gesture_id = self.classes.index(gesture)

        return data_cube, gesture_id

class GestureModel(nn.Module):
    def __init__(self, numChannels, numClasses):
        
        super(GestureModel, self).__init__()

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
        x = flatten(x, start_dim=1, end_dim=-1)
        x = self.fc1(x)
        x = self.relu3(x)

        # Layer 4 (linear)
        x = self.fc2(x)
        x = self.relu4(x)

        # Layer 5 (linear)
        x = self.fc3(x)
        x = self.logsoftmax(x)

        return x

def runCnn():
    classes = ("empty","virtual_tap","virtual_slider_left","virtual_slider_right","swipe_left","still_hand")
    input_csv = "data/gestures.csv"
    root_dir = "data"

    criterion = nn.CrossEntropyLoss()
    num_epochs = 40

    data_transforms = v2.Compose(
        [
            v2.Lambda(processData)
        ]
    )

    dataset = GestureDataset(input_csv, root_dir, classes, data_transforms)

    columns = []
    for i in range(5):
        for j in ("train_loss", "val_loss", "val_acc"):
            columns.append(j+str(i+1))

    out_df = pd.DataFrame(columns=columns)
    kfold = KFold(n_splits=5, shuffle=True)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        model = GestureModel(1, len(classes))
        optimiser = optim.Adam(model.parameters(), lr=0.005)
        
        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler =  SubsetRandomSampler(val_ids)

        train_loader = DataLoader(dataset, batch_size=16, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=16, sampler=val_sampler)

        loss_history = runModel(model, optimiser, train_loader, val_loader, criterion, num_epochs)
        
        for i in ("train_loss", "val_loss", "val_acc"):
            out_df[i + str(fold+1)] = loss_history[i]

    out_df.to_csv (r'results.csv', index = False, header=True)        

def evaluate(model, loader, criterion):
    model.eval()
    # initialise evaluation parameters
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad(): # evaluating so don't produce gradients
        for data in loader:
            inputs, labels = data # get data from dataloader
            outputs = model(inputs) # predict outputs
            loss = criterion(outputs, labels) # calculate current loss
            _, predicted = torch.max(outputs.data, 1) # calculate predicted data
            total += labels.size(0) # total number of labels in the current batch
            correct += (predicted == labels).sum().item() # number of labels that are correct
            
            running_loss += loss.item() # loss? not 100% sure
        
    # Return mean loss, accuracy
    return running_loss / len(loader), correct / total

def runModel(model, optimiser, train_loader, val_loader, criterion, num_epochs):
    loss_history = {
        'train_loss': [],
        'val_loss': [], 
        'val_acc': []
    }

    iteration = 0

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print("Starting Epoch: {}".format(epoch+1))
        
        for i, data in enumerate(train_loader, 0):
            model.train()
            
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimiser.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            loss_history['train_loss'].append(loss.item())
            val_loss, val_acc = evaluate(model, val_loader, criterion)
            loss_history['val_loss'].append(val_loss)
            loss_history['val_acc'].append(val_acc)
            iteration+=1

    print('Finished Training')

    return loss_history

def runTest():
    file_path = os.fsencode("/home/rtdug/UCT-EEE4022S-2024/trial-2/data/Experiment_2024-09-17_11-00-31_049_virtual_tap.hdf5") # path to the hdf5 file
    file_hdf5 = h5py.File(file_path) # load the hdf5 file


    fig, (ax1, ax2) = plt.subplots(2)


    to_plot = processData(file_hdf5)[0, 0, ...]
    
    velocity_resolution = getVelocityResolution(radarConf(file_hdf5))
    velocity_max = to_plot.shape[0]//2*velocity_resolution
    range_resolution = getRangeResolution(radarConf(file_hdf5))
    range_max = to_plot.shape[1]*range_resolution

    
    # flip lr for plotting
    to_plot = torch.fliplr(to_plot)
    ax1.imshow(to_plot, interpolation='none', extent=(0, range_max, -velocity_max, velocity_max), aspect='auto')

    to_plot = processData(file_hdf5, False)[0, 0, ...]
    

    velocity_resolution = getVelocityResolution(radarConf(file_hdf5))
    velocity_max = to_plot.shape[0]//2*velocity_resolution
    range_resolution = getRangeResolution(radarConf(file_hdf5))
    range_max = to_plot.shape[1]*range_resolution

    # flip lr for plotting
    to_plot = torch.fliplr(to_plot)
    ax2.imshow(to_plot, interpolation='none', extent=(0.4, 0.4 + range_max, -velocity_max, velocity_max), aspect='auto')
    
    plt.savefig("/home/rtdug/UCT-EEE4022S-2024/trial-2/figures/fig.png")
    plt.close()


def timePoint(name):
    global prevTime
    curTime = time.time()
    timeDiff = curTime - prevTime
    print(name, ": ", timeDiff*1e3, sep="")
    prevTime = curTime

def main():
    # torch.set_default_device('cuda')
    global prevTime
    prevTime = time.time()
    # runCnn()
    runTest()

if __name__ == "__main__":
    main()