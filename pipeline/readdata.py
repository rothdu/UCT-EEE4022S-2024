import torch
import numpy as np

import h5py

from scipy import constants

def radarDataTensor(file_hdf5):
    '''Takes a an hdf5 file of data, outputs torch tensor of (frames, channels, chirps, adc_values)
    '''
    # start_time = time.time()
    data_idx_str = 'Sensors/TI_Radar/Data' # points to the hdf5 location containing the 

    num_frames = len(file_hdf5[f"{data_idx_str}"].keys()) # for tensor creation
    frame_0_shape = file_hdf5[f"{data_idx_str}/Frame_0/frame_data"].shape # for tensor creation
    data_cube = torch.zeros((num_frames, frame_0_shape[2], frame_0_shape[1], frame_0_shape[0]), dtype=torch.complex64, device='cpu') # create tensor
    for frame in range(num_frames): # move axes into a more sensible order (frames, channels, chirps, values)
        raw = torch.as_tensor(file_hdf5[f'{data_idx_str}/Frame_{frame}/frame_data'][:], dtype=torch.complex64, device='cpu')
        data_cube[frame] = torch.permute(raw, (2, 1, 0))
    
    # print("Data loading time: ", (time.time() - start_time)*1000)
    return data_cube

def rangeResolution(file_hdf5):
    '''Read radar config and get the range resolution for range-doppler plot'''

    profile_idx_str = 'Sensors/TI_Radar/Parameters/profileCfg'
    T_adc_us = (file_hdf5[f'{profile_idx_str}/numAdcSamples'][()])/(file_hdf5[f'{profile_idx_str}/digOutSampleRate'][()]*1e3)*1e6
    # freq_slope = file_hdf5[f'{profile_idx_str}/freqSlopeConst'][()]*3.6e9*900/(2**26) # convert frequency slope constant to value in Hz
    freq_slope = file_hdf5[f'{profile_idx_str}/freqSlopeConst'][()] * 1e6
    bandwidth = T_adc_us*freq_slope
    range_res = constants.c/(2*bandwidth)
    return range_res

def velocityResolution(file_hdf5):
    '''Read radar config and get the velocity resolution for range-doppler plot'''
    profile_idx_str = 'Sensors/TI_Radar/Parameters/profileCfg'
    frame_idx_str = 'Sensors/TI_Radar/Parameters/frameCfg'

    chirp_time = file_hdf5[f'{profile_idx_str}/idleTime'][()]*1e-6 + file_hdf5[f'{profile_idx_str}/rampEndTime'][()]*1e-6
     # is chirp time the full time or just ADC sampling time?
    num_chirps = file_hdf5[f'{frame_idx_str}/numChirps'][()]
    doppler_res = 1/(num_chirps * chirp_time)
    frequency_centre = file_hdf5[f'{profile_idx_str}/startFreq'][()]*1e9 + file_hdf5[f'{profile_idx_str}/freqSlopeConst'][()]*1e6*file_hdf5[f'{profile_idx_str}/rampEndTime'][()]/2

    velocity_res = doppler_res*(constants.c / (2*frequency_centre)) # is this a valid way of determining chirp wavelength?

    return velocity_res

def centreFrames(data_cube, num_frames):
    '''get the specified number of frames from the centre of a data cube'''
    low = data_cube.shape[0]//2 - num_frames//2
    high = data_cube.shape[0]//2 + num_frames - num_frames//2
    return data_cube[low:high, ...]


