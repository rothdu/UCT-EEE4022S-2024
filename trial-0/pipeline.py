import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from scipy import constants
from scipy import signal

def radarData(data_hdf5):
    '''
    Takes a processed hdf5 file of data
    Returns the radar data hdf5 only
    '''
    return data_hdf5['Sensors']['TI_Radar']['Data']

def radarConf(data_hdf5):
    '''
    Takes a processed hdf5 file of data
    Returns the radar configuration hdf5 only
    '''
    return data_hdf5['Sensors']['TI_Radar']['Parameters']

def numpyRadarData(radar_data_hdf5):
    '''
    Takes a processed hdf5 file of data
    Outputs numpy array of (frames, channels, chirps, adc_values)
    '''
    num_frames = len(radar_data_hdf5.keys()) # for numpy array creation
    frame_0_shape = radar_data_hdf5['Frame_0']['frame_data'].shape # for numpy array creation
    data_frames = np.zeros((num_frames, frame_0_shape[2], frame_0_shape[1], frame_0_shape[0]), dtype=np.complex64) # create numpy array
    for frame in range(num_frames): # move axes into a more sensible order (frames, channels, chirps, values)
        data_frames[frame] = np.transpose(np.array(radar_data_hdf5['Frame_' + str(frame)]['frame_data']))
    return data_frames

def getRangeResolution(radar_conf_hdf5):
    profile_cfg = radar_conf_hdf5['profileCfg']
    T_adc_us = (profile_cfg['numAdcSamples'][()])/(profile_cfg['digOutSampleRate'][()]*1e3)*1e6
    # freq_slope = profile_cfg['freqSlopeConst'][()]*3.6e9*900/(2**26) # convert frequency slope constant to value in Hz
    freq_slope = profile_cfg['freqSlopeConst'][()] * 1e6
    bandwidth = T_adc_us*freq_slope
    range_res = constants.c/(2*bandwidth)
    return range_res

def getVelocityResolution(radar_conf_hdf5):
    # This is largely adapted from Ethan Meknassi's code - still need to understand it properly
    profile_cfg = radar_conf_hdf5['profileCfg']
    chirp_time = profile_cfg['idleTime'][()] + profile_cfg['rampEndTime'][()]
    num_chirps = radar_conf_hdf5['frameCfg']['numChirps'][()]
    doppler_res = 1/(num_chirps * chirp_time)
    frequency_centre = profile_cfg['startFreq'][()] + profile_cfg['freqSlopeConst'][()]*1e12*profile_cfg['rampEndTime'][()]/2
    velocity_res = doppler_res*(constants.c / (2*frequency_centre))

def rangefft(data_frames, window='rectangular'):
    window_arr = signal.get_window(window, (data_frames.shape[3]))
    windowed = data_frames*np.expand_dims(window_arr, (0, 1, 2))
    range_fft = np.fft.fft(windowed, axis=3)
    return range_fft

def dopplerfft(data_frames, window='rectangular'):
    window_arr = signal.get_window(window, (data_frames.shape[2]))
    windowed = data_frames*np.expand_dims(window_arr, (0, 1, 3))
    doppler_fft = np.fft.fft(windowed, axis=2)
    return doppler_fft

def todB(data_frames):
    return 20*np.log10(data_frames)

path = "data/stationary_target_3m_1.hdf5"
os_path  = os.fsencode(path)

data_hdf5 = h5py.File(os_path)
data_frames = numpyRadarData(radarData(data_hdf5))
data_conf = radarConf(data_hdf5)

range_fft = rangefft(data_frames, 'rectangular')
doppler_fft = dopplerfft(range_fft, 'rectangular')

range_fft_dB = todB(range_fft)
doppler_fft_dB = todB(doppler_fft)
range_fft_dB_shifted = np.fft.fftshift(range_fft_dB, axes=2)
doppler_fft_dB_shifted = np.fft.fftshift(doppler_fft_dB, axes=2)

delta_R = getRangeResolution(data_conf)
max_R = delta_R * doppler_fft_dB_shifted.shape[3]

delta_V = getVelocityResolution(data_conf)
max_V = delta_R * doppler_fft_dB_shifted.shape[2]//2

fig, ax = plt.subplots()

ax.imshow(np.fliplr(np.abs(doppler_fft_dB_shifted)[50,0,:,:]), interpolation='none', extent = (0, max_R, -max_V, max_V), aspect='auto')

# ax.set_xlim(0, 5)

plt.show()