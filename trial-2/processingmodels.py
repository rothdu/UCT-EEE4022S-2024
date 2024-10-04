import torch

import radarprocessing as radar

import readdata as read

def cfarProcess1(data_cube, input_hdf5):
    '''Take in a data cube tensor and do the various radar processing'''    
    data_cube = radar.rangeDoppler(data_cube, torch.hann_window) # compute range doppler
    # calculate range and velocity resolutions

    # generate CFAR over all frames
    data_cube = radar.cfar(data_cube, radar.generateDopplerKernel(25, 11), 1e-3)
    # data_cube = radar.todB(torch.abs(data_cube))

    data_cube = torch.amax(data_cube, dim=(0, 1), keepdim=True) # sum CFAR across all channels and all frames

    # select first frame and channel (i.e., only frame and channel given CFAR), then add "channel" dimension for pytorch
    data_cube = data_cube[0, 0, :, :].unsqueeze(0)

    return data_cube

def cfarProcess2(data_cube, input_hdf5):
    '''Take in a data cube tensor and do the various radar processing'''    
    data_cube = radar.rangeDoppler(data_cube, torch.hann_window) # compute range doppler
    
    # generate CFAR over all frames
    kernel = torch.zeros((25, 11))
    kernel[12, :3] = 1
    kernel[:7, 5] = 1
    kernel[-7:, 5] = 1
    data_cube = radar.cfar(data_cube, kernel, 5e-2)

    data_cube = torch.amax(data_cube, dim=(0, 1), keepdim=True) # sum CFAR across all channels and all frames

    # select first frame and channel (i.e., only frame and channel given CFAR), then add "channel" dimension for pytorch
    data_cube = data_cube[0, 0, :, :].unsqueeze(0)

    return data_cube

def cfarProcess3(data_cube, input_hdf5):
    '''Take in a data cube tensor and do the various radar processing'''    
    data_cube = radar.rangeDoppler(data_cube, torch.hann_window) # compute range doppler
    data_cube = data_cube[:20, ...]

    # generate CFAR over all frames
    kernel = torch.zeros((25, 11))
    kernel[12, :3] = 1
    kernel[:7, 5] = 1
    kernel[-7:, 5] = 1
    data_cube = radar.cfar(data_cube, kernel, 5e-2)

    data_cube = torch.amax(data_cube, dim=(0, 1), keepdim=True) # sum CFAR across all channels and all frames

    # select first frame and channel (i.e., only frame and channel given CFAR), then add "channel" dimension for pytorch
    data_cube = data_cube[0, 0, :, :].unsqueeze(0)

    return data_cube


def cfarCrop1(data_cube, file_hdf5):
    range_res = read.rangeResolution(file_hdf5)
    velocity_res = read.velocityResolution(file_hdf5)
    
    range_low = int(0.25/range_res) # min range at 0.25 m
    range_high = int(0.75/range_res) # max range at 0.5 m

    # restrict doppler to useful portion
    velocity_centre = data_cube.shape[-2]//2
    velocity_low = velocity_centre - int(3/velocity_res)
    velocity_high = velocity_centre + int(3/velocity_res)
    
    # TODO: Get rid of this plotting section
    # fig, (ax1, ax2) = plt.subplots(2)
    # to_plot = data_cube[0, 0, ...].to('cpu')
    # ax1.imshow(to_plot, interpolation='none', extent=(range_max, 0, -velocity_max, velocity_max), aspect='auto')
    
    data_cube = data_cube[...,range_low:range_high] # remember to think... has the map had a fliplr?
    data_cube = data_cube[...,velocity_low:velocity_high, :]

    return data_cube

def cfarCrop2(data_cube, file_hdf5):
    range_res = read.rangeResolution(file_hdf5)
    velocity_res = read.velocityResolution(file_hdf5)
    
    range_low = int(0.4/range_res) # min range at 0.25 m
    range_high = int(0.8/range_res) # max range at 0.5 m

    # restrict doppler to useful portion
    velocity_centre = data_cube.shape[-2]//2
    velocity_low = velocity_centre - int(10/velocity_res)
    velocity_high = velocity_centre + int(10/velocity_res)
    
    # TODO: Get rid of this plotting section
    # fig, (ax1, ax2) = plt.subplots(2)
    # to_plot = data_cube[0, 0, ...].to('cpu')
    # ax1.imshow(to_plot, interpolation='none', extent=(range_max, 0, -velocity_max, velocity_max), aspect='auto')
    
    data_cube = data_cube[...,range_low:range_high] # remember to think... has the map had a fliplr?
    data_cube = data_cube[...,velocity_low:velocity_high, :]

    return data_cube

def microDopplerProcess1(data_cube, input_hdf5):
    
    # data_cube = read.centreFrames(data_cube, 20)
    data_cube = data_cube[:20, ...]

    range_res = read.rangeResolution(input_hdf5)

    spectrogram = radar.microDoppler(data_cube, (8, 14), n_fft=128, win_length=16, hop_length=2)
    # spectrogram = spectrogram[10:-10, :]


    spectrogram = torch.abs(spectrogram)
    spectrogram = radar.todB(spectrogram)

    
    spectrogram = spectrogram -35
    spectrogram = spectrogram / 40

    spectrogram[spectrogram < 0] = 0
    spectrogram[spectrogram > 1] = 1
    
    spectrogram = spectrogram.unsqueeze(0)

    return spectrogram
