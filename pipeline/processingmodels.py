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

def cfarProcess4(data_cube, input_hdf5):
    '''Take in a data cube tensor and do the various radar processing'''    
    data_cube = radar.rangeDoppler(data_cube, torch.hann_window) # compute range doppler
    data_cube = data_cube[:20, ...]

    # generate CFAR over all frames
    kernel = torch.zeros((25, 11))
    kernel[12, :3] = 1
    kernel[:7, 5] = 1
    kernel[-7:, 5] = 1
    data_cube = radar.cfar(data_cube, kernel, 5e-2)

    data_cube = torch.amax(data_cube, dim=(1,), keepdim=True) # sum CFAR across all channels and all frames
    data_cube_out = torch.zeros_like(data_cube[:10, ...])

    for i in range(10):
        data_cube_out[i] = torch.amax(data_cube[i*2:(i+1)*2, ...], dim=(0,), keepdim=False)

    # select first frame and channel (i.e., only frame and channel given CFAR), then add "channel" dimension for pytorch
    data_cube_out = data_cube_out[:, 0, :, :].unsqueeze(0)

    return data_cube_out


def cfarProcess5(data_cube, input_hdf5):
    '''Take in a data cube tensor and do the various radar processing'''    
    data_cube = radar.rangeDoppler(data_cube, torch.hann_window) # compute range doppler
    data_cube = data_cube - torch.sum(data_cube[0:1], dim=0, keepdim=True)
    data_cube = data_cube[2:, ...]

    # generate CFAR over all frames
    kernel = torch.zeros((25, 11))
    kernel[12, :3] = 1
    kernel[:7, 5] = 1
    kernel[-7:, 5] = 1
    data_cube = radar.cfar(data_cube, kernel, 5e-2)

    data_cube = torch.amax(data_cube, dim=(1,), keepdim=True) # sum CFAR across all channels and all frames
    data_cube_out = torch.zeros_like(data_cube[:9, ...])

    for i in range(9):
        data_cube_out[i] = torch.amax(data_cube[i*2:(i+1)*2, ...], dim=(0,), keepdim=False)

    # select first frame and channel (i.e., only frame and channel given CFAR), then add "channel" dimension for pytorch
    data_cube_out = data_cube_out[:, 0, :, :].unsqueeze(0)

    return data_cube_out


def rangeDopplerProcess1(data_cube, input_hdf5):
    data_cube = radar.rangeDoppler(data_cube, torch.hann_window)[:20, ...]


    data_cube_out = torch.zeros_like(data_cube[:10, ...])

    for i in range(10):
        data_cube_out[i] = torch.sum(data_cube[i*2:(i+1)*2, ...], dim=(0,), keepdim=False) / 2

    data_cube = data_cube_out
    data_cube = radar.todB(torch.abs(data_cube))
    max_per_frame = torch.amax(torch.abs(data_cube), dim=(2, 3,), keepdim=True)

    dynamic_range = 60

    data_cube = data_cube - max_per_frame + dynamic_range
    data_cube = data_cube / dynamic_range
    
    data_cube[data_cube < 0] = 0
    data_cube[data_cube > 1] = 1

    data_cube = data_cube[:, 0, :, :].unsqueeze(0)
    return data_cube

def rangeDopplerProcess2(data_cube, input_hdf5):
    data_cube = radar.rangeDoppler(data_cube, torch.hann_window)[:20, ...]
    
    max_per_frame = torch.amax(torch.abs(data_cube)[2:, ...], dim=(2, 3,), keepdim=True)
        
                               
    data_cube = data_cube - torch.sum(data_cube[0:1], dim=0, keepdim=True)
    data_cube = data_cube[2:, ...]

    data_cube_out = torch.zeros_like(data_cube[:9, ...])
    max_per_frame_out = torch.zeros_like(max_per_frame[:9, ...])

    for i in range(9):
        data_cube_out[i] = torch.sum(data_cube[i*2:(i+1)*2, ...], dim=(0,), keepdim=False) / 2
        max_per_frame_out[i] = torch.sum(max_per_frame[i*2:(i+1)*2, ...], dim=(0,), keepdim=False) / 2

    data_cube = data_cube_out
    max_per_frame = max_per_frame_out
    max_per_frame = radar.todB(max_per_frame)
    data_cube = radar.todB(torch.abs(data_cube))
    

    dynamic_range = 60

    data_cube = data_cube - max_per_frame + dynamic_range
    data_cube = data_cube / dynamic_range
    
    data_cube[data_cube < 0] = 0
    data_cube[data_cube > 1] = 1

    data_cube = data_cube[:, 0, :, :].unsqueeze(0)
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
    
    data_cube = data_cube[...,range_low:range_high] # remember to think... has the map had a fliplr?
    data_cube = data_cube[...,velocity_low:velocity_high, :]

    return data_cube

def cfarCrop2(data_cube, file_hdf5):
    range_res = read.rangeResolution(file_hdf5)
    velocity_res = read.velocityResolution(file_hdf5)
    
    range_low = int(0.35/range_res) # min range at 0.25 m
    range_high = int(0.85/range_res) # max range at 0.5 m

    # restrict doppler to useful portion
    velocity_centre = data_cube.shape[-2]//2
    velocity_low = velocity_centre - int(10/velocity_res)
    velocity_high = velocity_centre + int(10/velocity_res)
    
    data_cube = data_cube[...,range_low:range_high] # remember to think... has the map had a fliplr?
    data_cube = data_cube[...,velocity_low:velocity_high, :]

    return data_cube

def microDopplerProcess1(data_cube, input_hdf5):
    
    # data_cube = read.centreFrames(data_cube, 20)
    data_cube = data_cube[:20, ...]

    range_res = read.rangeResolution(input_hdf5)

    spectrogram = radar.microDoppler(data_cube, (8, 14), n_fft=128, win_length=16, hop_length=1)


    spectrogram = torch.abs(spectrogram)
    spectrogram = radar.todB(spectrogram)

    
    spectrogram = spectrogram -35
    spectrogram = spectrogram / 40

    spectrogram[spectrogram < 0] = 0
    spectrogram[spectrogram > 1] = 1
    
    spectrogram = spectrogram.unsqueeze(0)

    return spectrogram


def angleProcess1(data_cube, input_hdf5):

    # Concatenate the appropriate channels (assumed you want channels 0:4 and 8:12)
    data_cube = torch.cat([data_cube[: ,0:4, ...], data_cube[:, 8:12, ...]], dim=1)

    # Determine angles to analyse
    num_angles = 16
    angle_spacing = 2
    start_angle = - num_angles*angle_spacing/2

    # Compute angles in radians (torch.sin expects radians)
    angles = torch.arange(start_angle, start_angle + num_angles * angle_spacing, angle_spacing) * torch.pi / 180

    # Data cube shape: [frames, channels, doppler, range]
    d = 0.5
    data_cube_angles = torch.zeros((data_cube.shape[0], num_angles, data_cube.shape[2], data_cube.shape[3]), dtype=data_cube.dtype)

    # Create the steering matrix for all angles at once (shape: [num_angles, channels])
    channels = torch.arange(data_cube.shape[1])  # Assuming data_cube.shape[1] is the number of channels
    steering_matrix = torch.exp(-2j * torch.pi * d * channels.unsqueeze(0) * torch.sin(angles.unsqueeze(1)))

    # Apply steering matrix to the data_cube
    # Expand the steering matrix to match the shape of data_cube [num_angles, channels, 1, 1]
    steering_matrix = steering_matrix.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # Shape: [1, num_angles, channels, 1, 1]

    # Expand data_cube to shape [frames, 1, channels, doppler, range] for broadcasting
    data_cube_expanded = data_cube.unsqueeze(1)  # Shape: [frames, 1, channels, doppler, range]

    # Multiply data_cube with steering_matrix, then sum over the channels dimension (dim=2)
    data_cube_angles = torch.sum(data_cube_expanded * steering_matrix, dim=2)



    # generate CFAR over all frames
    kernel = torch.zeros((25, 11))
    kernel[12, :3] = 1
    kernel[:7, 5] = 1
    kernel[-7:, 5] = 1
    data_cube_angles = radar.cfar(data_cube_angles, kernel, 5e-2)
    data_cube_angles = torch.amax(data_cube_angles, dim=(0,)) # sum CFAR across all all frames

    # data_cube_angles = torch.abs(data_cube_angles)
    # data_cube_angles = torch.amax(data_cube_angles, dim=(0,))

    data_cube_angles = data_cube_angles.unsqueeze(0)

    # select first frame (i.e., only frame and channel given CFAR), then add "channel" dimension for pytorch
    return data_cube_angles


def angleProcess2(data_cube, input_hdf5):

    # Concatenate the appropriate channels (assumed you want channels 0:4 and 8:12)
    data_cube = torch.cat([data_cube[: ,0:4, ...], data_cube[:, 8:12, ...]], dim=1)

    data_cube_angles = radar.angleFft(data_cube)



    # generate CFAR over all frames
    kernel = torch.zeros((25, 11))
    kernel[12, :3] = 1
    kernel[:7, 5] = 1
    kernel[-7:, 5] = 1
    data_cube_angles = radar.cfar(data_cube_angles, kernel, 5e-2)
    data_cube_angles = torch.amax(data_cube_angles, dim=(0,)) # sum CFAR across all all frames

    # data_cube_angles = torch.abs(data_cube_angles)
    # data_cube_angles = torch.amax(data_cube_angles, dim=(0,))

    data_cube_angles = data_cube_angles.unsqueeze(0)

    # select first frame (i.e., only frame and channel given CFAR), then add "channel" dimension for pytorch
    return data_cube_angles


def angleCrop1(data_cube, file_hdf5):
    range_res = read.rangeResolution(file_hdf5)
    velocity_res = read.velocityResolution(file_hdf5)
    
    range_low = int(0.3/range_res) # min range at 0.4 m
    range_high = int(0.9/range_res) # max range at 0.8 m

    # restrict doppler to useful portion
    velocity_centre = data_cube.shape[-2]//2
    velocity_low = velocity_centre - int(3/velocity_res)
    velocity_high = velocity_centre + int(3/velocity_res)

    
    data_cube = data_cube[...,range_low:range_high] # remember to think... has the map had a fliplr?
    data_cube = data_cube[...,velocity_low:velocity_high, :]

    return data_cube





def handLocateProcess1(data_cube, file_hdf5):

    data_cube = radar.rangeDoppler(data_cube, torch.hann_window)

    # range res is useful for a few calculations down the line
    range_res = read.rangeResolution(file_hdf5)
    range_low = int(0.25/range_res)
    range_high = int(0.75/range_res)

    # choose the best frames to analyse based on average power in the selected region of the range doppler map
    best_frame = radar.bestFrame(data_cube, range_low, range_high)
    start_frame = best_frame - 10
    if start_frame < 0: start_frame = 0
    if start_frame+20 > data_cube.shape[0]: start_frame = data_cube.shape[0] - 20
    data_cube = data_cube[start_frame:start_frame+20, ...]
    # data_cube = data_cube[:20, ...]

    # generate CFAR over all frames
    kernel = torch.zeros((25, 11))
    kernel[12, :3] = 1
    kernel[:7, 5] = 1
    kernel[-7:, 5] = 1
    cfar = radar.cfar(data_cube - torch.sum(data_cube[0:2], dim=0, keepdim=True)/2, kernel, 1e-2)
    
    # find the first target:
    range_bin = radar.bestRangeBin(cfar, range_low, range_high)

    # print(range_bin * range_res)

    #  find max in each frame (this should be the max value of the crosstalk)
    max_per_frame = torch.amax(torch.abs(data_cube)[2:, ...], dim=(2, 3,), keepdim=True)
    
    # crop down to appropriate range interval:
    data_cube = data_cube[..., range_bin-2:range_bin+4]


        
    # remove clutter
    data_cube = data_cube - (torch.sum(data_cube[0:2], dim=0, keepdim=True) / 2)
    data_cube = data_cube[2:, ...]
    
    # sum pairs of frames for time dimension in 3D CNN... a bit like an early max pooling stage
    data_cube_out = torch.zeros_like(data_cube[:9, ...])
    max_per_frame_out = torch.zeros_like(max_per_frame[:9, ...])

    for i in range(9):
        data_cube_out[i] = torch.sum(data_cube[i*2:(i+1)*2, ...], dim=(0,), keepdim=False) / 2
        max_per_frame_out[i] = torch.sum(max_per_frame[i*2:(i+1)*2, ...], dim=(0,), keepdim=False) / 2

    data_cube = data_cube_out
    max_per_frame = max_per_frame_out
    max_per_frame = radar.todB(max_per_frame)
    data_cube = radar.todB(torch.abs(data_cube))
    
    # normalise around the calculated max and with the given dynamic range
    dynamic_range = 50
    data_cube = data_cube - max_per_frame + dynamic_range
    data_cube = data_cube / dynamic_range
    
    # Get rid of any additional values (mostly applicable to cutting out noise below 60 dB)
    data_cube[data_cube < 0] = 0
    data_cube[data_cube > 1] = 1

    # Select first channel for final run through CNN
    # also add the CNN channels dimension
    data_cube = data_cube[:, 0, :, :].unsqueeze(0)
    return data_cube


def handLocateProcess2(data_cube, file_hdf5):

    data_cube = radar.rangeDoppler(data_cube, torch.hann_window)

    # range res is useful for a few calculations down the line
    range_res = read.rangeResolution(file_hdf5)
    range_low = int(0.25/range_res)
    range_high = int(0.75/range_res)

    # choose the best frames to analyse based on average power in the selected region of the range doppler map
    best_frame = radar.bestFrame(data_cube, range_low, range_high)
    start_frame = best_frame - 10
    if start_frame < 0: start_frame = 0
    if start_frame+20 > data_cube.shape[0]: start_frame = data_cube.shape[0] - 20
    data_cube = data_cube[start_frame:start_frame+20, ...]
    # data_cube = data_cube[:20, ...]

    # generate CFAR over all frames
    kernel = torch.zeros((25, 11))
    kernel[12, :3] = 1
    kernel[:7, 5] = 1
    kernel[-7:, 5] = 1
    cfar = radar.cfar(data_cube - torch.sum(data_cube[0:2], dim=0, keepdim=True)/2, kernel, 1e-2)
    
    # find the first target:
    range_bin = radar.bestRangeBin(cfar, range_low, range_high)

    spectrogram = radar.microDoppler(data_cube, range_bin, n_fft=128, hop_length = 2, win_length=32)
    
    dynamic_range = 60
    spectrogram = radar.todB(torch.abs(spectrogram))
    spectrogram = spectrogram - torch.amax(spectrogram) + dynamic_range
    spectrogram = spectrogram / dynamic_range


    
    # Get rid of any additional values (mostly applicable to cutting out noise below 60 dB)
    spectrogram[spectrogram < 0] = 0
    spectrogram[spectrogram > 1] = 1

    # also add the CNN channels dimension
    spectrogram = spectrogram.unsqueeze(0)
    return spectrogram

def handLocateProcess3(data_cube, file_hdf5):
    data_cube = radar.rangeDoppler(data_cube, torch.hann_window)

    # range res is useful for a few calculations down the line
    range_res = read.rangeResolution(file_hdf5)
    range_low = 7#int(0.4/range_res)
    range_high = 12#int(0.65/range_res)

    # choose the best frames to analyse based on average power in the selected region of the range doppler map
    best_frame = radar.bestFrame(data_cube, range_low, range_high)
    start_frame = best_frame - 10
    if start_frame < 0: start_frame = 0
    if start_frame+20 > data_cube.shape[0]: start_frame = data_cube.shape[0] - 20
    data_cube = data_cube[start_frame:start_frame+20, ...]
    # data_cube = data_cube[:20, ...]
    
    # generate CFAR over all frames
    kernel = torch.zeros((25, 11))
    kernel[12, :3] = 1
    kernel[:7, 5] = 1
    kernel[-7:, 5] = 1
    cfar = radar.cfar(data_cube - torch.sum(data_cube[0:2], dim=0, keepdim=True)/2, kernel, 1e-2)
    
    # find the first target:
    range_bin = 9

    # print(range_bin * range_res)

    #  find max in each frame (this should be the max value of the crosstalk)
    max_per_frame = torch.amax(torch.abs(data_cube)[1:, ...], dim=(2, 3,), keepdim=True)
    
    # crop down to appropriate range interval:
    data_cube = data_cube[..., 24:74, range_bin-2:range_bin+4]


        
    # remove clutter
    data_cube = (data_cube - data_cube[0:1, ...])
    data_cube = data_cube[1:, ...]

    data_cube = torch.sum(data_cube, dim=(1,), keepdim=True) / 12

    data_cube = radar.todB(torch.abs(data_cube))
    max_per_frame = radar.todB(max_per_frame)


    # normalise around the calculated max and with the given dynamic range
    dynamic_range = 50
    data_cube = data_cube - torch.amax(data_cube, dim=(2, 3,), keepdim=True) + dynamic_range
    data_cube = data_cube / dynamic_range



    # Get rid of any additional values (mostly applicable to cutting out noise below 60 dB)
    data_cube[data_cube < 0] = 0
    data_cube[data_cube > 1] = 1

    means = data_cube.mean(dim=(2, 3, ), keepdim=True)
    std_devs = data_cube.std(dim=(2, 3, ), keepdim=True)

    data_cube = (data_cube - means) / std_devs

    # Select first channel for final run through CNN
    # also add the CNN channels dimension
    data_cube = data_cube[:, 0, :, :].unsqueeze(0)
    return data_cube

def handLocateProcess4(data_cube, file_hdf5):
    data_cube = radar.rangeDoppler(data_cube, torch.hann_window)

    # range res is useful for a few calculations down the line
    range_res = read.rangeResolution(file_hdf5)
    range_low = 7#int(0.4/range_res)
    range_high = 12#int(0.65/range_res)

    # choose the best frames to analyse based on average power in the selected region of the range doppler map
    best_frame = radar.bestFrame(data_cube, range_low, range_high)
    start_frame = best_frame - 10
    if start_frame < 0: start_frame = 0
    if start_frame+20 > data_cube.shape[0]: start_frame = data_cube.shape[0] - 20
    data_cube = data_cube[start_frame:start_frame+20, ...]
    # data_cube = data_cube[:20, ...]
    
    # generate CFAR over all frames
    kernel = torch.zeros((25, 11))
    kernel[12, :3] = 1
    kernel[:7, 5] = 1
    kernel[-7:, 5] = 1
    cfar = radar.cfar(data_cube - torch.sum(data_cube[0:1], dim=0, keepdim=True), kernel, 1e-2)
    
    # find the first target:
    range_bin = 9
    
    # crop down to appropriate range interval:
    cfar = cfar[..., 24:74, range_bin-2:range_bin+4]

    # Select first channel for final run through CNN
    # also add the CNN channels dimension
    cfar = cfar[:, 0, :, :].unsqueeze(0)
    return cfar

def rangeDoppler2dProcessFinal(data_cube, file_hdf5):
    # compute range-doppler
    data_cube = radar.rangeDoppler(data_cube, torch.hann_window)

    data_cube = torch.sum(data_cube, dim=(1,), keepdim=True) / 12
    # range res is useful for a few calculations down the line
    range_low = 7
    range_high = 13

    # choose the best frames to analyse based on average power in the selected region of the range doppler map
    best_frame = radar.bestFrame(data_cube, range_low, range_high)
    start_frame = best_frame - 10
    if start_frame < 0: start_frame = 0
    if start_frame+20 > data_cube.shape[0]: start_frame = data_cube.shape[0] - 20
    data_cube = data_cube[start_frame:start_frame+20, ...]

    # remove clutter
    data_cube = (data_cube - data_cube[0:1, ...])
    data_cube = data_cube[1:, ...]
    
    # crop down to appropriate range interval:
    data_cube = data_cube[..., 24:74, range_low:range_high]

    # sum all frames into one
    data_cube = torch.sum(data_cube, dim=(0, ), keepdim=True) / 19

    data_cube = radar.todB(torch.abs(data_cube))
    # max_per_frame = torch.amax(data_cube, dim=(2, 3), keepdim=True)
    max_per_frame = torch.amax(data_cube)

    # normalise around the calculated max and with the given dynamic range
    dynamic_range = 40
    data_cube = data_cube - max_per_frame + dynamic_range
    data_cube = data_cube / dynamic_range

    # Get rid of any additional values (mostly applicable to cutting out noise below 60 dB)
    data_cube[data_cube < 0] = 0
    data_cube[data_cube > 1] = 1

    # Select first channel for final run through CNN
    # also add the CNN channels dimension
    data_cube = data_cube[0, 0, :, :].unsqueeze(0)
    return data_cube


def rangeDoppler3dProcessFinal(data_cube, file_hdf5, return_start_frame = False):
    # compute range-doppler
    data_cube = radar.rangeDoppler(data_cube, torch.hann_window)

    data_cube = torch.sum(data_cube, dim=(1,), keepdim=True) / 12
    # range res is useful for a few calculations down the line
    range_low = 7
    range_high = 13

    # choose the best frames to analyse based on average power in the selected region of the range doppler map
    best_frame = radar.bestFrame(data_cube, range_low, range_high)
    start_frame = best_frame - 10
    if start_frame < 0: start_frame = 0
    if start_frame+20 > data_cube.shape[0]: start_frame = data_cube.shape[0] - 20
    data_cube = data_cube[start_frame:start_frame+20, ...]

    # remove clutter
    data_cube = (data_cube - data_cube[0:1, ...])
    data_cube = data_cube[1:, ...]
    
    # crop down to appropriate range interval:
    data_cube = data_cube[..., 24:74, range_low:range_high]


    data_cube = radar.todB(torch.abs(data_cube))
    # max_per_frame = torch.amax(data_cube, dim=(2, 3), keepdim=True)
    max_per_frame = torch.amax(data_cube)

    # normalise around the calculated max and with the given dynamic range
    dynamic_range = 40
    data_cube = data_cube - max_per_frame + dynamic_range
    data_cube = data_cube / dynamic_range

    # Get rid of any additional values (mostly applicable to cutting out noise below 60 dB)
    data_cube[data_cube < 0] = 0
    data_cube[data_cube > 1] = 1

    # Select first channel for final run through CNN
    # also add the CNN channels dimension
    data_cube = data_cube[:, 0, :, :].unsqueeze(0)

    if return_start_frame:
        return data_cube, (start_frame + 1)
    return data_cube

def cfarProcessFinal(data_cube, file_hdf5, return_start_frame = False):
    data_cube = radar.rangeDoppler(data_cube, torch.hann_window)
    
    # sum over channels to improve SNR
    data_cube = torch.sum(data_cube, dim=(1,), keepdim=True) / 12
    
    # range res is useful for a few calculations down the line
    range_low = 7
    range_high = 13

    # choose the best frames to analyse based on average power in the selected region of the range doppler map
    best_frame = radar.bestFrame(data_cube, range_low, range_high)
    start_frame = best_frame - 10
    if start_frame < 0: start_frame = 0
    if start_frame+20 > data_cube.shape[0]: start_frame = data_cube.shape[0] - 20
    data_cube = data_cube[start_frame:start_frame+20, ...]

    # remove clutter
    data_cube = (data_cube - data_cube[0:1, ...])
    data_cube = data_cube[1:, ...]

    
    # generate CFAR over all frames
    kernel = torch.zeros((25, 11))
    kernel[12, :3] = 1
    kernel[:7, 5] = 1
    kernel[-7:, 5] = 1
    cfar = radar.cfar(data_cube, kernel, 0.3)
    
    # crop down to appropriate range interval:
    cfar = cfar[..., 24:74, range_low:range_high]

    # Select first channel for final run through CNN
    # also add the CNN channels dimension
    cfar = cfar[:, 0, :, :].unsqueeze(0)
    if return_start_frame:
        return cfar, start_frame + 1
    return cfar

def beamformingProcessFinal(data_cube, file_hdf5):
    data_cube = radar.rangeDoppler(data_cube, torch.hann_window)

    # range res is useful for a few calculations down the line
    range_low = 7
    range_high = 13

    # choose the best frames to analyse based on average power in the selected region of the range doppler map
    best_frame = radar.bestFrame(data_cube, range_low, range_high)
    start_frame = best_frame - 10
    if start_frame < 0: start_frame = 0
    if start_frame+20 > data_cube.shape[0]: start_frame = data_cube.shape[0] - 20
    data_cube = data_cube[start_frame:start_frame+20, ...]

    # remove clutter
    data_cube = (data_cube - data_cube[0:1, ...])
    data_cube = data_cube[1:, ...]

    # Concatenate the appropriate channels
    data_cube = torch.cat([data_cube[: ,0:4, ...], data_cube[:, 8:12, ...]], dim=1)

    # Determine angles to analyse
    num_angles = 16
    angle_spacing = 2
    start_angle = - num_angles*angle_spacing/2 + 1

    # Compute angles in radians (torch.sin expects radians)
    angles = torch.arange(start_angle, start_angle + num_angles * angle_spacing, angle_spacing) * torch.pi / 180

    # Data cube shape: [frames, channels, doppler, range]
    d = 0.5
    data_cube_angles = torch.zeros((data_cube.shape[0], num_angles, data_cube.shape[2], data_cube.shape[3]), dtype=data_cube.dtype)

    # Create the steering matrix for all angles at once (shape: [num_angles, channels])
    channels = torch.arange(data_cube.shape[1])  # Assuming data_cube.shape[1] is the number of channels
    steering_matrix = torch.exp(-2j * torch.pi * d * channels.unsqueeze(0) * torch.sin(angles.unsqueeze(1)))

    # Apply steering matrix to the data_cube
    # Expand the steering matrix to match the shape of data_cube [num_angles, channels, 1, 1]
    steering_matrix = steering_matrix.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # Shape: [1, num_angles, channels, 1, 1]

    # Expand data_cube to shape [frames, 1, channels, doppler, range] for broadcasting
    data_cube_expanded = data_cube.unsqueeze(1)  # Shape: [frames, 1, channels, doppler, range]

    # Multiply data_cube with steering_matrix, then sum over the channels dimension (dim=2)
    data_cube_angles = torch.sum(data_cube_expanded * steering_matrix, dim=2)
    
    data_cube = data_cube_angles

    # sum all the frames into one
    data_cube = torch.sum(data_cube, dim=(0,), keepdim=True) / 19


    # crop down to appropriate range interval:
    data_cube = data_cube[..., 24:74, range_low:range_high]

    data_cube = radar.todB(torch.abs(data_cube))
    # max_per_frame = torch.amax(data_cube, dim=(2, 3), keepdim=True)
    max_per_frame = torch.amax(data_cube)


    # normalise around the calculated max and with the given dynamic range
    dynamic_range = 40
    data_cube = data_cube - max_per_frame + dynamic_range
    data_cube = data_cube / dynamic_range

    # Get rid of any additional values (mostly applicable to cutting out noise below 60 dB)
    data_cube[data_cube < 0] = 0
    data_cube[data_cube > 1] = 1

    
    data_cube = data_cube[0, :, :, :].unsqueeze(0)
    # select first frame (i.e., only frame and channel given CFAR), then add "channel" dimension for pytorch
    return data_cube

def angleProcessFinal(data_cube, file_hdf5):

    data_cube = radar.rangeDoppler(data_cube, torch.hann_window)

    # range res is useful for a few calculations down the line
    range_low = 7
    range_high = 13

    # choose the best frames to analyse based on average power in the selected region of the range doppler map
    best_frame = radar.bestFrame(data_cube, range_low, range_high)
    start_frame = best_frame - 10
    if start_frame < 0: start_frame = 0
    if start_frame+20 > data_cube.shape[0]: start_frame = data_cube.shape[0] - 20
    data_cube = data_cube[start_frame:start_frame+20, ...]

    # remove clutter
    data_cube = (data_cube - data_cube[0:1, ...])
    data_cube = data_cube[1:, ...]

    # Concatenate the appropriate channels
    data_cube = torch.cat([data_cube[: ,0:4, ...], data_cube[:, 8:12, ...]], dim=1)

    data_cube = radar.angleFft(data_cube)
    
    # sum all the frames into one
    data_cube = torch.sum(data_cube, dim=(0,), keepdim=True) / 19

    # crop down to appropriate range interval:
    data_cube = data_cube[..., 24:74, range_low:range_high]

    data_cube = radar.todB(torch.abs(data_cube))
    # max_per_frame = torch.amax(data_cube, dim=(2, 3), keepdim=True)
    max_per_frame = torch.amax(data_cube)


    # normalise around the calculated max and with the given dynamic range
    dynamic_range = 40
    data_cube = data_cube - max_per_frame + dynamic_range
    data_cube = data_cube / dynamic_range

    # Get rid of any additional values (mostly applicable to cutting out noise below 60 dB)
    data_cube[data_cube < 0] = 0
    data_cube[data_cube > 1] = 1

    
    data_cube = data_cube[0, :, :, :].unsqueeze(0)
    # select first frame (i.e., only frame and channel given CFAR), then add "channel" dimension for pytorch
    return data_cube


def microDopplerProcessFinal(data_cube, file_hdf5, return_start_frame=False):

    # compute range-doppler
    data_cube = radar.rangeFft(data_cube, torch.hann_window)

    data_cube = torch.sum(data_cube, dim=(1,), keepdim=True) / 12 # sum channels for SNR reduction

    # range res is useful for a few calculations down the line
    range_low = 7
    range_high = 13

    # choose the best frames to analyse based on average power in the selected region of the range doppler map
    best_frame = radar.bestFrame(data_cube, range_low, range_high)
    start_frame = best_frame - 10
    if start_frame < 0: start_frame = 0
    if start_frame+20 > data_cube.shape[0]: start_frame = data_cube.shape[0] - 20
    data_cube = data_cube[start_frame:start_frame+20, ...]

    # remove clutter
    data_cube = (data_cube - data_cube[0:1, ...])
    data_cube = data_cube[1:, ...]
    
    # crop down to appropriate range interval:
    data_cube = data_cube[..., range_low:range_high]

    
    data_cube = torch.sum(data_cube, dim=(3, ), keepdim=True) # sum all selected range bins for spectrogram
    spectrogram = radar.microDoppler(data_cube, n_fft=128, hop_length = 2, win_length=32, doRangeFft=False)

    spectrogram = radar.todB(torch.abs(spectrogram))
    
    # normalise around the calculated max and with the given dynamic range
    dynamic_range = 40
    spectrogram = spectrogram - torch.amax(spectrogram) + dynamic_range
    spectrogram = spectrogram / dynamic_range

    # Get rid of any additional values (mostly applicable to cutting out noise below 60 dB)
    spectrogram[spectrogram < 0] = 0
    spectrogram[spectrogram > 1] = 1

    # also add the CNN channels dimension
    spectrogram = spectrogram.unsqueeze(0)

    if return_start_frame:
        return spectrogram, start_frame + 1
    return spectrogram

def rangeDoppler3dFakeProcessFinal(data_cube, file_hdf5):
    # compute range-doppler
    data_cube = radar.rangeDoppler(data_cube, torch.hann_window)

    data_cube = torch.sum(data_cube, dim=(1,), keepdim=True) / 12
    # range res is useful for a few calculations down the line
    range_low = 7
    range_high = 13

    # choose the best frames to analyse based on average power in the selected region of the range doppler map
    best_frame = radar.bestFrame(data_cube, range_low, range_high)
    start_frame = best_frame - 10
    if start_frame < 0: start_frame = 0
    if start_frame+20 > data_cube.shape[0]: start_frame = data_cube.shape[0] - 20
    data_cube = data_cube[start_frame:start_frame+20, ...]

    # remove clutter
    data_cube = (data_cube - data_cube[0:1, ...])
    data_cube = data_cube[1:, ...]
    
    # crop down to appropriate range interval:
    data_cube = data_cube[..., 24:74, range_low:range_high]

    # sum all frames into one
    data_cube = torch.sum(data_cube, dim=(0, ), keepdim=True) / 19

    data_cube = radar.todB(torch.abs(data_cube))
    # max_per_frame = torch.amax(data_cube, dim=(2, 3), keepdim=True)
    max_per_frame = torch.amax(data_cube)

    # normalise around the calculated max and with the given dynamic range
    dynamic_range = 40
    data_cube = data_cube - max_per_frame + dynamic_range
    data_cube = data_cube / dynamic_range

    # Get rid of any additional values (mostly applicable to cutting out noise below 60 dB)
    data_cube[data_cube < 0] = 0
    data_cube[data_cube > 1] = 1

    # Select first channel for final run through CNN
    # also add the CNN channels dimension
    data_cube = data_cube.expand(16, -1, -1, -1)
    data_cube = data_cube[:, 0, :, :].unsqueeze(0)

    return data_cube