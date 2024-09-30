import torch

import radarprocessing as radar

import readdata as read

def cfarProcess2(_hdf5): # TODO: Get rid of the label part
    '''Take in a data cube tensor and do the various radar processing'''
    data_cube = radar.rangeDoppler(data_cube, torch.hann_window) # compute range doppler
    data_cube = torch.fft.fftshift(data_cube, 2) # fftshift so that doppler axis is centred
    # TODO: Possibly add a fliplr?
    # calculate range and velocity resolutions

    # generate CFAR over all frames
    data_cube = radar.cfar(data_cube, radar.generateDopplerKernel(25, 11), 1e-3) # TODO: Put this back with a better cfar...
    # data_cube = radar.todB(torch.abs(data_cube))

    # restrict range to useful portion - bearing in mind that 0 is at the end without a fliplr
    range_max = data_cube.shape[3]*range_res
    range_low = int(0.25/range_res) # max range at 1m
    range_high = int(0.75/range_res) # minimum range at 0.4m

    # restrict doppler to useful portion
    velocity_max = data_cube.shape[2]//2 * velocity_res
    velocity_centre = data_cube.shape[2]//2
    velocity_low = velocity_centre - int(3/velocity_res)
    velocity_high = velocity_centre + int(3/velocity_res)
    
    # TODO: Get rid of this plotting section
    # fig, (ax1, ax2) = plt.subplots(2)
    # to_plot = data_cube[0, 0, ...].to('cpu')
    # ax1.imshow(to_plot, interpolation='none', extent=(range_max, 0, -velocity_max, velocity_max), aspect='auto')
    
    data_cube = data_cube[:,:,:,-range_high:-range_low]
    data_cube = data_cube[:,:,velocity_low:velocity_high, :]

    # to_plot = data_cube[0, 0, ...].to('cpu')
    # ax2.imshow(to_plot, interpolation='none', extent=(0.75, 0.25, -3, 3), aspect='auto')

    # plt.savefig(f"figures/{label}.png")
    # plt.close()


    # select first frame and channel (i.e., only frame and channel given CFAR), then add "channel" dimension for pytorch
    data_cube = data_cube[0, 0, :, :].unsqueeze(0)

    return data_cube

def cfarProcess1(data_cube, input_hdf5): # TODO: Get rid of the label part
    '''Take in a data cube tensor and do the various radar processing'''    
    data_cube = radar.rangeDoppler(data_cube, torch.hann_window) # compute range doppler
    # calculate range and velocity resolutions

    # generate CFAR over all frames
    data_cube = radar.cfar(data_cube, radar.generateDopplerKernel(25, 11), 1e-3) # TODO: Put this back with a better cfar...
    # data_cube = radar.todB(torch.abs(data_cube))

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

