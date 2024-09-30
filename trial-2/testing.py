import readdata as read
import radarprocessing as radar

import matplotlib.pyplot as plt
import h5py

import torch

import os

import cnnmodels as models

def cfarTestProcess(data_cube, range_res, velocity_res, label): # TODO: Get rid of the label part
    '''Take in a data cube tensor and do the various radar processing'''
    data_cube = radar.rangeDoppler(data_cube, torch.hann_window) # compute range doppler
    data_cube = torch.fft.fftshift(data_cube, 2) # fftshift so that doppler axis is centred
    # TODO: Possibly add a fliplr?
    # calculate range and velocity resolutions

    # generate CFAR over all frames
    # data_cube = radar.cfar(data_cube, radar.generateDopplerKernel(25, 11), 1e-6) # TODO: Put this back with a better cfar...
    data_cube = radar.todB(torch.abs(data_cube))

    # restrict range to useful portion - bearing in mind that 0 is at the end without a fliplr
    range_max = data_cube.shape[3]*range_res
    range_low = int(0.4/range_res) # max range at 1m
    range_high = int(1/range_res) # minimum range at 0.4m

    # restrict doppler to useful portion
    velocity_max = data_cube.shape[2]//2 * velocity_res
    velocity_centre = data_cube.shape[2]//2
    velocity_low = velocity_centre - int(10/velocity_res)
    velocity_high = velocity_centre + int(10/velocity_res)
    
    # TODO: Get rid of this plotting section
    fig, (ax1, ax2) = plt.subplots(2)
    to_plot = data_cube[0, 0, ...].to('cpu')
    ax1.imshow(to_plot, interpolation='none', extent=(0, range_max, -velocity_max, velocity_max), aspect='auto')
    
    data_cube = data_cube[:,:,:,range_low:range_high]
    data_cube = data_cube[:,:,velocity_low:velocity_high, :]

    to_plot = data_cube[0, 0, ...].to('cpu')
    ax2.imshow(to_plot, interpolation='none', extent=(0.4/range_res, 1/range_res, -velocity_max, velocity_max), aspect='auto')

    plt.savefig(f"figures/{label}.png")


    # select first frame and channel (i.e., only frame and channel given CFAR), then add "channel" dimension for pytorch
    data_cube = data_cube[0, 0, :, :].unsqueeze(0)

    return data_cube

def runTest():
    file_path = os.fsencode("/home/rtdug/UCT-EEE4022S-2024/trial-2/data/Experiment_2024-09-17_15-34-50_137_swipe_right.hdf5") # path to the hdf5 file
    file_hdf5 = h5py.File(file_path) # load the hdf5 file

    data_cube = read.radarDataTensor(file_hdf5)
    data_cube = radar.rangeDoppler(data_cube)
    data_cube = torch.flip(torch.fft.fftshift(data_cube, 2), (3,))
    data_cube = radar.cfar(data_cube, radar.generateDopplerKernel(25, 11), 1e-6)
    # data_cube = radar.todB(torch.abs(data_cube))
    # data_cube = torch.amax(data_cube, (0, 1), keepdim=True)


    range_res = read.rangeResolution(file_hdf5)
    velocity_res = read.velocityResolution(file_hdf5)



    
    fig, (ax1, ax2) = plt.subplots(2)


    
    velocity_max = data_cube.shape[2]//2*velocity_res
    range_max = data_cube.shape[3]*range_res



    ax1.imshow(data_cube[0,0,...], interpolation='none', extent=(0, range_max, -velocity_max, velocity_max), aspect='auto')    

    range_high = int(0.75/range_res)
    range_low = int(0.25/range_res)
    velocity_high = int(-2/velocity_res)
    velocity_low = int(2/velocity_res)

    ax2.imshow(data_cube[0, 0, velocity_low:velocity_high, range_low:range_high], interpolation='none', extent=(0.25, 0.75, -2, 2), aspect='auto')
    
    plt.savefig("/home/rtdug/UCT-EEE4022S-2024/sample-figs/fig.png")
    plt.close()

def microDopplerTest():
    file_path = os.fsencode("/home/rtdug/UCT-EEE4022S-2024/trial-2/data/Experiment_2024-09-17_15-34-50_137_swipe_right.hdf5") # path to the hdf5 file
    file_hdf5 = h5py.File(file_path) # load the hdf5 file

    data_cube = read.radarDataTensor(file_hdf5)
    num_frames_analysed = 20
    discard_num = int((data_cube.shape[0]-num_frames_analysed)/2)
    data_cube = data_cube[discard_num:discard_num+num_frames_analysed, ...]

    range_res = read.rangeResolution(file_hdf5)

    print(data_cube.shape)

    spectrogram = radar.microDoppler(data_cube, range_bin= int(0.5/range_res))

    spectrogram = spectrogram.to('cpu')
    spectrogram = torch.abs(spectrogram)

    fig, ax = plt.subplots()

    ax.imshow(spectrogram, interpolation='none')

    plt.savefig("/home/rtdug/UCT-EEE4022S-2024/sample-figs/fig.png")
    plt.close()



def main():
    # runTest()
    microDopplerTest()

if __name__ == "__main__":
    main()