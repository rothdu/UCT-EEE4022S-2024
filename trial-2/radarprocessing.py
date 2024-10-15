import torch

from scipy import constants
import matplotlib.pyplot as plt

def rangeFft(data_cube, window=None):
    '''Compute FFT over range axis with the specific windowing function'''
    if window:
        window_tensor = window(data_cube.shape[3])[None, None, None, ...] # create window tensor with correct number of axes
        data_cube = data_cube*window_tensor.expand(data_cube.shape) #
    range_fft = torch.flip(torch.fft.fft(data_cube, dim=3), (3,))
    return range_fft

def dopplerFft(data_cube, window=None):
    '''Compute FFT over doppler axis with the specified windowing funtion'''
    if window:
        window_tensor = window(data_cube.shape[2])[None, None, ..., None] # create window tensor with correct number of axes
        data_cube = data_cube*window_tensor.expand(data_cube.shape) #
    doppler_fft = torch.fft.fftshift(torch.fft.fft(data_cube, dim=2), (2,))
    return doppler_fft

def angleFft(data_cube, window=None):
    '''Compute FFT over doppler axis with the specified windowing funtion'''
    if window:
        window_tensor = window(data_cube.shape[2])[None, ..., None, None] # create window tensor with correct number of axes
        data_cube = data_cube*window_tensor.expand(data_cube.shape) #
    doppler_fft = torch.fft.fftshift(torch.fft.fft(data_cube, dim=1), (1,))
    return doppler_fft

def rangeDoppler(data_cube, window=None):
    '''Get range doppler tensor for inputting data cube tensor'''
    data_cube = dopplerFft(rangeFft(data_cube, window), window)
    return data_cube

def todB(data_cube):
    '''Convert input torch tensor to dB'''
    return 20*torch.log10(data_cube)

def cfar(data_cube, kernel, probability_false_alarm=1e-3):
    '''Return radar CFAR output from range-doppler map'''
    # currently expects a 2d kernel
    num_training_cells = torch.sum(kernel) # number of training cells
    alpha = num_training_cells * (torch.pow(probability_false_alarm, -1/num_training_cells) -1) # threshold gain

    # convert data cube to power
    data_cube = torch.pow(torch.abs(data_cube), 2)

    # cfar kernel
    # snr = torch.zeros(torch.shape(data_cube)) # extra stuff needs to be done to find SNR as well if wanted
    noise_cube = torch.zeros_like(data_cube[0:1, 0:1, ...])
    noise_cube[:, :, 0:kernel.shape[0], 0:kernel.shape[1]] = kernel # zero pad in range and doppler dimensions
    # print(noise_cube)
    noise_cube = noise_cube.expand(data_cube.shape) # expand across all channels and frames

    noise_cube = torch.fft.ifft2(torch.conj(torch.fft.fft2(noise_cube))*torch.fft.fft2(data_cube)) # do the fancy frequency domain based convolution
    # noise_cube = torch.roll(noise_cube, kernel.shape[0]//2, dims=2) # Not sure I understand this step but it is apparently necessary
    # noise_cube = torch.roll(noise_cube, kernel.shape[1]//2, dims=1) # TODO: hmmm

    data_cube = torch.where(data_cube.gt(torch.abs(noise_cube)*alpha), 1.0, 0) # generate CFAR for each range-doppler map in the cube

    return data_cube

def generateDopplerKernel(len, guard_len):
    '''Generate a a kernel for the doppler dimension with specified length and guard length'''
    len = 2*(len//2)+1 #  might make a kernel bigger than desired
    guard_len = 2*(guard_len//2)+1 # might make guard kernel bigger than desired
    unguarded_len = (len-guard_len)//2
    kernel = torch.ones(len, 1)
    kernel[unguarded_len:-unguarded_len] = 0
    return kernel.reshape((-1, 1))

def microDoppler(data_cube, n_fft=64, hop_length=2, win_length=16, range_window = torch.hann_window, doppler_window = torch.hann_window, doRangeFft=True):
    '''Create micro doppler spectrogram from radar data cube'''
    if doRangeFft:
        data_cube = rangeFft(data_cube, range_window)
    
    data_cube = data_cube[:, 0, ...]

    doppler_tensor = data_cube.reshape((-1,))

    if doppler_window:
        spectrogram = torch.stft(doppler_tensor, n_fft, hop_length, win_length, doppler_window(win_length)) # generate spectrogram
    else:
        spectrogram = torch.stft(doppler_tensor, n_fft, hop_length, win_length, window=None) # generate spectrogram
    spectrogram = torch.fft.fftshift(spectrogram, 0)
    return spectrogram

def bestFrame(data_cube, range_low, range_high):

    data_cube_analysis = data_cube[:, :, :, range_low:range_high]

    data_cube_analysis = torch.pow(torch.abs(data_cube_analysis), 2)

    data_cube_analysis = torch.sum(data_cube_analysis, dim=(2, 3,), keepdim=True)
    data_cube_analysis = data_cube_analysis[:, 0, 0, 0]

    best_frame = torch.argmax(data_cube_analysis)

    return best_frame

def bestRangeBin(cfar, range_low, range_high):
    analysis_cube = cfar[:, 0:1, :, range_low:range_high]
    analysis_cube = torch.sum(analysis_cube, dim=2, keepdim=True)
    analysis_cube = torch.amax(analysis_cube, dim=0, keepdim=True)

    for i, detection in  enumerate(analysis_cube.squeeze()):
        if detection >= 3:
            return range_low + i
    
    return range_low + (range_high-range_low)//2