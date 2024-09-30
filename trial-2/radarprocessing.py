import torch

from scipy import constants

def rangeFft(data_cube, window=None):
    '''Compute FFT over range axis with the specific windowing function'''
    if window:
        window_tensor = window(data_cube.shape[3])[None, None, None, ...] # create window tensor with correct number of axes
        data_cube = data_cube*window_tensor.expand(data_cube.shape) #
    range_fft = torch.fft.fft(data_cube, dim=3)
    return range_fft

def dopplerFft(data_cube, window=None):
    '''Compute FFT over doppler axis with the specified windowing funtion'''
    if window:
        window_tensor = window(data_cube.shape[2])[None, None, ..., None] # create window tensor with correct number of axes
        data_cube = data_cube*window_tensor.expand(data_cube.shape) #
    doppler_fft = torch.fft.fft(data_cube, dim=2)
    return doppler_fft

def rangeDoppler(data_cube, window=None):
    '''Get range doppler tensor for inputting data cube tensor'''
    data_cube = dopplerFft(rangeFft(data_cube, window), window)
    data_cube = torch.flip(torch.fft.fftshift(data_cube, (2,)), (3,))
    return data_cube

def todB(data_cube):
    '''Convert input torch tensor to dB'''
    return 20*torch.log10(data_cube)

def cfar(data_cube, kernel, probability_false_alarm=1e-6):
    '''Return radar CFAR output from range-doppler map'''
    # currently expects a 2d kernel
    num_training_cells = torch.sum(kernel) # number of training cells
    alpha = num_training_cells * (torch.pow(probability_false_alarm, -1/num_training_cells) -1) # threshold gain

    # convert data cube to power
    data_cube = torch.pow(torch.abs(data_cube), 2)

    # cfar kernel
    # snr = torch.zeros(torch.shape(data_cube)) # extra stuff needs to be done to find SNR as well if wanted
    noise_cube = torch.zeros(data_cube.shape[2:], dtype=data_cube.dtype)
    noise_cube[0:kernel.shape[0], 0:kernel.shape[1]] = kernel # zero pad in range and doppler dimensions
    # print(noise_cube)
    noise_cube = noise_cube[None, None, ...].expand(data_cube.shape) # expand across all channels and frames

    noise_cube = torch.fft.ifft2(torch.conj(torch.fft.fft2(noise_cube))*torch.fft.fft2(data_cube)) # do the fancy frequency domain based convolution
    noise_cube = torch.roll(noise_cube, kernel.shape[0]//2, dims=2) # Not sure I understand this step but it is apparently necessary

    data_cube = torch.where(data_cube.gt(torch.abs(noise_cube)*alpha), 1.0, 0) # generate CFAR for each range-doppler map in the cube

    return data_cube

def generateDopplerKernel(len, guard_len):
    '''Generate a a kernel for the doppler dimension with specified length and guard length'''
    len = 2*(len//2)+1 #  might make a kernel bigger than desired
    guard_len = 2*(guard_len//2)+1 # might make guard kernel bigger than desired
    unguarded_len = (len-guard_len)//2
    kernel = torch.ones(len, 1)
    kernel[unguarded_len:-unguarded_len] = 0

    return kernel

def generateRangeDopplerKernel(len, guard_len):
    '''Generate a a kernel for the range and doppler dimensions with specified length and guard length'''
    len = 2*(len//2)+1 #  might make a kernel bigger than desired
    guard_len = 2*(guard_len//2)+1 # might make guard kernel bigger than desired
    unguarded_len = (len-guard_len)//2
    kernel = torch.ones(len, len)
    kernel[:, unguarded_len: -unguarded_len] = 0

    return kernel



def microDoppler(data_cube, range_bin, n_fft=64, hop_length=None, win_length=16, range_window = None, doppler_window = None):
    '''Create micro doppler spectrogram from radar data cube'''
    data_cube = rangeFft(data_cube, range_window)
    print(data_cube.shape)
    doppler_tensor = torch.reshape(data_cube[:, 0, :, -range_bin], (-1, )) # 1D doppler string of specified range bin
    print(doppler_tensor.shape)
    spectrogram = torch.stft(doppler_tensor, n_fft, hop_length, win_length, doppler_window) # generate spectrogram
    return spectrogram