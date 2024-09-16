import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import pipeline

recording_path = 'data/stationary_target_1m_1.hdf5'
recording_path_enc = os.fsencode(recording_path)

recording_hdf5 = h5py.File(os.fsdecode(recording_path_enc))

recording_radar = recording_hdf5['Sensors']['TI_Radar']
radar_data = recording_radar['Data']
radar_conf = recording_radar['Parameters']

frame_0 = radar_data['Frame_0']['frame_data']
frame_0_chan_0 = frame_0[::,::,0]

print(frame_0.shape)
range_fft = np.fft.fftshift(np.fft.fft(frame_0_chan_0, axis=0))
doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=1))
doppler_fft_dB = 20*np.log10(np.abs(doppler_fft))
range_fft_dB = 20*np.log10(np.abs(range_fft))

doppler_fft_dB_swapped = np.swapaxes(doppler_fft_dB, 0, 1)
range_fft_dB_swapped = np.swapaxes(range_fft_dB, 0, 1)

T_sweep  = radar_conf['profileCfg']['rampEndTime'][()]*1e-6
T_data = (radar_conf['profileCfg']['numAdcSamples'][()])/(radar_conf['profileCfg']['digOutSampleRate'][()]*1000)
print(T_data)
c = 3*10**8

freq_slope_Hz = (1/20.71)*radar_conf['profileCfg']['freqSlopeConst'][()]
B = freq_slope_Hz*T_data*1e8 # this doesn't seem to be right yet... TODO
print(B)

delta_R = pipeline.getRangeResolution(pipeline.radarConf(recording_hdf5))
max_R = delta_R * doppler_fft_dB_swapped.shape[0]

fig, ax = plt.subplots()

ax.imshow(np.fliplr(range_fft_dB_swapped), interpolation='none', extent = (0, max_R, 0, 203), aspect='auto')
# ax.set_xlim(0, 5)

plt.show()
