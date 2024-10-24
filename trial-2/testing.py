import readdata as read
import radarprocessing as radar
import processingmodels as processes

import matplotlib.pyplot as plt
import h5py

import torch

import os

import cnnmodels as models

import pandas as pd

def plotData():
    torch.set_default_device('cuda')

    input_csv = "csvs/smallset.csv"
    input_df = pd.read_csv(input_csv)

    proc_types = ("rd2d","rd3d", "cfar", "bf", "ang", "md")
    procs = (processes.rangeDoppler2dProcessFinal, processes.rangeDoppler3dProcessFinal, processes.cfarProcessFinal, 
             processes.beamformingProcessFinal, processes.angleProcessFinal, processes.microDopplerProcessFinal)
    
    root_dirs = ("data", "data-test-same", "data-test-new")
    for name in input_df["file_name"]:
        print(name)
        for root_dir in root_dirs:
            file_path = os.path.join(root_dir, name)
            if os.path.exists(file_path):
                file_hdf5 = h5py.File(file_path) # load the hdf5 file

        data_cube = read.radarDataTensor(file_hdf5)
        data_cube = data_cube.to('cuda')

        for proc_id in range(len(proc_types)):
            processed = procs[proc_id](data_cube, file_hdf5)[0]
            processed = processed.to('cpu')
            vmax = torch.amax(processed)
            vmin = torch.amin(processed)
            if len(processed.shape) > 2:
            
                num_plots = processed.shape[0]

                fig, axes = plt.subplots(1, num_plots)

                for i, ax in enumerate(axes):
                    ax.imshow(processed[i], interpolation=None, vmax=vmax, vmin=vmin)
                    if i>0:
                        ax.set_yticks([])
            else:
                fig, ax = plt.subplots()
                ax.imshow(processed, interpolation=None, vmax=vmax, vmin=vmin)

            plt.savefig("../sample-figs/" + proc_types[proc_id] + "/" + name[:-5] + ".png")
            plt.close()



def main():
    plotData()

if __name__ == "__main__":
    main()