import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import h5py

import processingmodels as processes

import radarprocessing as radar

import readdata as read

import torch

import colorcet as cc

import seaborn as sn


# global parameters

# cmap = cc.cm.rainbow_bgyr_10_90_c83
cmap = 'viridis'
file_path = file_path = "/home/rtdug/UCT-EEE4022S-2024/trial-2/data/Experiment_2024-09-17_16-56-10_560_palm_grab.hdf5"
gesture_df = pd.read_csv("csvs/smallset.csv")


def main():
    lossGraphs()
    results()
    confusionMatrices()
    rangeDopplerPlot()
    snrIncrease()
    cropping()
    frameSelection()
    clutterRemoval()
    rangeDoppler2d()
    rangeDoppler3d()
    cfar()
    angleFft()
    beamforming()
    microDoppler()

def reportSaveFig(fig, name):
    # fig.savefig("../sample-figs/report/" + name + ".png", bbox_inches='tight')
    fig.savefig("../sample-figs/" + name + ".png", bbox_inches='tight')


def createTicks(num_ticks, lim, scale_max, scale_min, decimals):
    
    scale_max_rounded = scale_max
    scale_min_rounded = scale_min

    scale_max_rounded = np.around(scale_max_rounded, decimals=decimals)
    scale_min_rounded = np.around(scale_min_rounded, decimals=decimals)

    if scale_max_rounded > scale_max:
        scale_max_rounded -= np.pow(0.1, decimals)
    
    if scale_min_rounded < scale_min:
        scale_min_rounded += np.pow(0.1, decimals)
    
    ticklabels = np.linspace(scale_min_rounded, scale_max_rounded, num_ticks)
    ticks = (ticklabels - scale_min)/(scale_max - scale_min) * (lim[1] - lim[0]) + lim[0]
    
    return ticks, ticklabels

def lossGraphs():

    # results = ("rd2d_train_results_14", "rd3d_train_results_14", "cfar_train_results_14", 
    #              "ang_train_results_14", "bf_train_results_14", "md_train_results_14", 
    #             "rd2d_train_results_6", "rd3d_train_results_6", "cfar_train_results_6", 
    #              "ang_train_results_6", "bf_train_results_6", "md_train_results_6", )
    results = ("rd3d_test_same_results_14", "rd3d_test_new_results_14", "rd3d_all_prop_results_14", 
               "bf_test_same_results_14", "bf_test_new_results_14", "bf_all_prop_results_14",
                "rd3d_test_same_results_6", "rd3d_test_new_results_6", "rd3d_all_prop_results_6", 
                "bf_test_same_results_6", "bf_test_new_results_6", "bf_all_prop_results_6")
    for name in results:
        csv_file = "results/" + name + ".csv"

        results_df = pd.read_csv(csv_file, header=0)

        
        stats_df = pd.DataFrame()

        for output_type in ("train_loss", "val_loss", "val_acc"):
            mask = results_df.columns.str.contains(output_type + '_.')

            stats_df.loc[:, output_type + "_avg"] = results_df.loc[:, mask].mean(axis=1)
            stats_df.loc[:, output_type + "_min"] = results_df.loc[:, mask].min(axis=1)
            stats_df.loc[:, output_type + "_max"] = results_df.loc[:, mask].max(axis=1)
        
        fig, ax1 = plt.subplots()

        x = np.linspace(0, 200, stats_df.shape[0])

        ax2 = ax1.twinx()
        ax1.plot(x, stats_df.loc[:, "train_loss_avg"], label='train loss')
        ax1.plot(x, stats_df.loc[:, "val_loss_avg"], label='validation loss')
        ax2.plot(x, stats_df.loc[:, "val_acc_avg"]*100, label='validation accuracy', color='tab:green')
        
        ax2.set_ylim(-5, 105)

        ax2.set_ylabel('Accuracy')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Cross-Entropy Loss')

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='center right')

        fig.suptitle("Average loss and accuracy during training")

        reportSaveFig(fig, "loss_" + name)


def results():
    results = ("rd3d_test_same_results_14", "rd3d_test_new_results_14", "rd3d_all_prop_results_14", 
               "bf_test_same_results_14", "bf_test_new_results_14", "bf_all_prop_results_14",
                "rd3d_test_same_results_6", "rd3d_test_new_results_6", "rd3d_all_prop_results_6", 
                "bf_test_same_results_6", "bf_test_new_results_6", "bf_all_prop_results_6")
    nums_epochs = (50, 30, 50, 30, 40, 60, 
                   25, 50, 60, 25, 30, 50)
    for num, name in enumerate(results):
        csv_file = "results/" + name + ".csv"

        results_df = pd.read_csv(csv_file, header=0)

        stats_df = pd.DataFrame()

        for output_type in ("train_loss", "val_loss", "val_acc"):
            mask = results_df.columns.str.contains(output_type + '_.')

            stats_df.loc[:, output_type + "_avg"] = results_df.loc[:, mask].mean(axis=1)
            stats_df.loc[:, output_type + "_min"] = results_df.loc[:, mask].min(axis=1)
            stats_df.loc[:, output_type + "_max"] = results_df.loc[:, mask].max(axis=1)


            num_epochs = nums_epochs[num]
            loc = int((num_epochs/200)*stats_df.shape[0])

        # avg_formatted = "{:.1f}".format(np.around(stats_df.at[stats_df.shape[0]-1, "val_acc_avg"]*100, 1))
        avg_formatted = "{:.1f}".format(np.around(stats_df.at[loc, "val_acc_avg"]*100, 1))

        # min_formatted = "{:.1f}".format(np.around(stats_df.at[stats_df.shape[0]-1, "val_acc_min"]*100, 1))

        print(name + " & " + avg_formatted)


def confusionMatrices():
    
    def helper(confusion, confusion_df):
        confusion_df = confusion_df.div(confusion_df.sum(axis=1), axis=0)*100
        confusion_df = np.around(confusion_df, 1)
        confusion_df.columns = confusion_df.columns.str.replace('_', ' ')
        confusion_df.index = confusion_df.index.str.replace('_', ' ')
        if "14" in confusion:
            new_order = ("non gesture", "palm grab", "palm release", "swipe left","swipe right", "swipe up", "swipe down",
                          "virtual tap","virtual slider left","virtual slider right", 
                          "virtual knob clockwise", "virtual knob anticlockwise", 
                          "pinch out horizontal", "pinch out vertical")
        elif "6" in confusion:
            new_order = ("palm grab", "palm release", "swipe left","swipe right", "swipe up", "swipe down")
        
        confusion_df = confusion_df.loc[new_order, new_order]
        fig, ax = plt.subplots()
        sn.heatmap(confusion_df.T, ax=ax, annot=True, cbar=False, vmin=0, vmax=100, cmap=cc.cm.blues)
        ax.set_xlabel("Actual gesture")
        ax.set_ylabel("Predicted gesture (%)")
        ax.set_title("Confusion Matrix")
        # fig.suptitle("Confusion Matrix")

        
        
        reportSaveFig(fig, confusion)
        plt.close(fig)

        


    confusions = ("rd2d_train_confusion_14", "rd3d_train_confusion_14", "cfar_train_confusion_14", 
                "ang_train_confusion_14", "bf_train_confusion_14", "md_train_confusion_14", 
            "rd2d_train_confusion_6", "rd3d_train_confusion_6", "cfar_train_confusion_6", 
                "ang_train_confusion_6", "bf_train_confusion_6", "md_train_confusion_6",)
    
    for confusion in confusions:
        confusion_df = pd.read_csv("results_validation/" + confusion + ".csv", header=0, index_col=0)
        helper(confusion, confusion_df)

    confusions = ("rd2d_test_same_confusion_14", "rd3d_test_same_confusion_14", "cfar_test_same_confusion_14", 
                "ang_test_same_confusion_14", "bf_test_same_confusion_14", "md_test_same_confusion_14", 
            "rd2d_test_same_confusion_6", "rd3d_test_same_confusion_6", "cfar_test_same_confusion_6", 
                "ang_test_same_confusion_6", "bf_test_same_confusion_6", "md_test_same_confusion_6",
                "rd2d_test_new_confusion_14", "rd3d_test_new_confusion_14", "cfar_test_new_confusion_14", 
                "ang_test_new_confusion_14", "bf_test_new_confusion_14", "md_test_new_confusion_14", 
            "rd2d_test_new_confusion_6", "rd3d_test_new_confusion_6", "cfar_test_new_confusion_6", 
                "ang_test_new_confusion_6", "bf_test_new_confusion_6", "md_test_new_confusion_6",)
    
    for confusion in confusions:
        confusion_df = pd.read_csv("results_test/" + confusion + ".csv", header=0, index_col=0)
        helper(confusion, confusion_df)

                


def rangeDopplerPlot():

    file_hdf5 = h5py.File(file_path)
    range_res = read.rangeResolution(file_hdf5)
    velocity_res = read.velocityResolution(file_hdf5)

    data_cube = read.radarDataTensor(file_hdf5)
    data_cube = radar.rangeDoppler(data_cube, torch.hann_window)
    
    to_plot = radar.todB(torch.abs(data_cube[13, 0, :, :]))

    range_max = data_cube.shape[3]*range_res
    range_min = 0
    velocity_max = (data_cube.shape[2]/2)*velocity_res
    velocity_min = -velocity_max

    fig, ax = plt.subplots()

    ax.imshow(to_plot, interpolation='none', cmap=cmap)
    ax.set_xlabel("Range [m]")
    xticks, xticklabels = createTicks(5, ax.get_xlim(), range_max, range_min, 0)
    ax.set_xticks(xticks, labels=xticklabels)

    ax.set_ylabel("Velocity [m/s]")
    yticks, yticklabels = createTicks(9, ax.get_ylim(), velocity_max, velocity_min, 0)
    ax.set_yticks(yticks, labels=yticklabels)
    
    plt.title("Range Doppler Map")

    reportSaveFig(fig, "range_doppler_example")
    plt.close(fig)

def snrIncrease():
    
    file_hdf5 = h5py.File(file_path)
    range_res = read.rangeResolution(file_hdf5)
    velocity_res = read.velocityResolution(file_hdf5)

    data_cube = read.radarDataTensor(file_hdf5)
    data_cube = radar.rangeDoppler(data_cube, torch.hann_window)
    data_cube = torch.sum(data_cube, dim=(1,), keepdim=True)

    range_max = data_cube.shape[3]*range_res
    range_min = 0
    velocity_max = (data_cube.shape[2]/2)*velocity_res
    velocity_min = -velocity_max

    to_plot = radar.todB(torch.abs(data_cube[13, 0, :, :]))

    
    fig, ax = plt.subplots()

    ax.imshow(to_plot, interpolation='none', cmap=cmap)
    
    ax.set_xlabel("Range [m]")
    xticks, xticklabels = createTicks(5, ax.get_xlim(), range_max, range_min, 0)
    ax.set_xticks(xticks, labels=xticklabels)

    ax.set_ylabel("Velocity [m/s]")
    yticks, yticklabels = createTicks(9, ax.get_ylim(), velocity_max, velocity_min, 0)
    ax.set_yticks(yticks, labels=yticklabels)

    plt.title("Range Doppler Map, channel sum")

    reportSaveFig(fig, "range_doppler_snr_increase")
    plt.close(fig)

def cropping():
    for row in gesture_df.iterrows():
        file_hdf5 = h5py.File("data/" + row[1]["file_name"])

        range_res = read.rangeResolution(file_hdf5)
        velocity_res = read.velocityResolution(file_hdf5)

        data_cube = read.radarDataTensor(file_hdf5)
        data_cube = radar.rangeDoppler(data_cube, torch.hann_window)
        data_cube = torch.sum(data_cube, dim=(1,), keepdim=True)

        range_low = 7
        range_high = 13
        
        velocity_low = 24
        velocity_high = 74
        
        best_frame = radar.bestFrame(data_cube, range_low, range_high)

        data_cube = data_cube[:, :, velocity_low:velocity_high, range_low:range_high]
        
        to_plot = radar.todB(torch.abs(data_cube[best_frame, 0, :, :]))

        range_min = range_low*range_res
        range_max = range_high*range_res

        velocity_max = to_plot.shape[0]/2*velocity_res
        velocity_min = -velocity_max

        fig, ax = plt.subplots()

        ax.imshow(to_plot, interpolation=None, cmap=cmap)

        ax.set_xlabel("Range [m]")
        xticks, xticklabels = createTicks(2, ax.get_xlim(), range_max, range_min, 2)
        ax.set_xticks(xticks, labels=xticklabels)

        ax.set_ylabel("Velocity [m/s]")
        yticks, yticklabels = createTicks(5, ax.get_ylim(), velocity_max, velocity_min, 0)
        ax.set_yticks(yticks, labels=yticklabels)


        plt.title("Range Doppler Map, cropped")


        reportSaveFig(fig, "range_doppler_cropped_" + row[1]["label"])
        plt.close(fig)

def frameSelection():
    for row in gesture_df.iterrows():
        file_hdf5 = h5py.File("data/" + row[1]["file_name"])

        range_res = read.rangeResolution(file_hdf5)
        velocity_res = read.velocityResolution(file_hdf5)

        data_cube = read.radarDataTensor(file_hdf5)
        data_cube = radar.rangeDoppler(data_cube, torch.hann_window)
        data_cube = torch.sum(data_cube, dim=(1,), keepdim=True)

        range_low = 7
        range_high = 13
        
        velocity_low = 24
        velocity_high = 74
        
        best_frame = radar.bestFrame(data_cube, range_low, range_high)
        start_frame = best_frame - 10
        if start_frame < 0: start_frame = 0
        if start_frame+20 > data_cube.shape[0]: start_frame = data_cube.shape[0] - 20
        data_cube = data_cube[start_frame:start_frame+20, ...]

        data_cube = data_cube[:, :, velocity_low:velocity_high, range_low:range_high]
        
        to_plot = radar.todB(torch.abs(data_cube[:, 0, :, :]))

        range_min = range_low*range_res
        range_max = range_high*range_res

        velocity_max = to_plot.shape[-2]/2*velocity_res
        velocity_min = -velocity_max

        vmax = torch.amax(to_plot)
        vmin = torch.amin(to_plot)

        fig, axes = plt.subplots(1, to_plot.shape[0], constrained_layout=True)
        for i, ax in enumerate(axes):
            ax.imshow(to_plot[i], interpolation=None, cmap=cmap, vmax=vmax, vmin=vmin)
            ax.set_xlabel(int(start_frame + i+1))

            yticks, yticklabels = createTicks(5, ax.get_ylim(), velocity_max, velocity_min, 0)
            ax.set_yticks(yticks, labels=yticklabels)
            ax.label_outer(remove_inner_ticks=True)
            ax.set_xticks([])

        fig.suptitle("Range Doppler Map, selected frames", y=0.75)
        fig.supxlabel("Frame number", y=0.22)
        fig.supylabel("Velocity [m/s]")

        reportSaveFig(fig, "range_doppler_selected_frames_" + row[1]["label"])
        plt.close(fig)

def clutterRemoval():
    
    file_hdf5 = h5py.File(file_path)
    range_res = read.rangeResolution(file_hdf5)
    velocity_res = read.velocityResolution(file_hdf5)

    data_cube = read.radarDataTensor(file_hdf5)
    data_cube = radar.rangeDoppler(data_cube, torch.hann_window)
    data_cube = torch.sum(data_cube, dim=(1,), keepdim=True)

    range_max = data_cube.shape[3]*range_res
    range_min = 0
    velocity_max = (data_cube.shape[2]/2)*velocity_res
    velocity_min = -velocity_max

    to_plot = radar.todB(torch.abs(data_cube[13, 0, :, :] - data_cube[3, 0, :, :]))

    fig, ax = plt.subplots()

    ax.imshow(to_plot, interpolation='none', cmap=cmap)
    
    ax.set_xlabel("Range [m]")
    xticks, xticklabels = createTicks(5, ax.get_xlim(), range_max, range_min, 0)
    ax.set_xticks(xticks, labels=xticklabels)

    ax.set_ylabel("Velocity [m/s]")
    yticks, yticklabels = createTicks(9, ax.get_ylim(), velocity_max, velocity_min, 0)
    ax.set_yticks(yticks, labels=yticklabels)

    plt.title("Range Doppler Map, clutter removed")

    reportSaveFig(fig, "range_doppler_clutter_removal")
    plt.close(fig)

def rangeDoppler2d():
    for row in gesture_df.iterrows():
        file_hdf5 = h5py.File("data/" + row[1]["file_name"])

        range_res = read.rangeResolution(file_hdf5)
        velocity_res = read.velocityResolution(file_hdf5)

        data_cube = read.radarDataTensor(file_hdf5)

        range_low = 7
        range_high = 13

        to_plot = processes.rangeDoppler2dProcessFinal(data_cube, file_hdf5)[0]

        range_min = range_low*range_res
        range_max = range_high*range_res

        velocity_max = to_plot.shape[0]/2*velocity_res
        velocity_min = -velocity_max

        fig, ax = plt.subplots()

        ax.imshow(to_plot, interpolation=None, cmap=cmap)

        ax.set_xlabel("Range [m]")
        xticks, xticklabels = createTicks(2, ax.get_xlim(), range_max, range_min, 2)
        ax.set_xticks(xticks, labels=xticklabels)

        ax.set_ylabel("Velocity [m/s]")
        yticks, yticklabels = createTicks(5, ax.get_ylim(), velocity_max, velocity_min, 0)
        ax.set_yticks(yticks, labels=yticklabels)


        plt.title("Range Doppler Map, 2D frame sum")


        reportSaveFig(fig, "range_doppler_2d_" + row[1]["label"])
        plt.close(fig)

def rangeDoppler3d():
    for row in gesture_df.iterrows():
        file_hdf5 = h5py.File("data/" + row[1]["file_name"])

        range_res = read.rangeResolution(file_hdf5)
        velocity_res = read.velocityResolution(file_hdf5)

        data_cube = read.radarDataTensor(file_hdf5)

        range_low = 7
        range_high = 13

        to_plot, start_frame = processes.rangeDoppler3dProcessFinal(data_cube, file_hdf5, return_start_frame=True)
        to_plot = to_plot[0]

        range_min = range_low*range_res
        range_max = range_high*range_res

        velocity_max = to_plot.shape[-2]/2*velocity_res
        velocity_min = -velocity_max

        vmax = torch.amax(to_plot)
        vmin = torch.amin(to_plot)

        fig, axes = plt.subplots(1, to_plot.shape[0], constrained_layout=True)
        for i, ax in enumerate(axes):
            ax.imshow(to_plot[i], interpolation=None, cmap=cmap, vmax=vmax, vmin=vmin)
            ax.set_xlabel(int(start_frame + i+1))
            yticks, yticklabels = createTicks(5, ax.get_ylim(), velocity_max, velocity_min, 0)
            ax.set_yticks(yticks, labels=yticklabels)
            ax.label_outer(remove_inner_ticks=True)
            ax.set_xticks([])

        fig.suptitle("Range Doppler Map + Time", y=0.75)
        fig.supxlabel("Frame number", y=0.22)
        fig.supylabel("Velocity [m/s]")

        reportSaveFig(fig, "range_doppler_3d_" + row[1]["label"])
        plt.close(fig)

def cfar():
    for row in gesture_df.iterrows():
        file_hdf5 = h5py.File("data/" + row[1]["file_name"])

        range_res = read.rangeResolution(file_hdf5)
        velocity_res = read.velocityResolution(file_hdf5)

        data_cube = read.radarDataTensor(file_hdf5)

        range_low = 7
        range_high = 13

        to_plot, start_frame = processes.cfarProcessFinal(data_cube, file_hdf5, return_start_frame=True)
        to_plot = to_plot[0]

        range_min = range_low*range_res
        range_max = range_high*range_res

        velocity_max = to_plot.shape[-2]/2*velocity_res
        velocity_min = -velocity_max

        vmax = torch.amax(to_plot)
        vmin = torch.amin(to_plot)

        fig, axes = plt.subplots(1, to_plot.shape[0], constrained_layout=True)
        for i, ax in enumerate(axes):
            ax.imshow(to_plot[i], interpolation=None, cmap='gray', vmax=vmax, vmin=vmin)
            ax.set_xlabel(int(start_frame + i+1))
            yticks, yticklabels = createTicks(5, ax.get_ylim(), velocity_max, velocity_min, 0)
            ax.set_yticks(yticks, labels=yticklabels)
            ax.label_outer(remove_inner_ticks=True)
            ax.set_xticks([])

        fig.suptitle("Range Doppler Map + Time, with CFAR detection", y=0.75)
        fig.supxlabel("Frame number", y=0.22)
        fig.supylabel("Velocity [m/s]")

        reportSaveFig(fig, "cfar_" + row[1]["label"])
        plt.close(fig)

def angleFft():
    for row in gesture_df.iterrows():
        file_hdf5 = h5py.File("data/" + row[1]["file_name"])

        range_res = read.rangeResolution(file_hdf5)
        velocity_res = read.velocityResolution(file_hdf5)

        data_cube = read.radarDataTensor(file_hdf5)

        range_low = 7
        range_high = 13

        to_plot = processes.angleProcessFinal(data_cube, file_hdf5)
        to_plot = to_plot[0]

        range_min = range_low*range_res
        range_max = range_high*range_res

        velocity_max = to_plot.shape[-2]/2*velocity_res
        velocity_min = -velocity_max

        vmax = torch.amax(to_plot)
        vmin = torch.amin(to_plot)

        fig, axes = plt.subplots(1, to_plot.shape[0], constrained_layout=True)
        angles = np.linspace(11.25, 168.75, 8)

        for i, ax in enumerate(axes):
            ax.imshow(to_plot[i], interpolation=None, cmap=cmap, vmax=vmax, vmin=vmin)
            ax.set_xlabel(angles[i])
            yticks, yticklabels = createTicks(5, ax.get_ylim(), velocity_max, velocity_min, 0)
            ax.set_yticks(yticks, labels=yticklabels)
            ax.label_outer(remove_inner_ticks=True)
            ax.set_xticks([])

        fig.suptitle("Range Doppler Map + Angle FFT")
        fig.supxlabel("Centre angle [degrees]")
        fig.supylabel("Velocity [m/s]")

        reportSaveFig(fig, "angle_" + row[1]["label"])
        plt.close(fig)

def beamforming():
    for row in gesture_df.iterrows():
        file_hdf5 = h5py.File("data/" + row[1]["file_name"])

        range_res = read.rangeResolution(file_hdf5)
        velocity_res = read.velocityResolution(file_hdf5)

        data_cube = read.radarDataTensor(file_hdf5)

        range_low = 7
        range_high = 13

        to_plot = processes.beamformingProcessFinal(data_cube, file_hdf5)
        to_plot = to_plot[0]

        range_min = range_low*range_res
        range_max = range_high*range_res

        velocity_max = to_plot.shape[-2]/2*velocity_res
        velocity_min = -velocity_max

        vmax = torch.amax(to_plot)
        vmin = torch.amin(to_plot)

        fig, axes = plt.subplots(1, to_plot.shape[0], constrained_layout=True)
        angles = np.linspace(-15, 15, 16)

        for i, ax in enumerate(axes):
            ax.imshow(to_plot[i], interpolation=None, cmap=cmap, vmax=vmax, vmin=vmin)
            ax.set_xlabel(angles[i])
            yticks, yticklabels = createTicks(5, ax.get_ylim(), velocity_max, velocity_min, 0)
            ax.set_yticks(yticks, labels=yticklabels)
            ax.label_outer(remove_inner_ticks=True)
            ax.set_xticks([])

        fig.suptitle("Range Doppler Map + Beam Steering", y=0.8)
        fig.supxlabel("Centre angle [degrees]", y=0.16)
        fig.supylabel("Velocity [m/s]")

        reportSaveFig(fig, "beamforming_" + row[1]["label"])
        plt.close(fig)

def microDoppler():
    for row in gesture_df.iterrows():
        file_hdf5 = h5py.File("data/" + row[1]["file_name"])

        range_res = read.rangeResolution(file_hdf5)
        velocity_res = read.velocityResolution(file_hdf5)

        data_cube = read.radarDataTensor(file_hdf5)

        range_low = 7
        range_high = 13

        to_plot, start_frame = processes.microDopplerProcessFinal(data_cube, file_hdf5, return_start_frame=True)
        to_plot = to_plot[0]

        range_min = range_low*range_res
        range_max = range_high*range_res

        velocity_max = to_plot.shape[0]/2*velocity_res
        velocity_min = -velocity_max

        sample_rate = file_hdf5["Sensors/TI_Radar/Parameters/profileCfg/digOutSampleRate"][()]

        freq_max = sample_rate/2000

        fig, ax = plt.subplots()

        ax.imshow(to_plot, interpolation=None, cmap=cmap)

        ax.set_xlabel("Frame number")
        xticks, xticklabels = createTicks(19, ax.get_xlim(), 19.5+start_frame, 0.5+start_frame, 0)
        xticklabels += start_frame
        xticklabels = np.int32(xticklabels)

        ax.set_xticks(xticks, labels=xticklabels)

        ax.set_ylabel("Frequency [MHz]")
        freq_max *= 2
        yticks, yticklabels = createTicks(5, ax.get_ylim(), freq_max, -freq_max, 0)
        yticklabels /= 2
        ax.set_yticks(yticks, labels=yticklabels)


        plt.title("Micro-Doppler Spectrogram")


        reportSaveFig(fig, "spectrogram_" + row[1]["label"])
        plt.close(fig)


if __name__ == "__main__":
    main()