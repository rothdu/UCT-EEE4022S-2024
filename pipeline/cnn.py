# torch imports
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, Subset, DataLoader, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence

# torchvision imports
from torchvision.transforms import v2

# sklearn imports
from sklearn.model_selection import KFold

# other processing imports
import h5py
import pandas as pd
import matplotlib.pyplot as plt

# general python imports
import os
import time

# my other files
import readdata as read

import radarprocessing as radar

import cnnmodels as cnns

import processingmodels as processes

from dataset import GestureDataset

import random

def main():
    torch.set_default_device('cuda')

    # gestures14()
    # gestures6()
    # fake3dTest()
    trainOnTest()

def trainOnTest():
    root_dirs = ("data", "data-test-same", "data-test-new")
    
    num_splits = 5
    portion_val = 0.2
    # num_epochs = 250
    learning_rate = 0.0002
    num_channels = 1
    transform = None
    test = False
    
    num_epochs = 500
    in_csvs = ("csvs/test_same6.csv",)
    
    out_csvs = ("results/rd3d_test_same_results_6.csv",)
    confusion_csvs = ("results/rd3d_test_same_confusion_6.csv",)
    process = processes.rangeDoppler3dProcessFinal
    model = cnns.RangeDoppler3dModelFinal
    processInputs(process, in_csvs, root_dirs)
    runCnn(model, num_channels, transform, in_csvs, root_dirs, out_csvs, confusion_csvs, num_splits, portion_val, num_epochs, learning_rate, test)

    out_csvs = ("results/bf_test_same_results_6.csv", )
    confusion_csvs = ("results/bf_test_same_confusion_6.csv", )
    process = processes.beamformingProcessFinal
    model = cnns.BeamformingModelFinal
    processInputs(process, in_csvs, root_dirs)
    runCnn(model, num_channels, transform, in_csvs, root_dirs, out_csvs, confusion_csvs, num_splits, portion_val, num_epochs, learning_rate, test)

    in_csvs = ("csvs/test_new6.csv",)

    out_csvs = ("results/rd3d_test_new_results_6.csv",)
    confusion_csvs = ("results/rd3d_test_new_confusion_6.csv",)
    process = processes.rangeDoppler3dProcessFinal
    model = cnns.RangeDoppler3dModelFinal
    processInputs(process, in_csvs, root_dirs)
    runCnn(model, num_channels, transform, in_csvs, root_dirs, out_csvs, confusion_csvs, num_splits, portion_val, num_epochs, learning_rate, test)

    out_csvs = ("results/bf_test_new_results_6.csv", )
    confusion_csvs = ("results/bf_test_new_confusion_6.csv", )
    process = processes.beamformingProcessFinal
    model = cnns.BeamformingModelFinal
    processInputs(process, in_csvs, root_dirs)
    runCnn(model, num_channels, transform, in_csvs, root_dirs, out_csvs, confusion_csvs, num_splits, portion_val, num_epochs, learning_rate, test)

    num_epochs = 250
    in_csvs = ("csvs/all_prop6.csv",)

    out_csvs = ("results/rd3d_all_prop_results_6.csv",)
    confusion_csvs = ("results/rd3d_all_prop_confusion_6.csv",)
    process = processes.rangeDoppler3dProcessFinal
    model = cnns.RangeDoppler3dModelFinal
    processInputs(process, in_csvs, root_dirs)
    runCnn(model, num_channels, transform, in_csvs, root_dirs, out_csvs, confusion_csvs, num_splits, portion_val, num_epochs, learning_rate, test)

    out_csvs = ("results/bf_all_prop_results_6.csv", )
    confusion_csvs = ("results/bf_all_prop_confusion_6.csv", )
    process = processes.beamformingProcessFinal
    model = cnns.BeamformingModelFinal
    processInputs(process, in_csvs, root_dirs)
    runCnn(model, num_channels, transform, in_csvs, root_dirs, out_csvs, confusion_csvs, num_splits, portion_val, num_epochs, learning_rate, test)

    num_epochs = 500
    in_csvs = ("csvs/test_same14.csv",)    
    
    out_csvs = ("results/rd3d_test_same_results_14.csv",)
    confusion_csvs = ("results/rd3d_test_same_confusion_14.csv",)
    process = processes.rangeDoppler3dProcessFinal
    model = cnns.RangeDoppler3dModelFinal
    processInputs(process, in_csvs, root_dirs)
    runCnn(model, num_channels, transform, in_csvs, root_dirs, out_csvs, confusion_csvs, num_splits, portion_val, num_epochs, learning_rate, test)

    out_csvs = ("results/bf_test_same_results_14.csv", )
    confusion_csvs = ("results/bf_test_same_confusion_14.csv", )
    process = processes.beamformingProcessFinal
    model = cnns.BeamformingModelFinal
    processInputs(process, in_csvs, root_dirs)
    runCnn(model, num_channels, transform, in_csvs, root_dirs, out_csvs, confusion_csvs, num_splits, portion_val, num_epochs, learning_rate, test)

    in_csvs = ("csvs/test_new14.csv",)

    out_csvs = ("results/rd3d_test_new_results_14.csv",)
    confusion_csvs = ("results/rd3d_test_new_confusion_14.csv",)
    process = processes.rangeDoppler3dProcessFinal
    model = cnns.RangeDoppler3dModelFinal
    processInputs(process, in_csvs, root_dirs)
    runCnn(model, num_channels, transform, in_csvs, root_dirs, out_csvs, confusion_csvs, num_splits, portion_val, num_epochs, learning_rate, test)

    out_csvs = ("results/bf_test_new_results_14.csv", )
    confusion_csvs = ("results/bf_test_new_confusion_14.csv", )
    process = processes.beamformingProcessFinal
    model = cnns.BeamformingModelFinal
    processInputs(process, in_csvs, root_dirs)
    runCnn(model, num_channels, transform, in_csvs, root_dirs, out_csvs, confusion_csvs, num_splits, portion_val, num_epochs, learning_rate, test)

    num_epochs = 250
    in_csvs = ("csvs/all_prop14.csv",)

    out_csvs = ("results/rd3d_all_prop_results_14.csv",)
    confusion_csvs = ("results/rd3d_all_prop_confusion_14.csv",)
    process = processes.rangeDoppler3dProcessFinal
    model = cnns.RangeDoppler3dModelFinal
    processInputs(process, in_csvs, root_dirs)
    runCnn(model, num_channels, transform, in_csvs, root_dirs, out_csvs, confusion_csvs, num_splits, portion_val, num_epochs, learning_rate, test)

    out_csvs = ("results/bf_all_prop_results_14.csv", )
    confusion_csvs = ("results/bf_all_prop_confusion_14.csv", )
    process = processes.beamformingProcessFinal
    model = cnns.BeamformingModelFinal
    processInputs(process, in_csvs, root_dirs)
    runCnn(model, num_channels, transform, in_csvs, root_dirs, out_csvs, confusion_csvs, num_splits, portion_val, num_epochs, learning_rate, test)

def fake3dTest():
    in_csvs = ("csvs/train14.csv", "csvs/test_same14.csv", "csvs/test_new14.csv")
    root_dirs = ("data", "data-test-same", "data-test-new")
    
    num_splits = 5
    portion_val = 0.2
    num_epochs = 100
    learning_rate = 0.0002
    num_channels = 1
    transform = None
    test = False
    
    out_csvs = ("results/rd3dfake_train_results_14.csv", "results/rd3dfake_test_same_results_14.csv", "results/rd3dfake_test_new_results_14.csv")
    confusion_csvs = ("results/rd3dfake_train_confusion_14.csv", "results/rd3dfake_test_same_confusion_14.csv", "results/rd3dfake_test_new_confusion_14.csv")
    process = processes.rangeDoppler3dFakeProcessFinal
    model = cnns.BeamformingModelFinal
    processInputs(process, in_csvs, root_dirs)
    runCnn(model, num_channels, transform, in_csvs, root_dirs, out_csvs, confusion_csvs, num_splits, portion_val, num_epochs, learning_rate, test)

    in_csvs = ("csvs/train6.csv", "csvs/test_same6.csv", "csvs/test_new6.csv")
    root_dirs = ("data", "data-test-same", "data-test-new")

    num_splits = 5
    portion_val = 0.2
    num_epochs = 100
    learning_rate = 0.0002
    num_channels = 1
    transform = None
    test = False

    out_csvs = ("results/rd3dfake_train_results_6.csv", "results/rd3dfake_test_same_results_6.csv", "results/rd3dfake_test_new_results_6.csv")
    confusion_csvs = ("results/rd3dfake_train_confusion_6.csv", "results/rd3dfake_test_same_confusion_6.csv", "results/rd3dfake_test_new_confusion_6.csv")
    process = processes.rangeDoppler3dFakeProcessFinal
    model = cnns.BeamformingModelFinal
    processInputs(process, in_csvs, root_dirs)
    runCnn(model, num_channels, transform, in_csvs, root_dirs, out_csvs, confusion_csvs, num_splits, portion_val, num_epochs, learning_rate, test)

def gestures14():
    in_csvs = ("csvs/train14.csv", "csvs/test_same14.csv", "csvs/test_new14.csv")
    root_dirs = ("data", "data-test-same", "data-test-new")
    
    num_splits = 5
    portion_val = 0.2
    # num_epochs = 200
    learning_rate = 0.0002
    num_channels = 1
    transform = None
    test = False

    num_epochs = 90
    out_csvs = ("results/rd2d_train_results_14.csv", "results/rd2d_test_same_results_14.csv", "results/rd2d_test_new_results_14.csv")
    confusion_csvs = ("results/rd2d_train_confusion_14.csv", "results/rd2d_test_same_confusion_14.csv", "results/rd2d_test_new_confusion_14.csv")
    process = processes.rangeDoppler2dProcessFinal
    model = cnns.RangeDoppler2dModelFinal
    processInputs(process, in_csvs, root_dirs)
    runCnn(model, num_channels, transform, in_csvs, root_dirs, out_csvs, confusion_csvs, num_splits, portion_val, num_epochs, learning_rate, test)

    num_epochs = 70
    out_csvs = ("results/rd3d_train_results_14.csv", "results/rd3d_test_same_results_14.csv", "results/rd3d_test_new_results_14.csv")
    confusion_csvs = ("results/rd3d_train_confusion_14.csv", "results/rd3d_test_same_confusion_14.csv", "results/rd3d_test_new_confusion_14.csv")
    process = processes.rangeDoppler3dProcessFinal
    model = cnns.RangeDoppler3dModelFinal
    processInputs(process, in_csvs, root_dirs)
    runCnn(model, num_channels, transform, in_csvs, root_dirs, out_csvs, confusion_csvs, num_splits, portion_val, num_epochs, learning_rate, test)

    num_epochs = 30
    out_csvs = ("results/cfar_train_results_14.csv", "results/cfar_test_same_results_14.csv", "results/cfar_test_new_results_14.csv")
    confusion_csvs = ("results/cfar_train_confusion_14.csv", "results/cfar_test_same_confusion_14.csv", "results/cfar_test_new_confusion_14.csv")
    process = processes.cfarProcessFinal
    model = cnns.Cfar3dModelFinal
    processInputs(process, in_csvs, root_dirs)
    runCnn(model, num_channels, transform, in_csvs, root_dirs, out_csvs, confusion_csvs, num_splits, portion_val, num_epochs, learning_rate, test)

    num_epochs = 70
    out_csvs = ("results/ang_train_results_14.csv", "results/ang_test_same_results_14.csv", "results/ang_test_new_results_14.csv")
    confusion_csvs = ("results/ang_train_confusion_14.csv", "results/ang_test_same_confusion_14.csv", "results/ang_test_new_confusion_14.csv")
    process = processes.angleProcessFinal
    model = cnns.AngleModelFinal
    processInputs(process, in_csvs, root_dirs)
    runCnn(model, num_channels, transform, in_csvs, root_dirs, out_csvs, confusion_csvs, num_splits, portion_val, num_epochs, learning_rate, test)

    num_epochs = 50
    out_csvs = ("results/bf_train_results_14.csv", "results/bf_test_same_results_14.csv", "results/bf_test_new_results_14.csv")
    confusion_csvs = ("results/bf_train_confusion_14.csv", "results/bf_test_same_confusion_14.csv", "results/bf_test_new_confusion_14.csv")
    process = processes.beamformingProcessFinal
    model = cnns.BeamformingModelFinal
    processInputs(process, in_csvs, root_dirs)
    runCnn(model, num_channels, transform, in_csvs, root_dirs, out_csvs, confusion_csvs, num_splits, portion_val, num_epochs, learning_rate, test)

    num_epochs = 30
    out_csvs = ("results/md_train_results_14.csv", "results/md_test_same_results_14.csv", "results/md_test_new_results_14.csv")
    confusion_csvs = ("results/md_train_confusion_14.csv", "results/md_test_same_confusion_14.csv", "results/md_test_new_confusion_14.csv")
    process = processes.microDopplerProcessFinal
    model = cnns.MicroDopplerModelFinal
    processInputs(process, in_csvs, root_dirs)
    runCnn(model, num_channels, transform, in_csvs, root_dirs, out_csvs, confusion_csvs, num_splits, portion_val, num_epochs, learning_rate, test)


def gestures6():
    in_csvs = ("csvs/train6.csv", "csvs/test_same6.csv", "csvs/test_new6.csv")
    root_dirs = ("data", "data-test-same", "data-test-new")

    num_splits = 5
    portion_val = 0.2
    # num_epochs = 200
    learning_rate = 0.0002
    num_channels = 1
    transform = None
    test = False

    num_epochs = 60
    out_csvs = ("results/rd2d_train_results_6.csv", "results/rd2d_test_same_results_6.csv", "results/rd2d_test_new_results_6.csv")
    confusion_csvs = ("results/rd2d_train_confusion_6.csv", "results/rd2d_test_same_confusion_6.csv", "results/rd2d_test_new_confusion_6.csv")
    process = processes.rangeDoppler2dProcessFinal
    model = cnns.RangeDoppler2dModelFinal
    processInputs(process, in_csvs, root_dirs)
    runCnn(model, num_channels, transform, in_csvs, root_dirs, out_csvs, confusion_csvs, num_splits, portion_val, num_epochs, learning_rate, test)

    num_epochs = 70
    out_csvs = ("results/rd3d_train_results_6.csv", "results/rd3d_test_same_results_6.csv", "results/rd3d_test_new_results_6.csv")
    confusion_csvs = ("results/rd3d_train_confusion_6.csv", "results/rd3d_test_same_confusion_6.csv", "results/rd3d_test_new_confusion_6.csv")
    process = processes.rangeDoppler3dProcessFinal
    model = cnns.RangeDoppler3dModelFinal
    processInputs(process, in_csvs, root_dirs)
    runCnn(model, num_channels, transform, in_csvs, root_dirs, out_csvs, confusion_csvs, num_splits, portion_val, num_epochs, learning_rate, test)

    num_epochs = 40
    out_csvs = ("results/cfar_train_results_6.csv", "results/cfar_test_same_results_6.csv", "results/cfar_test_new_results_6.csv")
    confusion_csvs = ("results/cfar_train_confusion_6.csv", "results/cfar_test_same_confusion_6.csv", "results/cfar_test_new_confusion_6.csv")
    process = processes.cfarProcessFinal
    model = cnns.Cfar3dModelFinal
    processInputs(process, in_csvs, root_dirs)
    runCnn(model, num_channels, transform, in_csvs, root_dirs, out_csvs, confusion_csvs, num_splits, portion_val, num_epochs, learning_rate, test)

    num_epochs = 80
    out_csvs = ("results/ang_train_results_6.csv", "results/ang_test_same_results_6.csv", "results/ang_test_new_results_6.csv")
    confusion_csvs = ("results/ang_train_confusion_6.csv", "results/ang_test_same_confusion_6.csv", "results/ang_test_new_confusion_6.csv")
    process = processes.angleProcessFinal
    model = cnns.AngleModelFinal
    processInputs(process, in_csvs, root_dirs)
    runCnn(model, num_channels, transform, in_csvs, root_dirs, out_csvs, confusion_csvs, num_splits, portion_val, num_epochs, learning_rate, test)

    num_epochs = 80
    out_csvs = ("results/bf_train_results_6.csv", "results/bf_test_same_results_6.csv", "results/bf_test_new_results_6.csv")
    confusion_csvs = ("results/bf_train_confusion_6.csv", "results/bf_test_same_confusion_6.csv", "results/bf_test_new_confusion_6.csv")
    process = processes.beamformingProcessFinal
    model = cnns.BeamformingModelFinal
    processInputs(process, in_csvs, root_dirs)
    runCnn(model, num_channels, transform, in_csvs, root_dirs, out_csvs, confusion_csvs, num_splits, portion_val, num_epochs, learning_rate, test)

    num_epochs = 40
    out_csvs = ("results/md_train_results_6.csv", "results/md_test_same_results_6.csv", "results/md_test_new_results_6.csv")
    confusion_csvs = ("results/md_train_confusion_6.csv", "results/md_test_same_confusion_6.csv", "results/md_test_new_confusion_6.csv")
    process = processes.microDopplerProcessFinal
    model = cnns.MicroDopplerModelFinal
    processInputs(process, in_csvs, root_dirs)
    runCnn(model, num_channels, transform, in_csvs, root_dirs, out_csvs, confusion_csvs, num_splits, portion_val, num_epochs, learning_rate, test)


def processInputs(radar_process, in_csvs, root_dirs, plot=False):
    
    processed_path = os.fsencode("processed_data/gestures.hdf5")

    if os.path.exists(processed_path):
        os.remove(processed_path)

    processed_hdf5 = h5py.File(processed_path, "w")
    
    for input_csv in in_csvs:
        gesture_df = pd.read_csv(input_csv)

        print_index = 50
        for index, row in gesture_df.iterrows():
            if index%print_index == 0:
                print(f"Pre-processing item: {index}", end = "")
                start_time = time.time()

            file_name = row["file_name"]

            for root_dir in root_dirs:
                if os.path.exists(os.path.join(root_dir, file_name)):
                    unprocessed_hdf5 = h5py.File(os.path.join(root_dir, file_name))

            data_cube = read.radarDataTensor(unprocessed_hdf5)

            data_cube = data_cube.to('cuda')

            data_cube = radar_process(data_cube, unprocessed_hdf5)

            data_cube = data_cube.to('cpu')

            if plot:

                fig, ax = plt.subplots()
                if len(data_cube.shape) >2:
                    to_plot = data_cube[:,4, ...]
                else:
                    to_plot = data_cube
                ax.imshow(to_plot[0], interpolation='none')

                plt.savefig(f"/home/rtdug/UCT-EEE4022S-2024/sample-figs/{file_name}.png")

                plt.close()

            unprocessed_hdf5.close()

            processed_hdf5.create_dataset(file_name, data=data_cube)
            if index%print_index == 0:
                end_time = time.time()
                print(f" | Time: {(end_time - start_time)*1e3} | {file_name}")

    
    processed_hdf5.close()


def runCnn(model_obj, 
           num_channels, 
           transform,
           in_csvs,
           root_dirs, 
           out_csvs,
           confusion_csvs,
           num_splits, 
           portion_val, 
           num_epochs, 
           learning_rate, 
           test=False, 
           ):


    dataset = GestureDataset(in_csvs[0], root_dirs, transform=transform)

    criterion = nn.CrossEntropyLoss()

    splits = generateSplits(dataset, num_splits, portion_val)

    results_df = pd.DataFrame()
    confusion_df = pd.DataFrame(0, index = dataset.getLabels(), columns = dataset.getLabels())
    test_same_results_df = pd.DataFrame()
    test_same_confusion_df = pd.DataFrame(0, index = dataset.getLabels(), columns = dataset.getLabels())
    test_new_results_df = pd.DataFrame()
    test_new_confusion_df = pd.DataFrame(0, index = dataset.getLabels(), columns = dataset.getLabels())



    for split, (train_ids, val_ids) in enumerate(splits):
        print(f"Split: {split+1}")

        model = model_obj(num_channels, len(dataset.getLabels()))
        optimiser = optim.Adam(model.parameters(), lr=learning_rate)

        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler =  SubsetRandomSampler(val_ids)           
            
        train_loader = DataLoader(dataset, batch_size=16, sampler=train_sampler, num_workers=4)
        val_loader = DataLoader(dataset, batch_size=16, sampler=val_sampler, num_workers=4)

        loss_history = runModel(model, optimiser, train_loader, val_loader, criterion, num_epochs)

        confusion(model, val_loader, confusion_df)
        
        if test:
            testset_same = GestureDataset(in_csvs[1], root_dirs, transform=transform, labels=dataset.getLabels())
            same_loader = DataLoader(testset_same, batch_size=16, num_workers=4)
            same_loss, same_acc = evaluate(model, same_loader, criterion)
            loc = test_same_results_df.shape[0]
            test_same_results_df.loc[loc, "loss"] = same_loss
            test_same_results_df.loc[loc, "acc"] = same_acc
            confusion(model, same_loader, test_same_confusion_df)
        
            testset_new = GestureDataset(in_csvs[2], root_dirs, transform=transform, labels=dataset.getLabels())
            new_loader = DataLoader(testset_new, batch_size=16, num_workers=4)
            new_loss, new_acc = evaluate(model, new_loader, criterion)
            loc = test_new_results_df.shape[0]
            test_new_results_df.loc[loc, "loss"] = new_loss
            test_new_results_df.loc[loc, "acc"] = new_acc
            confusion(model, new_loader, test_new_confusion_df)


        new_column_names = {
            "train_loss": f"train_loss_{split+1}",
            "val_loss": f"val_loss_{split+1}",
            "val_acc": f"val_acc_{split+1}"
        }
        loss_history = loss_history.rename(columns = new_column_names)

        results_df = pd.concat([results_df, loss_history], axis=1)

        

    results_df.to_csv(out_csvs[0], index = False, header=True)     
    confusion_df.to_csv(confusion_csvs[0], index=True, header=True)

    if test:
        test_same_results_df.to_csv(out_csvs[1], index=False, header=True)
        test_same_confusion_df.to_csv(confusion_csvs[1], index=True, header=True)
        test_new_results_df.to_csv(out_csvs[2], index=False, header=True)
        test_new_confusion_df.to_csv(confusion_csvs[2], index=True, header=True)

def generateSplits(dataset, num_splits, portion_val):
    labels = dataset.getLabels() # tuple of labels
    label_indices = dataset.getLabelIndices() # dictionary of tuples of indices for each label
    
    # create iterable which will be filled with releant indices
    splits = []
    for split in range(num_splits):
        splits.append([[], []])
    
    for label in labels:
        indices = list(label_indices[label]) # indices for current label
        random.shuffle(indices) # shuffle
        num_val = int(portion_val*len(indices)) # number to use for validation
        for split in range(num_splits):
            splits[split][0] += indices[:split*num_val] + indices[(split+1)*num_val:] # train set
            splits[split][1] += indices[split*num_val:(split+1)*num_val] # validation set
    
    
    return splits

def runModel(model, optimiser, train_loader, val_loader, criterion, num_epochs):
    columns = ('train_loss', 'val_loss', 'val_acc')
    loss_history = pd.DataFrame(columns=columns)
    iteration = 0
    print_batches = False

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print(f"Epoch: {epoch+1}")
        
        start_time = time.time()
        for batch, data in enumerate(train_loader, 0):
            print_iter = 50
            if iteration%print_iter == 0 and print_batches:
                print(f"Batch: {batch+1}", end="")            
            model.train()
            
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')

            # zero the parameter gradients
            optimiser.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            if iteration%50 == 0:
                train_loss = loss.item()

                val_loss, val_acc = evaluate(model, val_loader, criterion)
                loss_history.loc[len(loss_history.index)] = [train_loss, val_loss, val_acc]
            
            
            iteration+=1
        
            end_time = time.time()
            if iteration%print_iter == 0 and print_batches:
                print(f" | Time: {(end_time - start_time)*1e3}")
            start_time = time.time()


    print('Finished Split')

    return loss_history

def evaluate(model, loader, criterion): 
    model.eval()
    # initialise evaluation parameters
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad(): # evaluating so don't produce gradients
        for data in loader:
            inputs, labels = data
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            # get data from dataloader

            outputs = model(inputs) # predict outputs
            loss = criterion(outputs, labels) # calculate current loss
            _, predicted = torch.max(outputs.data, 1) # calculate predicted data
            total += labels.size(0) # total number of labels in the current batch
            correct += (predicted == labels).sum().item() # number of labels that are correct
            
            running_loss += loss.item() # loss? not 100% sure
        
    # Return mean loss, accuracy
    if len(loader) == 0:
        return_loss = 0
        return_acc = 0
    else:
        return_loss = running_loss/ len(loader)
        return_acc = correct/total
    return return_loss, return_acc

def confusion(model, loader, confusion_df):    
    model.eval()

    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            for i in range(len(labels)):
                confusion_df.iat[int(labels[i]), int(predicted[i])] += 1


if __name__ == "__main__":
    main()