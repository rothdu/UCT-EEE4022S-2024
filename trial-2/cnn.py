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

import random



class GestureDataset(Dataset):
    """Dataset of radar hand gestures"""

    def __init__(self, csv_file, root_dir, labels = None, transform=None, label_transform = None):
        """
        Arguments:
            csv_file (string) Path to the csv file with ground truth information
            root_dir (string) Directory with the data items
            transform (callable, optional): Optional transform to be applied on a sample
        """

        self.gesture_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label_transform = label_transform
        if labels:
            self.labels = labels
        else:
            self.labels = tuple(self.gesture_df["label"].unique())
        
        # Create a dictionary. 'label': (tuple of all indices where that label occurs in the dataset)
        # Useful for creating training-validation splits later on
        self.label_indices = {}
        for label in self.labels:
            self.label_indices[label] = tuple([i for i in range(len(self.gesture_df["label"])) if self.gesture_df["label"][i] == label])
    
    def __len__(self):
        return len(self.gesture_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # some of this could be done with transforms... but I am doing it here for now
        data_hdf5 = h5py.File(os.fsencode("processed_data/gestures.hdf5")) # load the hdf5 file
        item_name = self.gesture_df.iloc[idx, 0]

        data_cube = torch.as_tensor(data_hdf5[item_name][:], device='cpu')

        source_hdf5 = h5py.File(os.path.join(self.root_dir, item_name))
        
        
        if self.transform:
            data_cube = self.transform(data_cube, source_hdf5)

        label = self.gesture_df.iloc[idx, 1]
        
        if self.label_transform:
            label = self.label_transform(label)

        label_idnum = torch.as_tensor(self.labels.index(label), device='cpu')
        
        return data_cube, label_idnum

    def getLabels(self):
        return self.labels
    
    def getLabelIndices(self):
        return self.label_indices.copy()

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

def runCnn(model_obj, 
           num_channels, 
           input_csv,
           root_dir, 
           transform,
           out_csv,
           confusion_csv,
           num_splits, 
           portion_val, 
           num_epochs, 
           learning_rate, 
           test, 
           process
           ):


    dataset = GestureDataset(input_csv, root_dir, transform=transform)

    criterion = nn.CrossEntropyLoss()

    splits = generateSplits(dataset, num_splits, portion_val)

    results_df = pd.DataFrame()
    confusion_df = pd.DataFrame(0, index = dataset.getLabels(), columns = dataset.getLabels())
    test_results_df = pd.DataFrame()
    test_same_confusion_df = pd.DataFrame(0, index = dataset.getLabels(), columns = dataset.getLabels())
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
        print(loss_history)

        confusion(model, val_loader, confusion_df)
        
        if test:
            testset_new = GestureDataset("test_new_ge20_tiny.csv", "data-test-new", transform=transform, labels=dataset.getLabels())
            processInputs("test_new_ge20_tiny.csv", "data-test-new", process)

            new_loader = DataLoader(testset_new, batch_size=16, num_workers=4)
            new_loss, new_acc = evaluate(model, new_loader, criterion)
            test_results_df.loc[0, "new_loss"] = new_loss
            test_results_df.loc[0, "new_acc"] = new_acc

            confusion(model, new_loader, test_new_confusion_df)


            testset_same = GestureDataset("test_same_ge20_tiny.csv", "data-test-same", transform=transform, labels=dataset.getLabels())
            processInputs("test_same_ge20_tiny.csv", "data-test-same", process)
            same_loader = DataLoader(testset_same, batch_size=16, num_workers=4)
            same_loss, same_acc = evaluate(model, same_loader, criterion)
            test_results_df.loc[0, "same_loss"] = same_loss
            test_results_df.loc[0, "same_acc"] = same_acc

            confusion(model, same_loader, test_same_confusion_df)


        new_column_names = {
            "train_loss": f"train_loss_{split+1}",
            "val_loss": f"val_loss_{split+1}",
            "val_acc": f"val_acc_{split+1}"
        }
        loss_history = loss_history.rename(columns = new_column_names)

        results_df = pd.concat([results_df, loss_history], axis=1)

        

    results_df.to_csv(out_csv, index = False, header=True)     
    confusion_df.to_csv(confusion_csv, index=True, header=True)

    if test:
        test_results_df.to_csv("test_" + out_csv, index=False, header=True)
        test_same_confusion_df.to_csv("test_same_" + confusion_csv, index=True, header=True)
        test_new_confusion_df.to_csv("test_new_" + confusion_csv, index=True, header=True)



def evaluate(model, loader, criterion): 
    # Have to pass these labels in separately because my dynamic label grabbing messes things up
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



def runModel(model, optimiser, train_loader, val_loader, criterion, num_epochs):
    columns = ('train_loss', 'val_loss', 'val_acc')
    loss_history = pd.DataFrame(columns=columns)
    iteration = 0

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print(f"Epoch: {epoch+1}")
        
        start_time = time.time()
        for batch, data in enumerate(train_loader, 0):
            print_iter = 50
            if iteration%print_iter == 0:
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
            if iteration%print_iter == 0:
                print(f" | Time: {(end_time - start_time)*1e3}")
            start_time = time.time()


    print('Finished Split')

    return loss_history


def processInputs(input_csv, root_dir, radar_process, plot=False):
    
    processed_path = os.fsencode("processed_data/gestures.hdf5")

    if os.path.exists(processed_path):
        os.remove(processed_path)
    

    gesture_df = pd.read_csv(input_csv)

    processed_hdf5 = h5py.File(processed_path, "w")


    for index, row in gesture_df.iterrows():
        print(f"Pre-processing item: {index}", end = "")
        start_time = time.time()

        file_name = row["file_name"]


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

        end_time = time.time()
        print(f" | Time: {(end_time - start_time)*1e3} | {file_name}")

    processed_hdf5.close()

def main():
    torch.set_default_device('cuda')
    input_csv = "gestures_ge20_tiny.csv"
    root_dir = "data"
    num_splits = 1
    portion_val = 0.1
    num_epochs = 150
    learning_rate = 0.0002
    num_channels = 1
    transform = processes.cfarCrop2
    test = True

    out_csv = "results.csv"
    confusion_csv = "confusion.csv"
    process = processes.handLocateProcess1
    model = cnns.NewModel1
    transform = None
    processInputs(input_csv, root_dir, process)
    runCnn(model, num_channels, input_csv, root_dir, transform, out_csv, confusion_csv, num_splits, portion_val, num_epochs, learning_rate, test, process)

    

    # processInputs("test_same_ge20.csv", "data-test-same", processes.rangeDopplerProcess2, True)
    #TODOTODOTODO:  Check all the  todos lol
    # NOTE: Check the max pooling on the md model

    # out_csv = "results_cfar_2d_clutter.csv"
    # confusion_csv = "confusion_cfar_2d_clutter.csv"
    # model = cnns.CfarModel1
    # process = processes.cfarProcess3
    # processInputs(input_csv, root_dir, process)
    # runCnn(model, num_channels, input_csv, root_dir, transform, out_csv, confusion_csv, num_splits, portion_val, num_epochs, learning_rate, test, process)

    # out_csv = "results_cfar_3d_clutter.csv"
    # confusion_csv = "confusion_cfar_3d_clutter.csv"
    # model = cnns.CfarModel3
    # process = processes.cfarProcess4
    # processInputs(input_csv, root_dir, process)
    # runCnn(model, num_channels, input_csv, root_dir, transform, out_csv, confusion_csv, num_splits, portion_val, num_epochs, learning_rate, test, process)

    # out_csv = "results_cfar_3d_noclutter.csv"
    # confusion_csv = "confusion_cfar_3d_noclutter.csv"
    # model = cnns.CfarModel3
    # process = processes.cfarProcess5
    # processInputs(input_csv, root_dir, process)
    # runCnn(model, num_channels, input_csv, root_dir, transform, out_csv, confusion_csv, num_splits, portion_val, num_epochs, learning_rate, test, process)

    # out_csv = "results_rd_3d_clutter.csv"
    # confusion_csv = "confusion_rd_clutter.csv"
    # model = cnns.CfarModel3
    # process = processes.rangeDopplerProcess1
    # processInputs(input_csv, root_dir, process)
    # runCnn(model, num_channels, input_csv, root_dir, transform, out_csv, confusion_csv, num_splits, portion_val, num_epochs, learning_rate, test, process)

    # out_csv = "results_rd_3d_noclutter.csv"
    # confusion_csv = "confusion_rd_3d_noclutter.csv"
    # model = cnns.CfarModel3
    # process = processes.rangeDopplerProcess2
    # processInputs(input_csv, root_dir, process)
    # runCnn(model, num_channels, input_csv, root_dir, transform, out_csv, confusion_csv, num_splits, portion_val, num_epochs, learning_rate, test, process)


if __name__ == "__main__":
    main()