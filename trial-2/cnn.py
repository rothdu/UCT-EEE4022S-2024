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

import cnnmodels as models

import random

class GestureDataset(Dataset):
    """Dataset of radar hand gestures"""

    def __init__(self, csv_file, root_dir, transform=None, label_transform = None):
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
        file_path = os.path.join(self.root_dir, self.gesture_df.iloc[idx, 0]) # path to the hdf5 file

        file_hdf5 = h5py.File(file_path) # load the hdf5 file

        data_cube = read.radarDataTensor(file_hdf5)

        if self.transform:
            data_cube = self.transform(data_cube) # TODO: remove this    
            pass # I'm not using transforms during data loading because I want to collate first and process FFTs on GPU

        label = self.gesture_df.iloc[idx, 1]
        
        if self.label_transform:
            pass # I"m not using label transforms, should already be given in the csv

        label_idnum = torch.tensor(self.labels.index(label), device='cpu')
        range_res = read.rangeResolution(file_hdf5)
        velocity_res = read.velocityResolution(file_hdf5)

        return data_cube, label_idnum, data_cube.shape[0], range_res, velocity_res

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
           input_csv,
           root_dir, 
           out_csv,
           num_splits, 
           portion_val, 
           num_epochs
           ):

    dataset = GestureDataset(input_csv, root_dir)
    criterion = nn.CrossEntropyLoss()

    splits = generateSplits(dataset, num_splits, portion_val)

    out_df = pd.DataFrame()


    for split, (train_ids, val_ids) in enumerate(splits):
        print(f"Split: {split+1}")
        
        model = model_obj(1, len(dataset.getLabels()))
        optimiser = optim.Adam(model.parameters(), lr=0.005)
        
        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler =  SubsetRandomSampler(val_ids)

        def collate_fn(data):
            data_cubes, labels, nums_frames, range_reses, velocity_reses = zip(*data)
            data_cubes = pad_sequence(data_cubes, batch_first=True)
            labels = torch.stack(labels)
            return data_cubes, labels, nums_frames, range_reses, velocity_reses

            
            
        train_loader = DataLoader(dataset, batch_size=4, sampler=train_sampler, collate_fn=collate_fn, num_workers=4)
        val_loader = DataLoader(dataset, batch_size=4, sampler=val_sampler, collate_fn=collate_fn, num_workers=4)

        loss_history = runModel(model, optimiser, train_loader, val_loader, criterion, num_epochs)

        new_column_names = {
            "train_loss": f"train_loss_{split+1}",
            "val_loss": f"val_loss_{split+1}",
            "val_acc": f"val_acc_{split+1}"
        }
        loss_history = loss_history.rename(columns = new_column_names)

        out_df = pd.concat([out_df, loss_history], axis=1)

    out_df.to_csv(out_csv, index = False, header=True)        

def processStackedData(data, process):
    data_cubes, labels, nums_frames, range_res, velocity_res = data
    data_cubes = data_cubes.to('cuda')
    labels = labels.to('cuda')

    inputs = []            
    for sample in range(data_cubes.shape[0]):
        inputs.append(process(data_cubes[sample, :nums_frames[sample], ...], range_res[sample], velocity_res[sample], labels[sample]))
        # TODO: Get rid of the label part
    inputs = torch.stack(inputs)

    return inputs, labels

def evaluate(model, loader, criterion):
    model.eval()
    # initialise evaluation parameters
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad(): # evaluating so don't produce gradients
        for data in loader:
            inputs, labels = processStackedData(data, models.cfarProcess1)
            data = None
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

def runModel(model, optimiser, train_loader, val_loader, criterion, num_epochs):
    columns = ('train_loss', 'val_loss', 'val_acc')
    loss_history = pd.DataFrame(columns=columns)
    iteration = 0

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print(f"Epoch: {epoch+1}")
        
        for batch, data in enumerate(train_loader, 0):
            print(f"Batch: {batch+1}", end="")
            start_time = time.time()
            
            model.train()
            
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = processStackedData(data, models.cfarProcess1)

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
            print(f" | Time: {(end_time - start_time)*1e3}")


    print('Finished Split')

    return loss_history

def main():
    torch.set_default_device('cuda')

    model = models.CfarModel1
    input_csv = "gestures.csv"
    root_dir = "data"
    out_csv = "results.csv"
    num_splits = 1
    portion_val = 0.02
    num_epochs = 15

    # model = models.CfarModel1
    # input_csv = "smallset.csv"
    # root_dir = "data"
    # out_csv = "na.csv"
    # num_splits = 1
    # portion_val = 0
    # num_epochs = 1
    runCnn(model, input_csv, root_dir, out_csv, num_splits, portion_val, num_epochs)

if __name__ == "__main__":
    main()