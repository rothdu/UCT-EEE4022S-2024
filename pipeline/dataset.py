from torch.utils.data import Dataset
import torch
import pandas as pd
import h5py
import os

class GestureDataset(Dataset):
    """Dataset of radar hand gestures"""
    def __init__(self, csv_file, root_dirs, labels = None, transform=None, label_transform = None):
        """
        Arguments:
            csv_file (string) Path to the csv file with ground truth information
            root_dir (string) Directory with the data items
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.test = False
        self.gesture_df = pd.read_csv(csv_file)
        self.root_dirs = root_dirs
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
        data_hdf5 = h5py.File(os.fsencode("processed-data/gestures.hdf5")) # load the hdf5 file
        item_name = self.gesture_df.iloc[idx, 0]

        data_cube = torch.as_tensor(data_hdf5[item_name][:], device='cpu')
        for root_dir in self.root_dirs:
            file_path = os.path.join(root_dir, item_name)
            if os.path.exists(file_path):
                source_hdf5 = h5py.File(file_path)
        
        
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
    
    def setTest(self):
        self.test = True